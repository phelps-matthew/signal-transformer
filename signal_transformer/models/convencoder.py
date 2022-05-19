import math
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        kernel_sizes: Union[list, tuple] = (3, 3, 3, 3, 3, 3),
        strides: Union[list, tuple] = (3, 2, 2, 2, 2, 2),
        channel_dropout: float = 0.0,
        projection_head: bool = False,
    ):
        """
        Args:
            in_channels: input channels, e.g. two for RF quadrature
            out_channels: number of conv1d output channels at each layer; last layer must match
            transformer hidden size
            kernel_sizes: kernel widths of 1d conv layers; BENDER uses odd (centerable) kernel widths
            strides: strides of 1d conv layers; product of strides approximates net downsampling
            channel_dropout: probability of dropping an entire channel along conv feature maps
            projection_head:

        Note:
            Given sequence length L_in, the output of each layer is given by
            L_out = floor(1 + (L_in + 2 * pad - kernel_size) / stride). For
            L_in >> kernel_size, to cover convolving all input elements of the sequence
            we require (L_in + 2 * pad) % stride = 0. At most, L_in % stride = stride - 1.
            Thus we can ensure kernel and stride params cover sampling all input if we
            set 2 * pad = stride - 1, or pad = (stride - 1) / 2.  Since pad must be an
            integer we require pad = ceil((stride - 1) / 2) = ceil(stride / 2) - 1
            = floor(stride / 2) = stride // 2.
        """
        super(ConvEncoder, self).__init__()
        assert len(strides) == len(kernel_sizes)

        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.encoder = nn.Sequential()

        for i, (k, stride) in enumerate(zip(kernel_sizes, strides)):
            self.encoder.add_module(
                f"encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.Dropout2d(channel_dropout),
                    nn.GroupNorm(out_channels // 2, out_channels),
                    nn.GELU(),
                ),
            )
            in_channels = out_channels

        if projection_head:
            self.encoder.add_module(
                "projection_1",
                nn.Sequential(
                    nn.Conv1d(out_channels, out_channels, 1),
                    nn.Dropout2d(channel_dropout * 2),
                    nn.GroupNorm(out_channels // 2, out_channels),
                    nn.GELU(),
                ),
            )

    def info(self, sequence_len=None, sample_freq=None):
        """Return information regarding receptive field, downsample rate, sample
        overlap, and number of encoded samples per trial."""
        # recursion relation for receptive field, see
        # https://distill.pub/2019/computing-receptive-fields/
        kernel_sizes = list(reversed(self.kernel_sizes))
        strides = list(reversed(self.strides))
        rf = 1
        for k, s in zip(kernel_sizes, strides):
            rf = s * rf + (k - s)
        desc = "Receptive field: {} samples".format(rf)

        if sample_freq is not None:
            desc += ", {:.2f} seconds".format(rf / sample_freq)
        ds_factor = np.prod(self.strides)
        desc += " | Asymptotic downsample factor: {}".format(ds_factor)

        if sample_freq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sample_freq / ds_factor)
        desc += " | Overlap: {} samples".format(rf - ds_factor)

        if sequence_len is not None:
            enc_seq_len = self.num_encoded(sequence_len)
            desc += f" | Encoded samples: {enc_seq_len}".format(enc_seq_len)

        return desc

    def num_encoded(self, sequence_len):
        """Given input sequence length, computes number of encoded samples based on
        stride downsamplings"""
        # For sequence_len >> stride and kernel size, the downsample factor for each
        # layer goes to ceil(sequence_len / stride). For the default strides and kernel
        # sizes given in ConvEncoder, this approximation becomes exact.
        L_in = sequence_len
        for factor in self.strides:
            L_in = math.ceil(L_in / factor)
        return L_in

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":

    length = int(1e3)
    input = torch.rand(1, 1, length)
    encoder = ConvEncoder(1)
    print(encoder.info(sequence_len=length))
    print(encoder)
    print(encoder(input).shape)
