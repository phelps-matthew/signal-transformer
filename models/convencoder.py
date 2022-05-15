import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Union


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        kernel_sizes: Union[list, tuple] = (3, 3, 3, 3, 3, 3),
        strides: Union[list, tuple] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
        projection_head: bool = False,
    ):
        """
        Args:
            in_channels: input channels. two for quadrature
            out_channels: number of conv1d output channels at each layer, matches transformer hidden size
            kernel_sizes: kernel widths of 1d conv layers, suggested odd sized to be centerable
            strides: strides of 1d conv layers; downsamplings
            dropout:
            projection_head: 

        Note:
            Given sequence length L_in, the output of each layer is given by
            L_out = floor(1 + (L_in + 2 * pad - kernel_size) / stride). For
            L_in >> kernel_size, to cover convolving all input elements of the sequence
            we require (L_in + 2 * pad) % stride = 0. At most, L_in % stride = stride - 1.
            Thus we can ensure kernel and stride params evenly sample all input if we
            set 2 * pad = stride - 1, or pad = (stride - 1) / 2.  Since pad must be an
            integer we require pad = ceil((stride - 1) / 2) = ceil(stride / 2) - 1
            = floor(stride / 2) = stride // 2.
        """
        super(ConvEncoder, self).__init__()
        assert len(strides) == len(kernel_sizes)

        self.out_channels = out_channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.encoder = nn.Sequential()

        for i, (k, stride) in enumerate(zip(kernel_sizes, strides)):
            self.encoder.add_module(
                "encoder_{}".format(i),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.Dropout2d(dropout),
                    nn.GroupNorm(out_channels // 2, out_channels),
                    nn.GELU(),
                ),
            )
            in_features = out_channels

        if projection_head:
            self.encoder.add_module(
                "projection-1",
                nn.Sequential(
                    nn.Conv1d(in_features, in_features, 1),
                    nn.Dropout2d(dropout * 2),
                    nn.GroupNorm(in_features // 2, in_features),
                    nn.GELU(),
                ),
            )

    def description(self, sfreq=None, sequence_len=None):
        kernel_sizes = list(reversed(self.kernel_sizes))[1:]
        strides = list(reversed(self.strides))[1:]

        rf = self.kernel_sizes[-1]
        for k, s in zip(kernel_sizes, strides):
            rf = rf if k == 1 else (rf - 1) * s + 2 * (k // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self.strides)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, sequence_len):
        """Overall downsampling factor of ConvEncoder"""
        # For sequence_len >> stride and kernel size, the downsample factor for each
        # layer goes to ceil(sequence_len / stride). For the default strides and kernel
        # sizes given in ConvEncoder, this approximation becomes exact.
        L_in = sequence_len
        for factor in self.strides:
            L_in = math.ceil(L_in / factor)
        return L_in

    def forward(self, x):
        return self.encoder(x)

class ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        in_features,
        encoder_h=256,
        enc_width=(3, 2, 2, 2, 2, 2),
        dropout=0.0,
        projection_head=False,
        enc_downsample=(3, 2, 2, 2, 2, 2),
    ):
        super(ConvEncoderBENDR, self).__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features
        self.encoder_h = encoder_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable convolutions make life simpler
        enc_width = [e if e % 2 else e + 1 for e in enc_width]
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(
                "Encoder_{}".format(i),
                nn.Sequential(
                    nn.Conv1d(
                        in_features,
                        encoder_h,
                        width,
                        stride=downsample,
                        padding=width // 2,
                    ),
                    nn.Dropout2d(dropout),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            in_features = encoder_h

        if projection_head:
            self.encoder.add_module(
                "projection-1",
                nn.Sequential(
                    nn.Conv1d(in_features, in_features, 1),
                    nn.Dropout2d(dropout * 2),
                    nn.GroupNorm(in_features // 2, in_features),
                    nn.GELU(),
                ),
            )

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = math.ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)


class _Hax(nn.Module):
    """T-fixup assumes self-attention norms are removed"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


if __name__ == "__main__":
    #encoder = ConvEncoder(1, kernel_sizes=(3, 2, 2, 2, 2, 2))
    encoder = ConvEncoder(1)
    bencoder = ConvEncoderBENDR(1, enc_width=(3, 3, 3, 3, 3, 3))
    print(encoder.description(sequence_len=1e3))
    print(bencoder.description(sequence_len=1e3))
    print(encoder)
    print(bencoder)
