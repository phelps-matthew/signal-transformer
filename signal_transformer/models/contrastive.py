from signal_transformer.utils import make_mask, _make_span_from_seeds
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveSSL(nn.Module):
    """
    Implements contrastive semi-supervised learning using a convencoder and a transformer.
    Influenced by wav2vec 2.0 and BENDR.

    Args:
        convencoder: convolutional encoder model that yield signal representations
        transformer: transformer model applying MHSA to conencoder inputs
        mask_rate:
        mask_span:
        learning_rate:
        temp: temperature factor in contrastive loss (non-negative)
        enc_feat_l2:
        multi_gpu: invoke torch.nn.DataParallel
        l2_weight_decay:
        unmasked_negative_frac:
        encoder_grad_frac:
        num_negatives: number of distractors uniformly sampled from input sequence
    """

    def __init__(
        self,
        convencoder: Callable[..., torch.Tensor],
        transformer: Callable[..., torch.Tensor],
        mask_rate: float = 0.1,
        mask_span: int = 6,
        temp: float = 0.5,
        enc_feat_l2: float = 0.001,
        multi_gpu: bool = False,
        unmasked_negative_frac: float = 0.25,
        encoder_grad_frac: float = 1.0,
        num_negatives: int = 100,
        loss_fn: Callable[..., torch.Tensor] = torch.nn.CrossEntropyLoss(),
    ):
        super(ContrastiveSSL, self).__init__()
        self._enc_downsample = convencoder.num_encoded
        self.predict_length = mask_span
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.beta = enc_feat_l2
        self.start_token = getattr(transformer, "start_token", None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives
        self.loss_fn = loss_fn

        self.convencoder = convencoder
        self.transformer = transformer
        if multi_gpu:
            self.convencoder = nn.DataParallel(self.convencoder)
            self.transformer = nn.DataParallel(self.transformer)
        if encoder_grad_frac < 1:
            convencoder.register_backward_hook(
                lambda module, in_grad, out_grad: tuple(
                    encoder_grad_frac * ig for ig in in_grad
                )
            )

    def info(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples,
            self.mask_span,
            self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span),
        )
        return desc

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        N, C, L = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, C)
        negative_inds = torch.empty(N, L, self.num_negatives).long()
        ind_weights = torch.ones(L, L) - torch.eye(L)
        with torch.no_grad():
            # candidates = torch.arange(L).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            for i in range(N):
                negative_inds[i] = (
                    torch.multinomial(ind_weights, self.num_negatives) + i * L
                )
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

        z_k = z_k[negative_inds.view(-1)].view(
            N, L, self.num_negatives, C
        )
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([z, negatives], dim=-2)

        logits = F.cosine_similarity(c, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, *inputs):

        z = self.convencoder(inputs[0])
        unmasked_z = z.clone()

        N, C, L = z.shape
        # fmt: off
        import ipdb; ipdb.set_trace(context=30)  # noqa
        # fmt: on

        if self.training:
            mask = make_mask(
                (N, L), self.mask_rate, L, self.mask_span
            )
        else:
            mask = torch.zeros(
                (N, L), requires_grad=False, dtype=torch.bool
            )
            half_avg_num_seeds = max(1, int(L * self.mask_rate * 0.5))
            if L <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[
                :,
                _make_span_from_seeds(
                    (L // half_avg_num_seeds)
                    * np.arange(half_avg_num_seeds).astype(int),
                    self.mask_span,
                ),
            ] = True

        c = self.transformer(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(unmasked_z)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives)
        return logits, unmasked_z, mask, c

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def contrastive_accuracy(inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # return StandardClassification._simple_accuracy([labels], logits)
        return labels

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        # The 0'th index is the correct position
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note that loss_fn here integrates the softmax as per the normal classification
        # pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()


if __name__ == "__main__":
    from signal_transformer.models import ConvEncoder, Transformer

    encoder = ConvEncoder(1)
    tx = Transformer(512)
    model = ContrastiveSSL(encoder, tx)
    print(encoder.info(int(1e3)))
    print(model.info(int(1e3)))
    print(model(torch.randn(2, 1, 1000)))
