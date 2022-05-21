from signal_transformer.utils import (
    make_mask,
    _make_span_from_seeds,
    compute_mask_indices,
)
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
        l2_activation_convencoder: factor applied to convencoder regularizer, computed as
            mean squared activation penalty
        multi_gpu: invoke torch.nn.DataParallel
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

    @staticmethod
    def mask_pct(mask):
        """mask coming from forward output"""
        return mask.float().mean().item()

    def generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against.

        Returns:
            torch.float32 of shape (N, L, n_negatives, C)
        """
        N, C, L = z.shape
        # flatten over features (channels) and batch size
        z_k = z.permute([0, 2, 1]).reshape(-1, C)
        # intialize index tensor
        negative_inds = torch.empty(N, L, self.num_negatives).long()
        # create tensor of ones, with zero diagonal. this permits us to exclude the self
        # token (it is explicitly used in calculate similarity)
        ind_weights = torch.ones(L, L) - torch.eye(L)
        # for each batch, randomly sample non-negative elements from rows of ind_weights.
        # we add i * L to get proper indices when we reshape below
        with torch.no_grad():
            for i in range(N):
                negative_inds[i] = (
                    torch.multinomial(ind_weights, self.num_negatives) + i * L
                )

        # evaluate z_k at negative indices, and reshape to desired output
        z_k = z_k[negative_inds.view(-1)].view(N, L, self.num_negatives, C)
        return z_k, negative_inds

    def calculate_similarity(self, z, c, negatives):
        """
        Args:
            z: unmasked feature vector sequence (N, C, L)
            c: transformer output feature vector sequence (N, C, L+1)
            negatives: (N, L, n_negatives, C)

        Returns:
            torch.float32 of shape (N * L, n_negatives + 1)
        """
        # remove start token, permute and add dimension. z and c have shape (N, L, 1, C)
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        # broadcast c, compute equivalence of entire feature vector (dim=-1) matching
        negative_in_target = (c == negatives).all(-1)

        # add current (self) token to list of distractors, (N, L, n_negatives + 1, C)
        targets = torch.cat([z, negatives], dim=-2)

        # compute cosine similarity between c and all distractors, (N, L, n_negatives + 1)
        logits = F.cosine_similarity(c, targets, dim=-1) / self.temp
        # ignoring self token, replace all exact matches with -inf so as to not contribute
        # to constastive loss denominator
        if negative_in_target.any():
            logits[:, :, 1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    @staticmethod
    def contrastive_accuracy(logits):
        labels = torch.zeros(logits, device=logits.device, dtype=torch.long)
        # return StandardClassification._simple_accuracy([labels], logits)
        return labels

    def calculate_loss(self, logits, unmasked_z):
        # The 0'th index is the correct position
        labels = torch.zeros(logits.shape, device=logits.device, dtype=torch.long)
        # Note that loss_fn here integrates the softmax as per the normal classification
        # pipeline (leveraging logsumexp)
        l2_activation_loss = self.beta * unmasked_z.pow(2).mean()
        return self.loss_fn(logits, labels) + l2_activation_loss

    def forward(self, x):
        z = self.convencoder(x)
        unmasked_z = z.clone()

        N, C, L = z.shape

        if self.training:
            mask = make_mask((N, L), self.mask_rate, L, self.mask_span)
        # during eval, use half the mask rate and evenly space masks
        else:
            mask = torch.zeros((N, L), requires_grad=False, dtype=torch.bool)
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

        # generate negative distractors for each transformed token
        negatives, _ = self.generate_negatives(unmasked_z)

        # calculate cosine similarity between transfored features and 
        # distractors as logits (i.e. inputs into softmax-like loss)
        logits = self.calculate_similarity(unmasked_z, c, negatives)
        # fmt: off
        import ipdb; ipdb.set_trace(context=30)  # noqa
        # fmt: on
        self.calculate_loss(logits, unmasked_z)
        return logits, unmasked_z, mask, c


if __name__ == "__main__":
    from signal_transformer.models import ConvEncoder, Transformer

    encoder = ConvEncoder(1)
    tx = Transformer(512)
    model = ContrastiveSSL(encoder, tx, num_negatives=5, mask_rate=0.01, mask_span=5)
    model.eval()
    L = int(96 * 1e3)
    print(encoder.info(L))
    print(model.info(L))
    print(model(torch.randn(2, 1, L)))
