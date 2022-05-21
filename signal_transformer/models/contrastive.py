from typing import Callable, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from signal_transformer.utils import _make_span_from_seeds, make_mask


class ContrastiveSSL(nn.Module):
    """
    Implements contrastive semi-supervised learning (pretraining).

    Influenced by wav2vec 2.0 and BENDR.

    Args:
        convencoder: convolutional encoder model yielding base signal feature vectors
        transformer: transformer model, MHSA provides contexualization among feature
            vectors
        mask_rate: probability of token lying at beginning of contiguous mask
        mask_span: token span of a contiguous mask
        temp: temperature divisor in cosine similarity loss (non-negative)
        l2_feature_encoder: regularization multiplicand applied to convencoder
            regularizer, computed as mean squared activation penalty
        multi_gpu: invoke torch.nn.DataParallel
        num_negatives: number of distractors uniformly sampled from tokenized sequence.
            distractors are sampled for each token independently
        encoder_grad_frac: gradient supression in convencoder
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
        num_negatives: int = 100,
        encoder_grad_frac: float = 1.0,
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
        self.num_negatives = num_negatives
        self.loss_fn = nn.CrossEntropyLoss()

        self.convencoder = convencoder
        self.transformer = transformer
        if multi_gpu:
            self.convencoder = nn.DataParallel(self.convencoder)
            self.transformer = nn.DataParallel(self.transformer)
        # how to module encoder gradients?
        if encoder_grad_frac < 1:
            convencoder.register_backward_hook(
                lambda module, in_grad, out_grad: tuple(
                    encoder_grad_frac * ig for ig in in_grad
                )
            )

    def _make_eval_mask(self, shape: Tuple[int, int]) -> torch.Tensor:
        """Create evenly spaced mask spans based on half of training mask rate.
        Ugly, refactor?

        Args:
            shape: tuple of two elements defining mask shape, (N, L)
        Returns:
            mask: torch.float32 (N, L)
        """
        N, L = shape
        mask = torch.zeros((N, L), requires_grad=False, dtype=torch.bool)
        half_avg_num_seeds = max(1, int(L * self.mask_rate * 0.5))
        if L <= self.mask_span * half_avg_num_seeds:
            raise ValueError("Masking the entire span, pointless.")
        seeds = (L // half_avg_num_seeds) * np.arange(half_avg_num_seeds).astype(int)
        mask_indices = _make_span_from_seeds(seeds, self.mask_span)
        mask[:, mask_indices] = True
        return mask

    def info(self, sequence_len: int) -> str:
        """Return information string regarding mask span, rate, and expectation of total
        masked elements.
        """
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "Samples: {} | Mask span: {} | Mask rate: {} | E[masked] ~= {}".format(
            encoded_samples,
            self.mask_span,
            self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span),
        )
        return desc

    def sample_negatives(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate negative samples to compare each sequence location against.

        Args:
            z: unmasked feature vector sequence (N, C, L)

        Returns:
            torch.float32 of shape (N, L, n_negatives, C)
        """
        N, C, L = z.shape
        # flatten over features (channels) and batch size
        z_k = z.permute([0, 2, 1]).reshape(-1, C)
        # intialize index tensor
        negative_inds = torch.empty(N, L, self.num_negatives).long()
        # create tensor of ones, with zero diagonal. this permits us to exclude the self
        # token (it is explicitly added in compute_logits)
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

    def compute_logits(
        self, z: torch.Tensor, c: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits of feature vector and distractors using cosine similarity.

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

        # if transformer output maches negative distactor exactly, we must explicitly
        # handle such elements to avoid division by zero, so here we gather such indices
        # broadcast c, compute potential equivalences of entire feature vectors (dim=-1)
        negative_in_target = (c == negatives).all(-1)

        # include own (self) token within list of distractors, (N, L, n_negatives + 1, C)
        targets = torch.cat([z, negatives], dim=-2)

        # compute cosine similarity between c and all distractors, (N, L, n_negatives + 1)
        logits = F.cosine_similarity(c, targets, dim=-1) / self.temp

        # ignoring self token, replace all exact matches with -inf so as to not contribute
        # to constastive loss denominator
        if negative_in_target.any():
            logits[:, :, 1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    @staticmethod
    def contrastive_accuracy(logits: torch.Tensor) -> float:
        """
        Compute accuracy of highest logit scores of transformed features matching
        encoded feature vectors.

        Args:
            logits: cosine similarity outputs, (N * L, n_negatives + 1)
        Returns:
            torch.tensor scalar (torch.float32)
        """
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        preds = logits.argmax(dim=-1)
        accuracy = preds.eq(targets).float().mean().item()
        return accuracy

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate crossentropy using logits from negative distractors.

        Args:
            logits: cosine similarity outputs, (N * L, n_negatives + 1)
        Returns:
            torch.tensor scalar (torch.float32)
        """
        # why use long instead of int?
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # `CrossEntropyLoss` applies log(softmax) internally
        contrastive_loss = self.loss_fn(logits, targets)
        return contrastive_loss

    def feature_activation_loss(self, unmasked_z: torch.Tensor) -> torch.Tensor:
        """Compute L2 feature activation loss (from convencoder)

        Args:
            unmasked_z: unmasked feature vector sequence, (N, C, L)
        Returns:
            torch.tensor scalar (torch.float32)
        """
        return self.beta * unmasked_z.pow(2).mean()

    def forward(self, x):
        # forward raw signal inputs through convencoder to form feature vectors
        z = self.convencoder(x)
        unmasked_z = z.clone()
        N, C, L = z.shape
        # compute masks, using half maks rate and even spacing for eval
        if self.training:
            mask = make_mask((N, L), self.mask_rate, L, self.mask_span)
        else:
            mask = self._make_eval_mask((N, L))
        # pass feature vectors and masks into transformer
        c = self.transformer(z, mask)
        # generate negative distractors for each transformed token
        negatives, _ = self.sample_negatives(unmasked_z)
        # calculate cosine similarity between transformed features and
        # distractors as logits (i.e. inputs into softmax-like loss)
        logits = self.compute_logits(unmasked_z, c, negatives)
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
