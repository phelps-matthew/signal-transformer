import copy

import numpy as np
import torch
from torch import nn


def _make_span_from_seeds(seeds, span, total=None):
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


def _make_mask(shape, p, total, span, allow_no_inds=False):
    # num_mask_spans = np.sum(np.random.rand(total) < p)
    # num_mask_spans = int(p * total)
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask


class Permute(nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class Transformer(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_feedforward=3076,
        heads=8,
        layers=8,
        dropout=0.15,
        activation="gelu",
        position_encoder=25,
        layer_drop=0.0,
        mask_p_t=0.1,
        mask_p_c=0.004,
        mask_t_span=6,
        mask_c_span=64,
        start_token=-5,
        finetuning=False,
    ):
        """
        Args:
            in_features: dimension of single sample input feature vector
            hidden_feedforward: dimension of fc layers in transformer encoder
            heads: number of transformer encoder heads
            layers: number of transformer encoder blocks
            dropout: dropout probability as applied to transformer encoder, and input conditioning
            activation: activate unit in transformer encoder layer
            position_encoder: group conv kernel size in positional encoding; zero to disable
            layer_drop: probability of dropping any individual layer within transformer encoder
            mask_p_t:
            mask_p_c:
            mask_t_span:
            mask_c_span:
            start_token: [CLS] token initialization value
            finetuning: create finetuning masks in forward function
        """
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        # group conv position encoding
        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(
                in_features,
                in_features,
                kernel_size=position_encoder,
                padding="same",
                groups=16,
            )
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        # upsample to dim = 3 * in_features using 1x1 conv
        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),  # normalizes over last dimension
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        # base encoder block with LayerNorm removed viz. T-fixup
        encoder = nn.TransformerEncoderLayer(
            d_model=in_features * 3,
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,
            activation=activation,
        )
        encoder.norm1 = nn.Identity()
        encoder.norm2 = nn.Identity()
        # repeat base encoder block to form n layers
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder) for _ in range(layers)]
        )

        # initialize replacement vector with 0's
        self.mask_replacement = nn.Parameter(
            torch.normal(0, in_features ** (-0.5), size=(in_features,)),
            requires_grad=True,
        )

        # downsample dim = in_features using 1x1 conv
        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)

        # initialize transforer fc layers according to T-fixup
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # T-fixup initialization
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )

    def forward(self, x, mask_t=None, mask_c=None):
        bs, feat, seq = x.shape

        if self.training and self.finetuning:
            if mask_t is None and self.p_t > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)

        # multi-gpu workaround, wastes memory (from official dn3/trainable/layers)
        x = x.clone()

        if mask_t is not None:
            x.transpose(2, 1)[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(
                x.device
            ).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)

        return self.output_layer(x.permute([1, 2, 0]))


if __name__ == "__main__":
    model = Transformer(512)
    input = torch.rand(2, 512, 12)
    print(model(input).shape)
    print(model)
