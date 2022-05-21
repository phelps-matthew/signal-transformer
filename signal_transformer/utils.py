import numpy as np
import torch
from typing import Optional, Tuple

# TODO: These masking procedures seem unecessarily convoluted and slow, perhaps can simplify?


def _make_span_from_seeds(seeds, span, total=None):
    """Helper function for `_make_mask`. Create mask index array by masking selected
    indices (seeds) up to contiguous spans.

    Args:
        seeds: selected indices to apply contiguous masking
        span: span of a contiguous mask
        total: last index by which masking is to be applied (last_index = total - 1)

    Returns:
        mask index array, np.int32 of shape
    """
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)


# TODO: Appears to be inverse-masking contiguous arrays. Issue?


def make_mask(shape, p, total, span, allow_no_inds=False):
    """Create mask of contiguous sub-masks.

    Args:
        shape: shape of mask array, only supports tensors of dims=2
        p: probability of masking contiguous sequence of span `span`
        total: last index by which masking is to be applied
        span: span of a contiguous mask
        allow_no_inds: permit returning array with no net masking

    Returns:
        mask tensor, torch.bool of shape (shape)
    """
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)
    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and p > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < p)[0]
        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True
    return mask


def compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device,
    attention_mask: Optional[torch.tensor] = None,
    min_masks: int = 0,
) -> torch.tensor:
    """
    Computes random mask spans for a given shape.
    <https://github.com/huggingface/transformers/blob/0f5488f79fabfaa0c49226c96409ab11d661396b/src/transformers/models/wav2vec2/modeling_wav2vec2.py>

    Args:
        shape: shape for which to compute masks. tuple of size 2 where first dimension
            is batch size and second is length of axis to span (tokens)
        mask_prob: percentage of whole axis (between 0 and 1) which will be masked. the 
            number of independently generated masks of spans of length `mask_length` is
            computed by `mask_prob*shape[1]/mask_length`. note that due to overlaps,
            `mask_prob` is an upper bound and the actual percentage will be smaller.
        mask_length: size of the mask
        attention_mask: a (right-padded) attention mask which independently shortens the
            feature axis of each batch dimension.
        min_masks: minimum number of masked spans
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            "`mask_length` has to be smaller than `sequence_length`, but got"
            + f"`mask_length`: {mask_length} and `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(
        mask_prob * sequence_length / mask_length + torch.rand((1,)).item()
    )
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = torch.zeros(
        (batch_size, sequence_length), device=device, dtype=torch.bool
    )

    # uniform distribution to sample from, make sure that offset samples are <
    # sequence_length
    uniform_dist = torch.ones(
        (batch_size, sequence_length - (mask_length - 1)), device=device
    )

    # get random indices to mask
    spec_aug_mask_idxs = torch.multinomial(uniform_dist, num_masked_spans)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = (
        spec_aug_mask_idxs.unsqueeze(dim=-1)
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    offsets = (
        torch.arange(mask_length, device=device)[None, None, :]
        .expand((batch_size, num_masked_spans, mask_length))
        .reshape(batch_size, num_masked_spans * mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    spec_aug_mask = spec_aug_mask.scatter(1, spec_aug_mask_idxs, True)

    if attention_mask is not None:
        # make sure padded input ids cannot be masked
        spec_aug_mask = torch.where(attention_mask.bool(), spec_aug_mask, False)

    return spec_aug_mask


def compute_mask_indices_np(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape (from official wav2vec 2.0)

    Args:
        shape: the the shape for which to compute masks. should be of size 2 where first
            element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will
            prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be
            masked. this will be multiplied by number of timesteps divided by length of
            mask span to mask approximately this percentage of all elements. however due
            to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev
                mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that
            prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep
            unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of
            masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len and require_same_masks:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        if mask_dropout > 0:
            num_holes = np.rint(len(mask_idc) * mask_dropout).astype(int)
            mask_idc = np.random.choice(
                mask_idc, len(mask_idc) - num_holes, replace=False
            )

        mask[i, mask_idc] = True

    return mask


def compute_mask_indices_usergit(
    shape: Tuple[int, int],
    padding_mask: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    no_overlap: bool = False,
    min_space: int = 0,
    attention_mask: torch.Tensor = None,
    min_masks: int = 0,
) -> torch.Tensor:
    """
    Computes random mask spans for a given shape.
    Adapted from `fairseq's data_utils.py
    <https://github.com/pytorch/fairseq/blob/e0788f7007a8473a76db573985031f3c94201e79/fairseq/data/data_utils.py#L376>`__.

    Args:
        shape: the the shape for which to compute masks. should be of size 2 where first
            element is batch size and 2nd is timesteps
        attention_mask: optional padding mask of the same size as shape, which will
            prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be
            masked. this will be multiplied by number of timesteps divided by length of
            mask span to mask approximately this percentage of all elements. however due
            to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans
    """

    bsz, all_sz = shape
    mask = torch.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + random.random()
    )

    all_num_mask = max(min_masks, all_num_mask)
    if all_num_mask == 0:
        return mask

    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None

    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            # sz = all_sz - padding_mask[i].sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + random.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = torch.full([num_mask], mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = int(min(lengths)) if not lengths.nelement() == 0 else 0
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        # mask_idc = torch.randint(sz - min_len, [num_mask]) # TODO: should sample w/o replacement
        mask_idc = random.sample(range(sz - min_len), num_mask)
        mask_idc = torch.Tensor(
            [
                mask_idc[j] + offset
                for j in range(num_mask)
                for offset in range(lengths[j])
            ]
        )
        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = mask_idc.gather(
                dim=0, index=torch.multinomial(mask_idc, min_len, replacement=False)
            )
        mask[i, mask_idc.long()] = True
        # mask[i, mask_idc] = True

    return mask
