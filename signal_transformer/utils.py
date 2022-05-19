import numpy as np
import torch

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
        shape: shape of mask array, only supported tensors of dims=2
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
