#!/usr/bin/env python3
import numpy as np
import math

def description_length(x, D, a):
    """
    Per‐voxel MDL: data‐fit + sparse‐code penalty (in bits).
    x : (T,)       time‐series
    D : (T, K)     dictionary
    a : (K,)       sparse code
    """
    resid = x - D @ a
    T = x.size
    sigma2 = np.mean(resid**2)
    sigma2 = max(sigma2, 1e-12)
    # data term (in bits)
    L_data = 0.5 * T * math.log2(2 * math.pi * sigma2) \
             + np.sum(resid**2) / (2 * sigma2) / math.log(2)
    # sparse penalty: k*(log2(K)+16)
    k = np.count_nonzero(a)
    if k == 0:
        return np.inf
    K = D.shape[1]
    L_sparse = k * (math.log2(K) + 16)
    return L_data + L_sparse


def get_cifti_region_indices(cifti_path):
    """Partition original CIFTI grayordinates into cortex and noncortex."""
    import nibabel as nib
    from nibabel.cifti2.cifti2_axes import BrainModelAxis

    img = nib.load(str(cifti_path))
    brain_axis = img.header.get_axis(1)

    if not isinstance(brain_axis, BrainModelAxis):
        raise ValueError("Expected BrainModelAxis on CIFTI axis 1.")

    # Each grayordinate has its own anatomical structure name.
    names = np.asarray(brain_axis.name).astype(str)

    cortical_names = [
        "CIFTI_STRUCTURE_CORTEX_LEFT",
        "CIFTI_STRUCTURE_CORTEX_RIGHT",
    ]

    cortex_mask = np.isin(names, cortical_names)

    regions = {
        "cortex": np.flatnonzero(cortex_mask).astype(np.int64),
        "noncortex": np.flatnonzero(~cortex_mask).astype(np.int64),
    }

    structures = []
    for name, count in zip(*np.unique(names, return_counts=True)):
        region = "cortex" if name in cortical_names else "noncortex"
        structures.append((name, region, int(count)))

    # Verify that all grayordinates are assigned exactly once.
    combined = np.concatenate([
        regions["cortex"],
        regions["noncortex"],
    ])

    if (
        combined.size != len(brain_axis)
        or not np.array_equal(
            np.sort(combined),
            np.arange(len(brain_axis))
        )
    ):
        raise ValueError("Invalid CIFTI regional partition.")

    return regions, structures
