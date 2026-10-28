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
    """Identify cortical and noncortical columns from CIFTI metadata.

    Returns original grayordinate indices and structure counts.
    """
    import nibabel as nib

    img = nib.load(str(cifti_path))
    brain_axis = img.header.get_axis(1)

    cortex = []
    noncortex = []
    structures = []

    for name, slc, _ in brain_axis.iter_structures():
        indices = np.arange(slc.start, slc.stop, dtype=np.int64)

        region = (
            "cortex"
            if name in (
                "CIFTI_STRUCTURE_CORTEX_LEFT",
                "CIFTI_STRUCTURE_CORTEX_RIGHT",
            )
            else "noncortex"
        )

        if region == "cortex":
            cortex.append(indices)
        else:
            noncortex.append(indices)

        structures.append((name, region, len(indices)))

    result = {
        "cortex": (
            np.concatenate(cortex)
            if cortex else np.array([], dtype=np.int64)
        ),
        "noncortex": (
            np.concatenate(noncortex)
            if noncortex else np.array([], dtype=np.int64)
        ),
    }

    combined = np.concatenate(list(result.values()))

    if (
        combined.size != len(brain_axis)
        or np.unique(combined).size != len(brain_axis)
    ):
        raise ValueError("Invalid CIFTI regional partition.")

    return result, structures
