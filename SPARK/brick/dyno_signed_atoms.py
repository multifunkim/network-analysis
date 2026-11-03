#!/usr/bin/env python3
"""
dynoSPARK — Signed spatial atom reconstruction.

Step 5 uses abs(A) for cluster membership.
This module reconstructs signed final atoms from Step-4 A maps.

The first (boot_000) dictionary defines preferred sign references.
Other bootstrap maps are aligned to those references before averaging.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


def build_signed_atoms(unit_dir):
    """
    Parameters
    ----------
    unit_dir : Path
        SPARK analysis directory containing clusters.mat and dict/.

    Returns
    -------
    dict
        Signed atom maps and sign-alignment diagnostics.
    """

    unit_dir = Path(unit_dir)

    clusters = loadmat(
        unit_dir / "clusters.mat",
        variable_names=["labels", "centroids"],
    )

    labels = np.asarray(
        clusters["labels"]
    ).ravel().astype(np.int64)

    centroids = np.asarray(
        clusters["centroids"],
        dtype=np.float64,
    )

    if centroids.ndim != 2 or labels.size == 0:
        raise ValueError("Invalid Step-5 cluster output.")

    K_final, V = centroids.shape

    dict_dir = unit_dir / "dict"
    dict_files = sorted(dict_dir.glob("boot_*_dict.mat"))

    if not dict_files or dict_files[0].name != "boot_000_dict.mat":
        raise FileNotFoundError("Missing boot_000 dictionary.")

    expected_names = [
        f"boot_{i:03d}_dict.mat"
        for i in range(len(dict_files))
    ]

    if [p.name for p in dict_files] != expected_names:
        raise ValueError("Bootstrap dictionary files are incomplete.")

    A0 = np.asarray(
        loadmat(
            dict_files[0],
            variable_names=["A"],
        )["A"],
        dtype=np.float64,
    )

    if A0.ndim != 2 or A0.shape[1] != V:
        raise ValueError("Original spatial dimensions do not match.")

    K_origin = A0.shape[0]
    n_boot = len(dict_files)

    if labels.size != n_boot * K_origin:
        raise ValueError("Cluster labels and dictionaries disagree.")

    if np.any(labels < 0) or np.any(labels >= K_final):
        raise ValueError("Invalid cluster IDs.")

    # --------------------------------------------------------
    # Reference orientation from the original decomposition
    # --------------------------------------------------------

    references = [None] * K_final
    reference_from_boot0 = np.zeros(K_final, dtype=bool)

    boot0_labels = labels[:K_origin]

    centroid_norms = np.linalg.norm(centroids, axis=1)

    for cluster_id in range(K_final):
        candidates = np.flatnonzero(
            boot0_labels == cluster_id
        )

        if not candidates.size:
            continue

        maps = A0[candidates]
        map_norms = np.linalg.norm(maps, axis=1)

        denominator = (
            map_norms * centroid_norms[cluster_id]
        )

        similarity = np.divide(
            np.abs(maps) @ centroids[cluster_id],
            denominator,
            out=np.zeros(candidates.size),
            where=denominator > 1e-12,
        )

        best = candidates[np.argmax(similarity)]

        references[cluster_id] = A0[best].copy()
        reference_from_boot0[cluster_id] = True

    # --------------------------------------------------------
    # Accumulate signed, aligned bootstrap maps
    # --------------------------------------------------------

    sums = np.zeros((K_final, V), dtype=np.float64)
    member_counts = np.zeros(K_final, dtype=np.int64)
    flip_counts = np.zeros(K_final, dtype=np.int64)
    similarity_sums = np.zeros(K_final, dtype=np.float64)

    for bootstrap_idx, path in enumerate(dict_files):

        A = np.asarray(
            loadmat(
                path,
                variable_names=["A"],
            )["A"],
            dtype=np.float64,
        )

        if A.shape != (K_origin, V):
            raise ValueError(
                f"Inconsistent dictionary shape: {path}"
            )

        start = bootstrap_idx * K_origin
        current_labels = labels[start:start + K_origin]

        for row_idx, cluster_id in enumerate(current_labels):

            cluster_id = int(cluster_id)
            spatial_map = A[row_idx].copy()

            # Fallback for clusters absent from boot_000.
            if references[cluster_id] is None:
                references[cluster_id] = spatial_map.copy()

            reference = references[cluster_id]

            numerator = float(
                spatial_map @ reference
            )

            denominator = (
                np.linalg.norm(spatial_map)
                * np.linalg.norm(reference)
            )

            similarity = (
                abs(numerator) / denominator
                if denominator > 1e-12
                else 0.0
            )

            if numerator < 0:
                spatial_map *= -1.0
                flip_counts[cluster_id] += 1

            sums[cluster_id] += spatial_map
            member_counts[cluster_id] += 1
            similarity_sums[cluster_id] += similarity

    if np.any(member_counts == 0):
        raise ValueError("One or more final clusters are empty.")

    # --------------------------------------------------------
    # Final signed spatial atoms
    # --------------------------------------------------------

    A_signed_raw = sums / member_counts[:, None]

    spatial_norms = np.linalg.norm(
        A_signed_raw,
        axis=1,
    )

    if np.any(spatial_norms <= 1e-12):
        raise ValueError(
            "One or more signed atoms have zero or near-zero norm."
        )

    A_signed = A_signed_raw / spatial_norms[:, None]

    mean_similarity = similarity_sums / member_counts

    return {
        "A_signed": A_signed,
        "A_signed_raw": A_signed_raw,
        "atom_ids": np.arange(K_final, dtype=np.int64),
        "spatial_norms": spatial_norms,
        "member_counts": member_counts,
        "flip_counts": flip_counts,
        "mean_abs_cosine": mean_similarity,
        "reference_from_boot0": reference_from_boot0,
        "n_boot": n_boot,
    }


def save_signed_atoms(unit_dir, output_path, overwrite=False):
    """Build and save signed atom maps with diagnostics."""

    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists. Use --overwrite to replace it."
        )

    result = build_signed_atoms(unit_dir)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save through a temporary file to avoid incomplete outputs.
    temporary = output_path.with_name(
        output_path.stem + ".tmp.mat"
    )

    try:
        savemat(
            temporary,
            result,
            do_compression=True,
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return result
