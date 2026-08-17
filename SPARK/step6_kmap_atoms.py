#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 — Final SPARK atoms + k-hubness

Supported outputs
-----------------
NIfTI input:
    atom_XX_<SUBJECT>.nii.gz
    k_hubness_<SUBJECT>.nii.gz

GIFTI input:
    atom_XX_<SUBJECT>.func.gii
    k_hubness_<SUBJECT>.func.gii

The original input format is recovered from tseries.mat,
which is produced by Step 1.

SPARK centroids always have shape:

    K × V

where:
    K = number of atoms/networks
    V = retained spatial elements

For NIfTI:
    V = retained voxels

For GIFTI:
    V = retained surface vertices
"""

import os
import argparse
import logging

import numpy as np
import nibabel as nib

from scipy.io import loadmat
from scipy.stats import norm

from spatial_io import save_gifti_map


# ============================================================
# LOGGING
# ============================================================

def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="[step6] %(asctime)s - %(levelname)s - %(message)s"
    )


# ============================================================
# HELPERS
# ============================================================

def load_centroids(path):
    """
    Load clustered SPARK centroids.

    Expected:
        centroids : K × V
    """

    m = loadmat(path)

    if "centroids" not in m:
        raise KeyError(
            f"'centroids' not found in {path}"
        )

    C = np.asarray(
        m["centroids"]
    )

    if C.ndim != 2:
        raise ValueError(
            f"'centroids' must be 2D (K×V), got {C.shape}"
        )

    return C.astype(
        np.float32,
        copy=False
    )


def matlab_string(value):
    """
    Robustly recover a Python string saved through scipy.io.savemat.

    Handles MATLAB object arrays / character arrays produced by
    Step 1 metadata fields such as input_format and gifti_template.
    """

    x = value

    while isinstance(x, np.ndarray):

        if x.size == 0:
            return ""

        if x.dtype.kind in ("U", "S"):

            flat = x.ravel()

            if flat.size == 1:
                return str(flat[0])

            return "".join(
                str(v)
                for v in flat
            )

        x = x.ravel()[0]

    if isinstance(x, bytes):
        return x.decode("utf-8")

    return str(x)


def get_input_format(tseries_mat):
    """
    Recover input format from Step-1 tseries.mat.
    """

    if "input_format" not in tseries_mat:

        # Backward compatibility:
        # old tseries.mat files had voxel coordinates but no
        # explicit input_format field.
        if "coords" in tseries_mat:

            logging.warning(
                "input_format missing from tseries.mat; "
                "assuming legacy NIfTI input."
            )

            return "nifti"

        raise KeyError(
            "'input_format' not found in tseries.mat"
        )

    fmt = matlab_string(
        tseries_mat["input_format"]
    ).strip().lower()

    if fmt not in (
        "nifti",
        "gifti"
    ):

        raise ValueError(
            f"Unsupported input_format in tseries.mat: {fmt}"
        )

    return fmt


def rowwise_z(C):
    """
    Z-score each centroid across spatial elements.

    Parameters
    ----------
    C : K × V

    Returns
    -------
    Z : K × V
    """

    K, V = C.shape

    Z = np.empty(
        (K, V),
        dtype=np.float32
    )

    for k in range(K):

        row = C[k]

        mu = float(
            row.mean()
        )

        sd = float(
            row.std()
        )

        if (
            not np.isfinite(sd)
            or sd == 0.0
        ):

            Z[k].fill(0.0)

        else:

            Z[k] = (
                row - mu
            ) / sd

    return Z


def pin_header_scaling(
    img,
    dtype
):
    """
    Force NIfTI slope/intercept and dtype to avoid
    viewer autoscaling.
    """

    hdr = img.header

    hdr.set_data_dtype(
        dtype
    )

    hdr["scl_slope"] = 1.0
    hdr["scl_inter"] = 0.0

    return img


# ============================================================
# NIFTI OUTPUT
# ============================================================

def save_nifti_outputs(
    C,
    Z,
    SEL,
    kept,
    tseries_mat,
    mask_path,
    outdir,
    subject_label
):
    """
    Reconstruct SPARK atoms and k-hubness into NIfTI volume.
    """

    if mask_path is None:

        raise ValueError(
            "NIfTI Step 6 requires --mask."
        )

    if "coords" not in tseries_mat:

        raise KeyError(
            "'coords' not found in tseries.mat "
            "for NIfTI reconstruction."
        )

    coords = np.asarray(
        tseries_mat["coords"]
    ).astype(int)

    if (
        coords.ndim != 2
        or coords.shape[1] != 3
    ):

        raise ValueError(
            f"'coords' must have shape (V,3), got {coords.shape}"
        )

    K, V = C.shape

    if coords.shape[0] != V:

        raise ValueError(
            "Spatial mismatch between centroids and "
            f"NIfTI coordinates: V={V}, coords={coords.shape[0]}"
        )

    maskimg = nib.load(
        mask_path
    )

    shape = maskimg.shape
    affine = maskimg.affine

    if len(shape) != 3:

        raise ValueError(
            f"NIfTI mask must be 3D, got {shape}"
        )

    ravel_idx = np.ravel_multi_index(
        coords.T,
        dims=shape
    )

    # --------------------------------------------------------
    # Individual atoms
    # --------------------------------------------------------

    for k in kept:

        sel = SEL[k]
        zrow = Z[k]

        flat = np.zeros(
            np.prod(shape),
            dtype=np.float32
        )

        if sel.any():

            flat[
                ravel_idx[sel]
            ] = zrow[sel]

        atom_img = nib.Nifti1Image(
            flat.reshape(shape),
            affine
        )

        # Preserve spatial transforms from mask
        qform, qcode = maskimg.get_qform(
            coded=True
        )

        sform, scode = maskimg.get_sform(
            coded=True
        )

        if qform is not None:
            atom_img.set_qform(
                qform,
                code=int(qcode)
            )

        if sform is not None:
            atom_img.set_sform(
                sform,
                code=int(scode)
            )

        atom_img = pin_header_scaling(
            atom_img,
            np.float32
        )

        atom_path = os.path.join(
            outdir,
            f"atom_{k:02d}_{subject_label}.nii.gz"
        )

        nib.save(
            atom_img,
            atom_path
        )

        logging.info(
            "Saved NIfTI atom → %s "
            "(voxels kept: %d)",
            atom_path,
            int(sel.sum())
        )

    # --------------------------------------------------------
    # k-hubness
    # --------------------------------------------------------

    if kept.size:

        hub_counts_V = SEL[
            kept
        ].sum(
            axis=0
        ).astype(
            np.int32
        )

    else:

        hub_counts_V = np.zeros(
            V,
            dtype=np.int32
        )

    hub_flat = np.zeros(
        np.prod(shape),
        dtype=np.int32
    )

    hub_flat[
        ravel_idx
    ] = hub_counts_V

    hub3d = hub_flat.reshape(
        shape
    )

    k_img = nib.Nifti1Image(
        hub3d.astype(
            np.int16,
            copy=False
        ),
        affine
    )

    qform, qcode = maskimg.get_qform(
        coded=True
    )

    sform, scode = maskimg.get_sform(
        coded=True
    )

    if qform is not None:
        k_img.set_qform(
            qform,
            code=int(qcode)
        )

    if sform is not None:
        k_img.set_sform(
            sform,
            code=int(scode)
        )

    k_img = pin_header_scaling(
        k_img,
        np.int16
    )

    k_img.header["cal_min"] = 0

    k_img.header["cal_max"] = int(
        hub_counts_V.max()
        if hub_counts_V.size
        else 0
    )

    k_path = os.path.join(
        outdir,
        f"k_hubness_{subject_label}.nii.gz"
    )

    nib.save(
        k_img,
        k_path
    )

    logging.info(
        "Saved NIfTI k-hubness → %s (max=%d)",
        k_path,
        int(
            hub_counts_V.max()
            if hub_counts_V.size
            else 0
        )
    )


# ============================================================
# GIFTI OUTPUT
# ============================================================

def save_gifti_outputs(
    C,
    Z,
    SEL,
    kept,
    tseries_mat,
    outdir,
    subject_label
):
    """
    Reconstruct SPARK atoms and k-hubness onto the
    original GIFTI surface.
    """

    # --------------------------------------------------------
    # Required Step-1 metadata
    # --------------------------------------------------------

    if "vertex_indices" not in tseries_mat:

        raise KeyError(
            "'vertex_indices' not found in tseries.mat "
            "for GIFTI reconstruction."
        )

    if "n_spatial_original" not in tseries_mat:

        raise KeyError(
            "'n_spatial_original' not found in tseries.mat."
        )

    vertex_indices = np.asarray(
        tseries_mat["vertex_indices"]
    ).astype(
        np.int64
    ).ravel()

    n_vertices_original = int(
        np.asarray(
            tseries_mat[
                "n_spatial_original"
            ]
        ).squeeze()
    )

    K, V = C.shape

    if vertex_indices.size != V:

        raise ValueError(
            "Spatial mismatch between centroids and "
            "GIFTI vertex indices:\n"
            f"  centroid V = {V}\n"
            f"  vertex_indices = {vertex_indices.size}"
        )

    if np.any(
        vertex_indices < 0
    ) or np.any(
        vertex_indices >= n_vertices_original
    ):

        raise ValueError(
            "GIFTI vertex indices fall outside the "
            "original surface dimensions."
        )

    # --------------------------------------------------------
    # Original GIFTI template
    # --------------------------------------------------------

    template_path = None

    if "gifti_template" in tseries_mat:

        template_path = matlab_string(
            tseries_mat[
                "gifti_template"
            ]
        ).strip()

    elif "source_file" in tseries_mat:

        template_path = matlab_string(
            tseries_mat[
                "source_file"
            ]
        ).strip()

    if (
        template_path
        and not os.path.exists(
            template_path
        )
    ):

        logging.warning(
            "Original GIFTI template does not exist: %s. "
            "Outputs will be written without template metadata.",
            template_path
        )

        template_path = None

    logging.info(
        "Original GIFTI vertices: %d",
        n_vertices_original
    )

    logging.info(
        "Retained GIFTI vertices: %d",
        V
    )

    # --------------------------------------------------------
    # Individual atoms
    # --------------------------------------------------------

    for k in kept:

        sel = SEL[k]
        zrow = Z[k]

        # Full original surface
        surface = np.zeros(
            n_vertices_original,
            dtype=np.float32
        )

        #
        # Only statistically selected locations receive
        # their z-score. All other vertices remain zero.
        #

        selected_vertices = (
            vertex_indices[sel]
        )

        surface[
            selected_vertices
        ] = zrow[sel]

        atom_path = os.path.join(
            outdir,
            f"atom_{k:02d}_{subject_label}.func.gii"
        )

        save_gifti_map(
            values=surface,
            output_path=atom_path,
            template_path=template_path
        )

        logging.info(
            "Saved GIFTI atom → %s "
            "(vertices kept: %d)",
            atom_path,
            int(sel.sum())
        )

    # --------------------------------------------------------
    # k-hubness
    # --------------------------------------------------------

    if kept.size:

        hub_counts_V = SEL[
            kept
        ].sum(
            axis=0
        ).astype(
            np.int32
        )

    else:

        hub_counts_V = np.zeros(
            V,
            dtype=np.int32
        )

    hub_surface = np.zeros(
        n_vertices_original,
        dtype=np.float32
    )

    hub_surface[
        vertex_indices
    ] = hub_counts_V.astype(
        np.float32
    )

    k_path = os.path.join(
        outdir,
        f"k_hubness_{subject_label}.func.gii"
    )

    save_gifti_map(
        values=hub_surface,
        output_path=k_path,
        template_path=template_path
    )

    logging.info(
        "Saved GIFTI k-hubness → %s (max=%d)",
        k_path,
        int(
            hub_counts_V.max()
            if hub_counts_V.size
            else 0
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Step 6 – Compute SPARK atoms and k-hubness "
            "for NIfTI or GIFTI data"
        )
    )

    ap.add_argument(
        "--clusters",
        required=True
    )

    ap.add_argument(
        "--tseries",
        required=True
    )

    ap.add_argument(
        "--mask",
        default=None,
        help=(
            "NIfTI mask used for volumetric reconstruction. "
            "Not required for GIFTI."
        )
    )

    ap.add_argument(
        "--pvalue",
        type=float,
        default=0.05
    )

    ap.add_argument(
        "--min_voxels",
        type=int,
        default=30,
        help=(
            "Minimum significant spatial elements per atom. "
            "For GIFTI these are vertices."
        )
    )

    ap.add_argument(
        "--outdir",
        required=True
    )

    ap.add_argument(
        "--subject_label",
        required=True
    )

    args = ap.parse_args()

    setup_logger()

    os.makedirs(
        args.outdir,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    C = load_centroids(
        args.clusters
    )

    tseries_mat = loadmat(
        args.tseries
    )

    input_format = get_input_format(
        tseries_mat
    )

    K, V = C.shape

    logging.info(
        "Detected original input format: %s",
        input_format.upper()
    )

    logging.info(
        "Centroids: K=%d, V=%d",
        K,
        V
    )

    # ========================================================
    # Z-SCORE + THRESHOLD
    # ========================================================

    z_thr = float(
        norm.ppf(
            1.0 - args.pvalue / 2.0
        )
    )

    logging.info(
        "Z-threshold "
        "(two-tailed p=%.4g): |z| > %.3f",
        args.pvalue,
        z_thr
    )

    Z = rowwise_z(
        C
    )

    SEL = (
        np.abs(Z) > z_thr
    )

    # ========================================================
    # FILTER SMALL ATOMS
    # ========================================================

    counts = SEL.sum(
        axis=1
    )

    kept = np.flatnonzero(
        counts >= int(
            args.min_voxels
        )
    )

    logging.info(
        "Kept atoms: %d/%d "
        "(min spatial elements=%d)",
        kept.size,
        K,
        args.min_voxels
    )

    # ========================================================
    # FORMAT-SPECIFIC OUTPUT
    # ========================================================

    if input_format == "nifti":

        save_nifti_outputs(
            C=C,
            Z=Z,
            SEL=SEL,
            kept=kept,
            tseries_mat=tseries_mat,
            mask_path=args.mask,
            outdir=args.outdir,
            subject_label=args.subject_label
        )

    elif input_format == "gifti":

        save_gifti_outputs(
            C=C,
            Z=Z,
            SEL=SEL,
            kept=kept,
            tseries_mat=tseries_mat,
            outdir=args.outdir,
            subject_label=args.subject_label
        )

    else:

        raise ValueError(
            f"Unsupported format: {input_format}"
        )

    logging.info(
        "✅ Step 6 completed successfully"
    )


if __name__ == "__main__":
    main()