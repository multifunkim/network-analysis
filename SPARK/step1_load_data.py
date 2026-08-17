#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 — Load, preprocess, and subsample SPARK input

Supported input formats
-----------------------
NIfTI:
    .nii
    .nii.gz

GIFTI:
    .func.gii

All supported formats are converted internally to:

    tseries_full : (T, V)

where:
    T = number of time points
    V = number of retained spatial elements

For NIfTI:
    spatial elements = voxels

For GIFTI:
    spatial elements = surface vertices
"""

import os
import argparse
import logging
import numpy as np
from scipy.io import savemat

from spatial_io import load_timeseries, detect_format


def setup_logger(out_mat):
    d = os.path.dirname(out_mat) or "."
    os.makedirs(d, exist_ok=True)

    logf = os.path.join(d, "step1_load_data.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[step1] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logf),
            logging.StreamHandler()
        ]
    )


def main():

    # ============================================================
    # ARGUMENTS
    # ============================================================

    p = argparse.ArgumentParser(
        description=(
            "Step 1 – Load, preprocess, and subsample "
            "NIfTI or GIFTI fMRI data"
        )
    )

    p.add_argument(
        "--fmri",
        required=True,
        help="Input .nii/.nii.gz or .func.gii file"
    )

    p.add_argument(
        "--mask",
        default=None,
        help=(
            "3D NIfTI mask. Required for NIfTI input; "
            "not required for GIFTI."
        )
    )

    p.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Keep every Nth spatial element"
    )

    p.add_argument(
        "--out",
        required=True,
        help="Output tseries.mat file"
    )

    args = p.parse_args()

    setup_logger(args.out)

    # ============================================================
    # VALIDATE SUBSAMPLING
    # ============================================================

    if args.subsample < 1:
        raise ValueError(
            f"--subsample must be >= 1, received {args.subsample}"
        )

    # ============================================================
    # DETECT INPUT FORMAT
    # ============================================================

    input_format = detect_format(args.fmri)

    logging.info(
        "Starting SPARK Step 1"
    )

    logging.info(
        "Input file: %s",
        args.fmri
    )

    logging.info(
        "Detected input format: %s",
        input_format.upper()
    )

    # ============================================================
    # FORMAT-AWARE LOADING
    # ============================================================

    loaded = load_timeseries(
        input_path=args.fmri,
        mask_path=args.mask
    )

    ts_full = np.asarray(
        loaded["tseries"],
        dtype=np.float64
    )

    spatial_index = np.asarray(
        loaded["spatial_index"]
    )

    n_spatial_original = int(
        loaded["n_spatial_original"]
    )

    logging.info(
        "Loaded data: T=%d, spatial elements=%d",
        ts_full.shape[0],
        ts_full.shape[1]
    )

    # ============================================================
    # REMOVE INVALID / ZERO-VARIANCE ELEMENTS
    # ============================================================

    #
    # Original SPARK Step 1 removed columns with variance <= 1e-6.
    # We preserve that behavior for both voxels and vertices.
    #

    finite = np.all(
        np.isfinite(ts_full),
        axis=0
    )

    var = np.var(
        ts_full,
        axis=0
    )

    keep = finite & (var > 1e-6)

    n_removed = int(
        np.sum(~keep)
    )

    ts_full = ts_full[:, keep]
    spatial_index = spatial_index[keep]

    logging.info(
        "Removed %d invalid/zero-variance spatial elements",
        n_removed
    )

    logging.info(
        "After filtering: T=%d, V=%d",
        ts_full.shape[0],
        ts_full.shape[1]
    )

    if ts_full.shape[1] == 0:
        raise ValueError(
            "No spatial elements remain after "
            "zero-variance/finite-value filtering."
        )

    # ============================================================
    # NORMALIZE EACH SPATIAL TIME SERIES
    # ============================================================

    #
    # Same normalization as original SPARK implementation:
    #
    #     x = (x - mean) / std
    #

    mu = ts_full.mean(
        axis=0,
        keepdims=True
    )

    sigma = ts_full.std(
        axis=0,
        keepdims=True
    )

    ts_full = (
        ts_full - mu
    ) / np.maximum(
        sigma,
        1e-6
    )

    logging.info(
        "Normalized full time-series: %s",
        ts_full.shape
    )

    # ============================================================
    # SUBSAMPLING
    # ============================================================

    if args.subsample > 1:

        idx_sub = np.arange(
            0,
            ts_full.shape[1],
            args.subsample,
            dtype=np.int64
        )

        ts_sub = ts_full[:, idx_sub]

        logging.info(
            "Subsampling factor: %d",
            args.subsample
        )

        logging.info(
            "Subsampled time-series: %s",
            ts_sub.shape
        )

    else:

        idx_sub = np.arange(
            ts_full.shape[1],
            dtype=np.int64
        )

        ts_sub = ts_full

        logging.info(
            "No spatial subsampling applied"
        )

    # ============================================================
    # COMMON OUTPUT
    # ============================================================

    #
    # These variables are format-independent and are used by
    # downstream SPARK Steps 2–5.
    #

    output = {
        "tseries_full": ts_full,
        "tseries_sub": ts_sub,
        "spatial_index": spatial_index,
        "idx_sub": idx_sub,
        "input_format": np.array(
            [input_format],
            dtype=object
        ),
        "n_spatial_original": np.array(
            [[n_spatial_original]],
            dtype=np.int64
        ),
        "n_spatial_retained": np.array(
            [[ts_full.shape[1]]],
            dtype=np.int64
        ),
        "n_timepoints": np.array(
            [[ts_full.shape[0]]],
            dtype=np.int64
        ),
        "source_file": np.array(
            [os.path.abspath(args.fmri)],
            dtype=object
        ),
    }

    # ============================================================
    # NIFTI-SPECIFIC METADATA
    # ============================================================

    if input_format == "nifti":

        #
        # Keep "coords" for backward compatibility with the
        # existing Step 6 NIfTI implementation.
        #

        output["coords"] = spatial_index

        output["shape"] = np.asarray(
            loaded["image_shape"],
            dtype=np.int64
        )

        output["affine"] = np.asarray(
            loaded["affine"],
            dtype=np.float64
        )

        if args.mask is not None:

            output["mask_file"] = np.array(
                [os.path.abspath(args.mask)],
                dtype=object
            )

        logging.info(
            "Stored NIfTI voxel coordinates and geometry"
        )

    # ============================================================
    # GIFTI-SPECIFIC METADATA
    # ============================================================

    elif input_format == "gifti":

        #
        # spatial_index contains the original GIFTI vertex indices
        # retained after variance filtering.
        #

        output["vertex_indices"] = spatial_index.astype(
            np.int64
        )

        output["gifti_template"] = np.array(
            [os.path.abspath(args.fmri)],
            dtype=object
        )

        logging.info(
            "Stored GIFTI vertex indices and template path"
        )

    # ============================================================
    # SAVE
    # ============================================================

    savemat(
        args.out,
        output
    )

    logging.info(
        "Saved SPARK time-series → %s",
        args.out
    )

    logging.info(
        "Step 1 completed successfully"
    )

    logging.info(
        "Summary: format=%s | T=%d | original=%d | retained=%d | subsampled=%d",
        input_format,
        ts_full.shape[0],
        n_spatial_original,
        ts_full.shape[1],
        ts_sub.shape[1]
    )


if __name__ == "__main__":
    main()