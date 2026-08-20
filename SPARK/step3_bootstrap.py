#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3 — Original + Circular Block Bootstrap surrogates

SPARK convention
----------------
Input time series has shape:

    T × V

where:
    T = time points
    V = voxels or surface vertices

Bootstrap behavior
------------------
boot_000.mat
    Untouched original time series.

boot_001.mat ... boot_N.mat
    Circular Block Bootstrap (CBB) surrogates.

This implementation matches the original MATLAB SPARK behavior through
NIAK's circular block bootstrap:

- sample block starts from all T frames
- preserve contiguous temporal samples inside each block
- use the same temporal indices for all voxels/vertices
- allow blocks to wrap around the end of the scan using modulo indexing
- truncate the concatenated sequence back to exactly T frames

Circular block bootstrap preserves local temporal autocorrelation within
blocks while maintaining synchronized whole-brain spatial patterns for
each selected frame.
"""

import os
import argparse
import logging

import numpy as np
from scipy.io import loadmat, savemat


# ============================================================
# LOGGING
# ============================================================

def setup_logger(outdir):

    os.makedirs(
        outdir,
        exist_ok=True
    )

    logging.basicConfig(
        level=logging.INFO,
        format="[step3] %(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    outdir,
                    "step3_bootstrap.log"
                )
            ),
            logging.StreamHandler()
        ]
    )


# ============================================================
# CIRCULAR BLOCK BOOTSTRAP
# ============================================================

def circular_block_bootstrap(
    X,
    L,
    rng=None
):
    """
    Circular block bootstrap for SPARK time series.

    Parameters
    ----------
    X : ndarray, shape (T, V)
        SPARK time-series matrix.

    L : int
        Temporal block length.

    rng : numpy.random.Generator, optional
        Random-number generator.

    Returns
    -------
    X_boot : ndarray, shape (T, V)
        Circular block bootstrap surrogate.

    Notes
    -----
    The same temporal indices are applied to every voxel/vertex.

    Example
    -------
    If T=10 and a block starts at frame 8 with L=4:

        [8, 9, 0, 1]

    so the block wraps around the scan boundary.

    This mirrors the original MATLAB SPARK / NIAK CBB behavior.
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"X must be a 2D T×V matrix, got shape {X.shape}"
        )

    T = X.shape[0]

    if T < 1:
        raise ValueError(
            "Time series contains no time points."
        )

    if L < 1:
        raise ValueError(
            f"Block length must be >= 1, received {L}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # Number of blocks required to cover T frames.
    n_blocks = int(
        np.ceil(
            T / L
        )
    )

    # Sample starts from ALL temporal frames.
    starts = rng.integers(
        low=0,
        high=T,
        size=n_blocks
    )

    offsets = np.arange(
        L,
        dtype=np.int64
    )

    # Build contiguous blocks and allow wraparound modulo T.
    indices = (
        starts[:, None]
        + offsets[None, :]
    ).reshape(-1)

    indices = (
        indices[:T]
        % T
    )

    return X[
        indices,
        :
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    p = argparse.ArgumentParser(
        description=(
            "Step 3 – Generate original + circular block "
            "bootstrap SPARK time series"
        )
    )

    p.add_argument(
        "--tseries",
        required=True,
        help="tseries.mat from Step 1"
    )

    p.add_argument(
        "--block-len",
        type=int,
        required=True,
        help="Circular bootstrap temporal block length"
    )

    p.add_argument(
        "--n-boot",
        type=int,
        required=True,
        help="TOTAL files to write, including original boot_000"
    )

    p.add_argument(
        "--outdir",
        required=True
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducible bootstrap "
            "generation"
        )
    )

    args = p.parse_args()

    setup_logger(
        args.outdir
    )

    # ========================================================
    # VALIDATE ARGUMENTS
    # ========================================================

    if args.n_boot < 1:
        raise ValueError(
            f"--n-boot must be >= 1, received {args.n_boot}"
        )

    if args.block_len < 1:
        raise ValueError(
            f"--block-len must be >= 1, received {args.block_len}"
        )

    # ========================================================
    # LOAD STEP-1 TIME SERIES
    # ========================================================

    mat = loadmat(
        args.tseries
    )

    if "tseries_full" not in mat:
        raise KeyError(
            f"'tseries_full' not found in {args.tseries}"
        )

    X = np.asarray(
        mat["tseries_full"]
    )

    if X.ndim != 2:
        raise ValueError(
            "tseries_full must be a 2D T×V matrix, "
            f"got {X.shape}"
        )

    T, V = X.shape

    logging.info(
        "Loaded time series: T=%d V=%d",
        T,
        V
    )

    logging.info(
        "Circular block bootstrap: block length=%d",
        args.block_len
    )

    if args.seed is not None:
        logging.info(
            "Random seed: %d",
            args.seed
        )

    # ========================================================
    # RANDOM GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        args.seed
    )

    # ========================================================
    # BOOTSTRAP 0 = ORIGINAL
    # ========================================================

    original_path = os.path.join(
        args.outdir,
        "boot_000.mat"
    )

    savemat(
        original_path,
        {
            "boot": X
        }
    )

    logging.info(
        "Saved boot_000.mat (original)"
    )

    # ========================================================
    # BOOTSTRAP SURROGATES
    # ========================================================

    for b in range(
        1,
        args.n_boot
    ):

        X_boot = circular_block_bootstrap(
            X=X,
            L=args.block_len,
            rng=rng
        )

        if X_boot.shape != X.shape:
            raise RuntimeError(
                "Bootstrap output shape mismatch: "
                f"expected {X.shape}, got {X_boot.shape}"
            )

        out_path = os.path.join(
            args.outdir,
            f"boot_{b:03d}.mat"
        )

        savemat(
            out_path,
            {
                "boot": X_boot
            }
        )

    logging.info(
        "✅ Wrote %d files "
        "(1 original + %d circular block surrogates)",
        args.n_boot,
        args.n_boot - 1
    )


if __name__ == "__main__":
    main()