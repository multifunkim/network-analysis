#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2 – Grid-search (K, L) via MDL

Quick reference
===============

coding mode        | default c_bits | best-pair rule
-------------------|----------------|---------------------------
threshold (MATLAB) | 0              | row-mean (K) -> min L
omp (Python)       | 16             | global MDL minimum
any --rowmean      | unchanged      | row-mean (K) -> min L

Example:
    --step2_extra --coding omp --c_bits 8 --rowmean

Notes
=====
For OMP coding, scikit-learn may emit repeated RuntimeWarnings when
dictionary atoms become linearly dependent. These warnings are suppressed
here to avoid flooding SLURM logs.

The OMP solver behavior itself is NOT changed.
"""

import os
import argparse
import logging
import math
import warnings

import numpy as np

from scipy.io import loadmat, savemat
from sklearn.linear_model import orthogonal_mp
from sklearn.utils.extmath import randomized_svd


# ============================================================
# LOGGING
# ============================================================

def setup_logger(out_mat):

    d = os.path.dirname(out_mat) or "."
    os.makedirs(d, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[step2] %(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(
                    d,
                    "step2_estimate_scale.log"
                )
            ),
            logging.StreamHandler()
        ]
    )


# ============================================================
# MDL
# ============================================================

def description_length(
    x,
    D,
    a,
    c_bits
):
    """
    Compute description length for one spatial element.

    Parameters
    ----------
    x : (T,)
        Time series.

    D : (T, K)
        Dictionary.

    a : (K,)
        Sparse coefficients.

    c_bits : float
        Cost per coefficient.

    Returns
    -------
    float
        MDL value in bits.
    """

    resid = x - D @ a

    sigma2 = max(
        np.mean(resid ** 2),
        1e-12
    )

    data = (
        0.5
        * len(x)
        * math.log2(
            2 * math.pi * sigma2
        )
        + np.sum(resid ** 2)
        / (2 * sigma2)
        / math.log(2)
    )

    k = np.count_nonzero(a)

    if k == 0:
        return np.inf

    return (
        data
        + k
        * (
            math.log2(
                D.shape[1]
            )
            + c_bits
        )
    )


# ============================================================
# SPARSE CODING
# ============================================================

def threshold_code(
    D,
    y,
    L
):
    """
    MATLAB-style threshold sparse coding.
    """

    if L <= 0:
        return np.zeros(
            D.shape[1]
        )

    proj = np.abs(
        D.T @ y
    )

    idx = np.argpartition(
        -proj,
        L - 1
    )[:L]

    a = np.zeros(
        D.shape[1]
    )

    a[idx], *_ = np.linalg.lstsq(
        D[:, idx],
        y,
        rcond=None
    )

    return a


def omp_code(
    D,
    y,
    L
):
    """
    Orthogonal Matching Pursuit sparse coding.

    scikit-learn may emit repeated RuntimeWarnings when the
    dictionary contains linearly dependent atoms.

    We suppress only that specific RuntimeWarning.

    Solver behavior and returned coefficients are unchanged.
    """

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            message=(
                "Orthogonal matching pursuit ended prematurely "
                "due to linear dependence in the dictionary.*"
            ),
            category=RuntimeWarning
        )

        return orthogonal_mp(
            D,
            y,
            n_nonzero_coefs=L
        )


# ============================================================
# MINI K-SVD
# ============================================================

def mini_ksvd(
    X,
    K,
    L,
    n_iter,
    coding
):
    """
    Small K-SVD used during Step-2 MDL grid search.

    Parameters
    ----------
    X : (T, V)
        Subsampled SPARK time series.

    K : int
        Number of dictionary atoms.

    L : int
        Sparsity level.

    n_iter : int
        Number of K-SVD scan iterations.

    coding : {"threshold", "omp"}
        Sparse coding method.

    Returns
    -------
    D : (T, K)
        Learned dictionary.

    A : (K, V)
        Sparse coefficients.
    """

    rng = np.random.default_rng(0)

    D = X[
        :,
        rng.choice(
            X.shape[1],
            K,
            replace=False
        )
    ]

    D /= np.maximum(
        np.linalg.norm(
            D,
            axis=0,
            keepdims=True
        ),
        1e-12
    )

    for _ in range(
        n_iter
    ):

        # ----------------------------------------------------
        # Sparse coding
        # ----------------------------------------------------

        if coding == "threshold":

            A = np.column_stack(
                [
                    threshold_code(
                        D,
                        X[:, v],
                        L
                    )
                    for v in range(
                        X.shape[1]
                    )
                ]
            )

        else:

            A = np.column_stack(
                [
                    omp_code(
                        D,
                        X[:, v],
                        L
                    )
                    for v in range(
                        X.shape[1]
                    )
                ]
            )

        # ----------------------------------------------------
        # Dictionary update
        # ----------------------------------------------------

        for k in range(
            K
        ):

            vox = np.flatnonzero(
                A[k]
            )

            if not vox.size:
                continue

            R = (
                X[:, vox]
                - D @ A[:, vox]
                + np.outer(
                    D[:, k],
                    A[k, vox]
                )
            )

            u, s, vt = np.linalg.svd(
                R,
                full_matrices=False
            )

            D[:, k] = u[:, 0]

            A[k, vox] = (
                s[0]
                * vt[0]
            )

    return D, A


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    ap = argparse.ArgumentParser(
        description=(
            "Step 2 – Estimate SPARK network scale "
            "using MDL grid search"
        )
    )

    ap.add_argument(
        "--tseries",
        required=True
    )

    ap.add_argument(
        "--k-min",
        type=int,
        required=True
    )

    ap.add_argument(
        "--k-step",
        type=int,
        required=True
    )

    ap.add_argument(
        "--k-max",
        type=int,
        required=True
    )

    ap.add_argument(
        "--l-max",
        type=int,
        help=(
            "Cap L grid "
            "(default: K/2)"
        )
    )

    ap.add_argument(
        "--mode",
        choices=[
            "ksvd",
            "fast"
        ],
        default="ksvd"
    )

    ap.add_argument(
        "--scan-iter",
        type=int,
        default=2
    )

    ap.add_argument(
        "--coding",
        choices=[
            "threshold",
            "omp"
        ],
        default="threshold"
    )

    ap.add_argument(
        "--c_bits",
        type=float,
        help=(
            "Override bits per sparse coefficient"
        )
    )

    ap.add_argument(
        "--rowmean",
        action="store_true",
        help=(
            "Force MATLAB row-mean K selection rule "
            "regardless of coding method"
        )
    )

    ap.add_argument(
        "--out",
        required=True
    )

    args = ap.parse_args()

    setup_logger(
        args.out
    )


    # ========================================================
    # COEFFICIENT-LENGTH COST
    # ========================================================

    if args.c_bits is None:

        args.c_bits = (
            0.0
            if args.coding == "threshold"
            else 16.0
        )

    logging.info(
        "coding = %s   c_bits = %.1f   rowmean = %s",
        args.coding,
        args.c_bits,
        args.rowmean
    )

    if args.coding == "omp":

        logging.info(
            "OMP linear-dependence RuntimeWarnings are suppressed "
            "to avoid repetitive log output; "
            "solver behavior is unchanged."
        )


    # ========================================================
    # LOAD DATA
    # ========================================================

    tseries_mat = loadmat(
        args.tseries
    )

    if "tseries_sub" not in tseries_mat:

        raise KeyError(
            f"'tseries_sub' not found in {args.tseries}"
        )

    X = np.asarray(
        tseries_mat[
            "tseries_sub"
        ]
    )

    if X.ndim != 2:

        raise ValueError(
            "tseries_sub must be a 2D T×V matrix, "
            f"got {X.shape}"
        )

    logging.info(
        "Loaded subsampled time series: T=%d V=%d",
        X.shape[0],
        X.shape[1]
    )


    # ========================================================
    # K GRID
    # ========================================================

    Ks = list(
        range(
            args.k_min,
            args.k_max + 1,
            args.k_step
        )
    )

    if not Ks:

        raise ValueError(
            "K grid is empty. Check "
            "--k-min, --k-step, and --k-max."
        )

    MDL_rows = []


    # ========================================================
    # GRID SEARCH OVER (K, L)
    # ========================================================

    for K in Ks:

        L_cap = (
            args.l_max
            if args.l_max
            else K // 2
        )

        if L_cap < 1:

            raise ValueError(
                f"Invalid L range for K={K}"
            )

        row = []

        for L in range(
            1,
            L_cap + 1
        ):

            # ------------------------------------------------
            # Dictionary + sparse codes
            # ------------------------------------------------

            if args.mode == "fast":

                U, _, _ = randomized_svd(
                    X,
                    n_components=K,
                    random_state=0
                )

                D = U

                if args.coding == "threshold":

                    A = np.column_stack(
                        [
                            threshold_code(
                                D,
                                X[:, v],
                                L
                            )
                            for v in range(
                                X.shape[1]
                            )
                        ]
                    )

                else:

                    #
                    # Keep sklearn warning behavior suppressed
                    # in fast mode as well.
                    #

                    with warnings.catch_warnings():

                        warnings.filterwarnings(
                            "ignore",
                            message=(
                                "Orthogonal matching pursuit "
                                "ended prematurely due to "
                                "linear dependence in the dictionary.*"
                            ),
                            category=RuntimeWarning
                        )

                        A = orthogonal_mp(
                            D,
                            X,
                            n_nonzero_coefs=L
                        )

            else:

                D, A = mini_ksvd(
                    X=X,
                    K=K,
                    L=L,
                    n_iter=args.scan_iter,
                    coding=args.coding
                )


            # ------------------------------------------------
            # MDL
            # ------------------------------------------------

            mdl = np.mean(
                [
                    description_length(
                        X[:, v],
                        D,
                        A[:, v],
                        args.c_bits
                    )
                    for v in range(
                        X.shape[1]
                    )
                ]
            )

            row.append(
                float(mdl)
            )

            logging.info(
                "K=%3d  L=%2d  MDL=%8.2f",
                K,
                L,
                mdl
            )

        MDL_rows.append(
            row
        )


    # ========================================================
    # RECTANGULARIZE MDL TABLE
    # ========================================================

    L_max = max(
        len(r)
        for r in MDL_rows
    )

    MDL_all = np.array(
        [
            r
            + [np.nan]
            * (
                L_max
                - len(r)
            )
            for r in MDL_rows
        ]
    )


    # ========================================================
    # SELECT BEST (K, L)
    # ========================================================

    use_rowmean = (
        args.rowmean
        or (
            args.coding
            == "threshold"
        )
    )

    if use_rowmean:

        #
        # MATLAB-style:
        #
        # 1. Average MDL across L for each K
        # 2. Select best K
        # 3. Select best L within that K
        #

        row_mean = np.nanmean(
            MDL_all,
            axis=1
        )

        k_idx = int(
            np.nanargmin(
                row_mean
            )
        )

        bestK = Ks[
            k_idx
        ]

        bestL = (
            int(
                np.nanargmin(
                    MDL_all[
                        k_idx
                    ]
                )
            )
            + 1
        )

        bestMDL = MDL_all[
            k_idx,
            bestL - 1
        ]

    else:

        #
        # Global MDL minimum
        #

        flat = int(
            np.nanargmin(
                MDL_all
            )
        )

        k_idx, l_idx = np.unravel_index(
            flat,
            MDL_all.shape
        )

        bestK = Ks[
            k_idx
        ]

        bestL = (
            l_idx
            + 1
        )

        bestMDL = MDL_all[
            k_idx,
            l_idx
        ]


    # ========================================================
    # SAVE
    # ========================================================

    savemat(
        args.out,
        {
            "Ks":
                np.array(
                    Ks
                ),

            "Ls":
                np.arange(
                    1,
                    L_max + 1
                ),

            "MDL_all":
                MDL_all,

            "bestK":
                bestK,

            "bestL":
                bestL,

            "MDL_best":
                bestMDL,
        }
    )

    logging.info(
        "✅ selected K=%d  L=%d  (MDL=%.2f)",
        bestK,
        bestL,
        bestMDL
    )