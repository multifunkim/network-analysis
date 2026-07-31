#!/usr/bin/env python3
"""
Step 2 – Grid-search (K, L) via MDL

Quick reference
===============

coding mode        | default c_bits | best-pair rule
-------------------|---------------|---------------------------
threshold *(MATLAB)| 0             | **row-mean (K)  ➜  min L**
omp *(Python)*     | 16            | **global MDL minimum**
any **--rowmean**  | (unchanged)   | **row-mean (K)  ➜  min L**

Use --step2_extra in the pipeline to forward extra flags, e.g.  
    --step2_extra --coding omp --c_bits 8 --rowmean
"""

import os, argparse, logging, math, numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import orthogonal_mp
from sklearn.utils.extmath import randomized_svd

# ── helpers ─────────────────────────────────────────────────────────────
def setup_logger(out_mat):
    d = os.path.dirname(out_mat) or "."
    os.makedirs(d, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[step2] %(asctime)s  %(message)s",
        handlers=[logging.FileHandler(os.path.join(d, "step2_estimate_scale.log")),
                  logging.StreamHandler()])

def description_length(x, D, a, c_bits):
    resid  = x - D @ a
    sigma2 = max(np.mean(resid**2), 1e-12)
    data   = 0.5 * len(x) * math.log2(2 * math.pi * sigma2) \
           + np.sum(resid**2) / (2 * sigma2) / math.log(2)
    k      = np.count_nonzero(a)
    if k == 0:
        return np.inf
    return data + k * (math.log2(D.shape[1]) + c_bits)

def threshold_code(D, y, L):
    if L <= 0:
        return np.zeros(D.shape[1])
    proj  = np.abs(D.T @ y)
    idx   = np.argpartition(-proj, L-1)[:L]
    a     = np.zeros(D.shape[1])
    a[idx], *_ = np.linalg.lstsq(D[:, idx], y, rcond=None)
    return a

def omp_code(D, y, L):
    return orthogonal_mp(D, y, n_nonzero_coefs=L)

def mini_ksvd(X, K, L, n_iter, coding):
    rng = np.random.default_rng(0)
    D   = X[:, rng.choice(X.shape[1], K, replace=False)]
    D  /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
    for _ in range(n_iter):
        if coding == "threshold":
            A = np.column_stack([threshold_code(D, X[:, v], L) for v in range(X.shape[1])])
        else:
            A = np.column_stack([omp_code(D, X[:, v], L) for v in range(X.shape[1])])
        for k in range(K):
            vox = np.flatnonzero(A[k])
            if not vox.size:
                continue
            R = X[:, vox] - D @ A[:, vox] + np.outer(D[:, k], A[k, vox])
            u, s, vt = np.linalg.svd(R, full_matrices=False)
            D[:, k]   = u[:, 0]
            A[k, vox] = s[0] * vt[0]
    return D, A

# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tseries", required=True)
    ap.add_argument("--k-min", type=int, required=True)
    ap.add_argument("--k-step", type=int, required=True)
    ap.add_argument("--k-max", type=int, required=True)
    ap.add_argument("--l-max", type=int, help="cap L (default K/2)")
    ap.add_argument("--mode", choices=["ksvd", "fast"], default="ksvd")
    ap.add_argument("--scan-iter", type=int, default=2)
    ap.add_argument("--coding", choices=["threshold", "omp"], default="threshold")
    ap.add_argument("--c_bits", type=float, help="override bits per coefficient")
    ap.add_argument("--rowmean", action="store_true",
                    help="force MATLAB row-mean rule regardless of coding")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logger(args.out)

    # ---- coefficient-length cost -------------------------------------
    if args.c_bits is None:
        args.c_bits = 0.0 if args.coding == "threshold" else 16.0
    logging.info("coding = %s   c_bits = %.1f   rowmean = %s",
                 args.coding, args.c_bits, args.rowmean)

    X   = loadmat(args.tseries)["tseries_sub"]          # (T, V_sub)
    Ks  = list(range(args.k_min, args.k_max + 1, args.k_step))
    MDL_rows = []

    # ---- grid search over (K,L) --------------------------------------
    for K in Ks:
        L_cap = args.l_max if args.l_max else K // 2
        row   = []
        for L in range(1, L_cap + 1):
            # ---- dictionary & codes ----------------------------------
            if args.mode == "fast":
                U, _, _ = randomized_svd(X, n_components=K, random_state=0)
                D = U
                if args.coding == "threshold":
                    A = np.column_stack([threshold_code(D, X[:, v], L) for v in range(X.shape[1])])
                else:
                    A = orthogonal_mp(D, X, n_nonzero_coefs=L)
            else:
                D, A = mini_ksvd(X, K, L, args.scan_iter, args.coding)
            mdl = np.mean([description_length(X[:, v], D, A[:, v], args.c_bits)
                           for v in range(X.shape[1])])
            row.append(float(mdl))
            logging.info("K=%3d  L=%2d  MDL=%8.2f", K, L, mdl)
        MDL_rows.append(row)

    # ---- rectangularise MDL table ------------------------------------
    L_max = max(len(r) for r in MDL_rows)
    MDL_all = np.array([r + [np.nan]*(L_max - len(r)) for r in MDL_rows])

    # ---- select best (K,L) -------------------------------------------
    use_rowmean = args.rowmean or (args.coding == "threshold")
    if use_rowmean:
        row_mean = np.nanmean(MDL_all, axis=1)
        k_idx    = int(np.nanargmin(row_mean))
        bestK    = Ks[k_idx]
        bestL    = int(np.nanargmin(MDL_all[k_idx])) + 1
        bestMDL  = MDL_all[k_idx, bestL-1]
    else:
        flat = int(np.nanargmin(MDL_all))
        k_idx, l_idx = np.unravel_index(flat, MDL_all.shape)
        bestK, bestL = Ks[k_idx], l_idx + 1
        bestMDL = MDL_all[k_idx, l_idx]

    # ---- save ---------------------------------------------------------
    savemat(args.out, dict(
        Ks=np.array(Ks),
        Ls=np.arange(1, L_max + 1),
        MDL_all=MDL_all,
        bestK=bestK,
        bestL=bestL,
        MDL_best=bestMDL
    ))
    logging.info("✅ selected K=%d  L=%d  (MDL=%.2f)", bestK, bestL, bestMDL)
