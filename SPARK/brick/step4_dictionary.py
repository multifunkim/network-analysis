#!/usr/bin/env python3
"""
step4_dictionary.py  •  K-SVD per bootstrap (MATLAB-style, no pruning)
"""

import os, argparse, logging, numpy as np
from scipy.io import loadmat, savemat
from joblib import Parallel, delayed
from sklearn.utils.extmath import randomized_svd

# ── utilities ────────────────────────────────────────────────────────────
def setup_logger(outdir):
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[step4] %(asctime)s  %(message)s',
        handlers=[logging.FileHandler(os.path.join(outdir,"step4_dictionary.log")),
                  logging.StreamHandler()])

def norm_sign(D):
    D = D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
    return D * np.sign(D[0,:])

def mpf_topL(x, D, L):
    proj   = np.abs(D.T @ x)
    idx    = np.argpartition(-proj, L-1)[:L]
    coef   = np.linalg.lstsq(D[:,idx], x, rcond=None)[0]
    a      = np.zeros(D.shape[1]);  a[idx] = coef
    return a

# ── per-bootstrap K-SVD ──────────────────────────────────────────────────
def run_ksvd(mat_path, K, L, n_iter, outdir, init_dir=None):
    name = os.path.basename(mat_path)[:-4]
    X    = loadmat(mat_path)["boot"]
    # init dictionary
    if init_dir:
        D0 = loadmat(os.path.join(init_dir,f"{name}_init.mat")
                     )["param"]["initialDictionary"][0,0][:,:K]
    else:
        rng = np.random.default_rng(0)
        D0  = X[:, rng.choice(X.shape[1], K, replace=False)]
    D = norm_sign(D0.astype(np.float64))

    for it in range(n_iter):
        A = np.column_stack([mpf_topL(X[:,v], D, L) for v in range(X.shape[1])])
        for k in range(K):
            vox = np.flatnonzero(A[k])
            if vox.size == 0: continue
            R   = X[:,vox] - D @ A[:,vox] + np.outer(D[:,k], A[k,vox])
            try:
                u,s,vt = np.linalg.svd(R, full_matrices=False)
            except np.linalg.LinAlgError:
                u,s,vt = randomized_svd(R, 1, random_state=0)
            D[:,k]   = u[:,0] / np.linalg.norm(u[:,0])
            A[k,vox] = s[0] * vt[0]
        D = norm_sign(D)
        logging.info(f"{name}: iter {it+1}/{n_iter}")
    savemat(os.path.join(outdir,f"{name}_dict.mat"),
            {"D":D.astype(np.float32),"A":A.astype(np.float32)})
    logging.info(f"{name}: saved")

# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap_dir", required=True)
    p.add_argument("--scale",         required=True)
    p.add_argument("--outdir",        required=True)
    p.add_argument("--n-iter",  type=int, default=10,
                   help="K-SVD iterations (MATLAB used 10)")
    p.add_argument("--n-jobs",  type=int, default=1)
    p.add_argument("--init_dir",      type=str)
    args = p.parse_args()
    setup_logger(args.outdir)

    # === Load dictionary scale info ===
    sc = loadmat(args.scale)
    K = int(sc["bestK"])
    L = int(sc["bestL"])
    logging.info(f"K={K}  L={L}  n_iter={args.n_iter}  n_jobs={args.n_jobs}")

    # === Determine which bootstraps need work ===
    mats = sorted(f for f in os.listdir(args.bootstrap_dir) if f.endswith(".mat"))
    todo = []
    for f in mats:
        name = f[:-4]
        outpath = os.path.join(args.outdir, f"{name}_dict.mat")
        if not os.path.exists(outpath):
            todo.append(os.path.join(args.bootstrap_dir, f))
        else:
            logging.info(f"✅ Skipping {name} (already done)")

    if not todo:
        logging.info("All bootstraps already processed. Nothing to do.")
        return

    logging.info(f"Processing {len(todo)} / {len(mats)} bootstraps ...")

    # === Parallel execution ===
    Parallel(n_jobs=args.n_jobs)(
        delayed(run_ksvd)(m, K, L, args.n_iter, args.outdir, init_dir=args.init_dir)
        for m in todo
    )

    logging.info(f"✅ Finished dictionaries for {len(todo)} new bootstraps "
                 f"(total now {len(mats)})")


if __name__ == "__main__":
    main()