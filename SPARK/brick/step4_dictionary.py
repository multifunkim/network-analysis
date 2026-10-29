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
        # Normalize D while preserving the paired factorization D @ A.
        norms = np.maximum(np.linalg.norm(D, axis=0), 1e-12)
        D = D / norms[None, :]
        A = A * norms[:, None]

        # Apply identical sign changes to D columns and A rows.
        signs = np.where(D[0, :] < 0, -1.0, 1.0)
        D = D * signs[None, :]
        A = A * signs[:, None]

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



def run_joint_cifti(subject_dir, n_boot, joint_l):
    """Combine regional dictionaries and fit whole-brain coefficients."""
    from pathlib import Path

    root = Path(subject_dir)
    outdir = root / "dict"
    outdir.mkdir(parents=True, exist_ok=True)

    def get_indices(path):
        data = loadmat(
            str(path),
            variable_names=["spatial_index"]
        )
        return np.asarray(
            data["spatial_index"]
        ).ravel().astype(np.int64)

    full_idx = get_indices(root / "tseries.mat")

    if not np.all(np.diff(full_idx) > 0):
        raise ValueError("Whole-brain indices must be ordered.")

    positions = {}

    for region in ("cortex", "noncortex"):
        idx = get_indices(root / region / "tseries.mat")
        pos = np.searchsorted(full_idx, idx)

        if (
            np.any(pos >= full_idx.size)
            or not np.array_equal(full_idx[pos], idx)
        ):
            raise ValueError(f"Invalid {region} index mapping.")

        positions[region] = pos

    combined = np.concatenate([
        positions["cortex"],
        positions["noncortex"]
    ])

    if not np.array_equal(
        np.sort(combined),
        np.arange(full_idx.size)
    ):
        raise ValueError("Incomplete regional partition.")

    for i in range(n_boot):

        name = f"boot_{i:03d}"
        output = outdir / f"{name}_dict.mat"

        if output.exists():
            old = loadmat(
                str(output),
                variable_names=["D", "A", "joint_L"]
            )

            if (
                "joint_L" not in old
                or int(old["joint_L"].item()) != joint_l
                or old["A"].shape[1] != full_idx.size
            ):
                raise ValueError(
                    f"Incompatible existing output: {output}"
                )

            print(f"[joint] Reusing {name}", flush=True)
            continue

        dictionaries = []

        for region in ("cortex", "noncortex"):
            path = root / region / "dict" / f"{name}_dict.mat"

            Dregion = loadmat(
                str(path),
                variable_names=["D"]
            )["D"]

            dictionaries.append(
                np.asarray(Dregion, dtype=np.float64)
            )

        D = np.concatenate(dictionaries, axis=1)

        T, K = D.shape

        if not 1 <= joint_l <= K:
            raise ValueError(f"Invalid joint L={joint_l} for K={K}")

        norms = np.linalg.norm(D, axis=0)

        if np.any(norms <= 1e-12):
            raise ValueError(f"Zero-norm atoms in {name}")

        D = D / norms[None, :]

        A = np.zeros(
            (K, full_idx.size),
            dtype=np.float32
        )

        # Each region uses its matching bootstrap realization.
        # Coefficients are placed in full-brain CIFTI order.
        for region in ("cortex", "noncortex"):

            path = root / region / "boot" / f"{name}.mat"

            X = loadmat(
                str(path),
                variable_names=["boot"]
            )["boot"]

            if X.shape != (T, len(positions[region])):
                raise ValueError(
                    f"Unexpected bootstrap dimensions: "
                    f"{region}/{name}: {X.shape}"
                )

            pos = positions[region]

            # Reuse existing Step 4 top-L sparse coding.
            for v in range(X.shape[1]):
                A[:, pos[v]] = mpf_topL(
                    X[:, v],
                    D,
                    joint_l
                )

            del X

        temp = outdir / f"{name}_dict.tmp.mat"

        savemat(
            str(temp),
            {
                "D": D.astype(np.float32),
                "A": A,
                "joint_L": np.array([[joint_l]]),
                "spatial_index": full_idx,
            }
        )

        temp.replace(output)

        print(
            f"[joint] Saved {name}: "
            f"D={D.shape}, A={A.shape}, L={joint_l}",
            flush=True
        )

if __name__ == "__main__":
    main()