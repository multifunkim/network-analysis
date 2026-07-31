#!/usr/bin/env python3
"""
step3_bootstrap.py  •  original + circular-block surrogates
-----------------------------------------------------------
The first file (boot_000.mat) is the untouched time-series.
--n-boot now means TOTAL files (original + surrogates).
"""

import os, argparse, logging, numpy as np
from scipy.io import loadmat, savemat

def setup_logger(outdir):
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[step3] %(asctime)s  %(message)s',
        handlers=[logging.FileHandler(os.path.join(outdir,"step3_bootstrap.log")),
                  logging.StreamHandler()])

def circ_block(X, L):
    """One circular-block resample of X (T,V)."""
    T = X.shape[0]
    idx = np.random.randint(0, T-L+1, size=(T//L + 1))
    return np.concatenate([X[i:i+L] for i in idx], axis=0)[:T]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tseries",   required=True)
    p.add_argument("--block-len", type=int, required=True)
    p.add_argument("--n-boot",    type=int, required=True,
                   help="TOTAL files to write, incl. original")
    p.add_argument("--outdir",    required=True)
    args = p.parse_args(); setup_logger(args.outdir)

    X = loadmat(args.tseries)["tseries_full"]          # (T,V)
    savemat(os.path.join(args.outdir,"boot_000.mat"), {"boot":X})
    logging.info("Saved boot_000.mat (original)")

    for b in range(1, args.n_boot):
        savemat(os.path.join(args.outdir, f"boot_{b:03d}.mat"),
                {"boot": circ_block(X, args.block_len)})
    logging.info(f"✅ Wrote {args.n_boot} files (1 original + {args.n_boot-1} resamples)")

if __name__ == "__main__":
    main()
