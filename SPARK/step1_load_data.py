#!/usr/bin/env python3
import os, argparse, logging
import numpy as np
import nibabel as nib
from scipy.io import savemat

def setup_logger(out_mat):
    d = os.path.dirname(out_mat) or '.'
    os.makedirs(d, exist_ok=True)
    logf = os.path.join(d, 'step1_load_data.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[step1] %(asctime)s - INFO - %(message)s',
        handlers=[logging.FileHandler(logf), logging.StreamHandler()]
    )

def main():
    p = argparse.ArgumentParser(description="Step 1 – Load, preprocess & subsample fMRI")
    p.add_argument('--fmri',      required=True, help='4D NIfTI path')
    p.add_argument('--mask',      required=True, help='3D mask NIfTI')
    p.add_argument('--subsample', type=int,   default=1, help='Keep every Nth voxel')
    p.add_argument('--out',       required=True, help='Output .mat file')
    args = p.parse_args()

    setup_logger(args.out)
    logging.info(f"Starting data loading from {args.fmri}")
    img   = nib.load(args.fmri)
    data  = img.get_fdata()                     # (X,Y,Z,T)
    mask3 = nib.load(args.mask).get_fdata() > 0  # (X,Y,Z)
    logging.info(f"Loaded fMRI={img.shape}, mask={mask3.shape}")

    coords = np.column_stack(np.where(mask3))       # (V_full,3)
    flat   = mask3.reshape(-1)
    ts_full = data.reshape(-1, img.shape[-1])[flat].T  # (T, V_full)
    logging.info(f"Applied mask, tseries shape: {ts_full.shape}")

    var     = np.var(ts_full, axis=0)
    keep    = var > 1e-6
    ts_full = ts_full[:, keep]
    coords  = coords[keep]
    logging.info(f"After dropping zero‐variance voxels: {ts_full.shape}")

    mu      = ts_full.mean(axis=0, keepdims=True)
    sigma   = ts_full.std(axis=0, keepdims=True)
    ts_full = (ts_full - mu) / np.maximum(sigma, 1e-6)
    logging.info(f"Normalized tseries shape: {ts_full.shape}")

    if args.subsample > 1:
        idx_sub = np.arange(0, ts_full.shape[1], args.subsample)
        ts_sub  = ts_full[:, idx_sub]
        logging.info(f"Subsampled shape: {ts_sub.shape}")
    else:
        idx_sub = np.arange(ts_full.shape[1])
        ts_sub  = ts_full
        logging.info("No subsampling applied")

    savemat(args.out, {
        'tseries_full': ts_full,
        'tseries_sub':  ts_sub,
        'coords':       coords,
        'idx_sub':      idx_sub,
        'shape':        img.shape[:3],
        'affine':       img.affine
    })
    logging.info("Data loading completed")

if __name__ == '__main__':
    main()
