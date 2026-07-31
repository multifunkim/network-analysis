#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 — Final atoms + k-hubness (MATLAB-equivalent for pipeline)

Inputs (from pipeline_steps1_6.py):
  --clusters  <out>/clusters.mat      (expects 'centroids' with shape K×V)
  --tseries   <out>/tseries.mat       (expects 'coords' with shape V×3)
  --mask      <mask>.nii[.gz]         (defines 3D grid & affine)
  --pvalue    float (default 0.05)    (two-tailed: |z| > z_thr)
  --min_voxels int (default 30)       (drop atoms smaller than this)
  --outdir    <out>/KMAP_<SUBJ>
  --subject_label <SUBJ>

Outputs:
  <outdir>/atom_{kk}_{SUBJ}.nii.gz          (thresholded z, zeros elsewhere)
  <outdir>/k_hubness_{SUBJ}.nii.gz          (INT16 counts 0..K')
"""

import os
import argparse
import logging
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import norm


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[step6] %(asctime)s - %(levelname)s - %(message)s'
    )


def load_centroids(path: str) -> np.ndarray:
    m = loadmat(path)
    if 'centroids' not in m:
        raise KeyError(f"'centroids' not found in {path}")
    C = np.asarray(m['centroids'])
    if C.ndim != 2:
        raise ValueError(f"'centroids' must be 2D (K×V), got {C.shape}")
    return C.astype(np.float32, copy=False)


def load_coords(path: str) -> np.ndarray:
    m = loadmat(path)
    if 'coords' not in m:
        raise KeyError(f"'coords' not found in {path}")
    coords = np.asarray(m['coords']).astype(int)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"'coords' must be (V×3), got {coords.shape}")
    return coords


def rowwise_z(C: np.ndarray) -> np.ndarray:
    """Z-score each row of C (K×V); safe for zero-variance rows."""
    K, V = C.shape
    Z = np.empty((K, V), dtype=np.float32)
    for k in range(K):
        row = C[k]
        mu = float(row.mean())
        sd = float(row.std())
        if not np.isfinite(sd) or sd == 0.0:
            Z[k].fill(0.0)
        else:
            Z[k] = (row - mu) / sd
    return Z


def pin_header_scaling(img: nib.Nifti1Image, dtype) -> nib.Nifti1Image:
    """Force slope/intercept to 1/0 and set dtype to avoid viewer autoscaling."""
    hdr = img.header
    hdr.set_data_dtype(dtype)
    hdr['scl_slope'] = 1.0
    hdr['scl_inter'] = 0.0
    return img


def main():
    ap = argparse.ArgumentParser(description="Step 6 – Compute atoms and k-hubness (MATLAB-style)")
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--tseries',  required=True)
    ap.add_argument('--mask',     required=True)
    ap.add_argument('--pvalue',   type=float, default=0.05)
    ap.add_argument('--min_voxels', type=int, default=30)
    ap.add_argument('--outdir',   required=True)
    ap.add_argument('--subject_label', required=True)
    args = ap.parse_args()

    setup_logger()
    os.makedirs(args.outdir, exist_ok=True)

    # Load inputs
    C       = load_centroids(args.clusters)      # (K, V)
    coords  = load_coords(args.tseries)          # (V, 3)
    maskimg = nib.load(args.mask)
    shape   = maskimg.shape
    affine  = maskimg.affine

    K, V = C.shape
    if coords.shape[0] != V:
        raise ValueError(f"V mismatch: centroids V={V}, coords V={coords.shape[0]}")

    # coords → flat indices on the mask grid
    ravel_idx = np.ravel_multi_index(coords.T, dims=shape)

    # Two-tailed threshold
    z_thr = float(norm.ppf(1.0 - args.pvalue / 2.0))
    logging.info(f"Z-threshold (two-tailed p={args.pvalue:.4g}): |z| > {z_thr:.3f}")

    # Z-score rows, then threshold → boolean selection (K×V)
    Z   = rowwise_z(C)
    SEL = (np.abs(Z) > z_thr)

    # Filter atoms by min_voxels (count in selected V only)
    kept = np.flatnonzero(SEL.sum(axis=1) >= int(args.min_voxels))
    logging.info(f"Kept atoms: {kept.size}/{K} (min_voxels={args.min_voxels})")

    # Save each kept atom as thresholded z-map (zeros elsewhere)
    for k in kept:
        sel  = SEL[k]               # (V,)
        zrow = Z[k]                 # (V,)
        flat = np.zeros(np.prod(shape), dtype=np.float32)
        if sel.any():
            flat[ravel_idx[sel]] = zrow[sel]
        atom_img = nib.Nifti1Image(flat.reshape(shape), affine)
        atom_img = pin_header_scaling(atom_img, np.float32)
        atom_path = os.path.join(args.outdir, f'atom_{k:02d}_{args.subject_label}.nii.gz')
        nib.save(atom_img, atom_path)
        logging.info(f"Saved atom → {atom_path} (voxels kept: {int(sel.sum())})")

    # k‑hubness = per‑voxel count across kept atoms (sum over rows → V)
    hub_counts_V = SEL[kept].sum(axis=0).astype(np.int32) if kept.size else np.zeros(V, dtype=np.int32)
    hub_flat     = np.zeros(np.prod(shape), dtype=np.int32)
    hub_flat[ravel_idx] = hub_counts_V
    hub3d = hub_flat.reshape(shape)

    # Save k‑map as INT16 with fixed scaling so viewers show integers 0..K'
    k_img = nib.Nifti1Image(hub3d.astype(np.int16, copy=False), affine)
    # carry q/sform from mask for safety
    k_img.set_qform(maskimg.get_qform(), code=int(maskimg.header.get('qform_code')))
    k_img.set_sform(maskimg.get_sform(), code=int(maskimg.header.get('sform_code')))
    k_img = pin_header_scaling(k_img, np.int16)
    hdr = k_img.header
    hdr['cal_min'] = 0
    hdr['cal_max'] = int(hub_counts_V.max() if hub_counts_V.size else 0)

    k_path = os.path.join(args.outdir, f'k_hubness_{args.subject_label}.nii.gz')
    nib.save(k_img, k_path)
    logging.info(f"Saved k‑hubness → {k_path}  (max={int(hub_counts_V.max() if hub_counts_V.size else 0)})")


if __name__ == '__main__':
    main()
