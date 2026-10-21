#!/usr/bin/env python3
"""Sign-aware SPARK spatial-atom diagnostic; read-only on existing SPARK outputs.

Run: python sign_diagnostic.py --root /path/to/network-analysis

Uses original noncortical CIFTI data for time correlations, analyzes all saved
bootstrap coefficient maps; compares MDL-selected baseline against fixed-K/L test.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import nibabel as nib

REGIONS = ('THALAMUS', 'HIPPOCAMPUS', 'PUTAMEN', 'PALLIDUM',
           'CAUDATE', 'CEREBELLUM', 'BRAIN_STEM')
SUBJECT = 'rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_rclean_tclean_subcortical'
EXPERIMENTS = {
    'MDL_K2_L1': 'test_subcortical',
    'FIXED_K10_L2': 'test_subcortical_K10_L2_fixed',
}

def corr_stats(Y, indices_a, indices_b=None):
    if indices_b is None:
        if len(indices_a) < 5:
            return np.nan
        m = Y[:, indices_a].T @ Y[:, indices_a]
        return float(np.median(m[np.triu_indices(len(indices_a), k=1)]))
    if len(indices_a) < 5 or len(indices_b) < 5:
        return np.nan
    return float(np.median(Y[:, indices_a].T @ Y[:, indices_b]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--subject', default=SUBJECT)
    parser.add_argument('--top-n', type=int, default=30)
    args = parser.parse_args()
    if args.top_n < 5:
        parser.error('--top-n must be >= 5')
    root = args.root.expanduser().resolve()
    fmri = root / 'inputs' / (args.subject + '.dtseries.nii')
    img = nib.load(str(fmri))
    raw = np.asarray(img.dataobj, dtype=np.float32)
    T, V = raw.shape
    axis = img.header.get_axis(1)
    assert len(axis) == V
    # Column standardization makes the dot product equal Pearson correlation.
    x = raw.astype(np.float32, copy=True)
    del raw
    x -= x.mean(axis=0, keepdims=True)
    std = x.std(axis=0)
    valid = np.isfinite(std) & (std > 1e-10)
    x[:, valid] /= std[valid]
    x[:, ~valid] = 0
    x /= np.sqrt(T)
    region_indices = {}
    for region in REGIONS:
        parts = [np.arange(s.start, s.stop) for name, s, _ in axis.iter_structures()
                 if region in name.upper()]
        if parts:
            region_indices[region] = np.concatenate(parts)
    rows = []
    for experiment, folder in EXPERIMENTS.items():
        out = root / 'outputs' / folder / args.subject
        scale = sio.loadmat(out / 'scale.mat')
        K, L = int(scale['bestK'].item()), int(scale['bestL'].item())
        files = sorted((out / 'dict').glob('boot_*_dict.mat'))
        assert files, f'No dictionaries in {out}'
        print(f'{experiment}: K={K}, L={L}, bootstraps={len(files)}', flush=True)
        for boot_i, file in enumerate(files):
            A = np.asarray(sio.loadmat(file)['A'], dtype=np.float32)
            assert A.shape == (K, V), (file, A.shape, (K, V))
            for atom in range(K):
                for region, idx in region_indices.items():
                    v = A[atom, idx]
                    positive = idx[v > 0]
                    negative = idx[v < 0]
                    positive = positive[np.argsort(A[atom, positive])[-args.top_n:]]
                    negative = negative[np.argsort(A[atom, negative])[:args.top_n]]
                    # Exclude constant time series to avoid artificial zero correlations.
                    positive = positive[valid[positive]]
                    negative = negative[valid[negative]]
                    cp = corr_stats(x, positive)
                    cn = corr_stats(x, negative)
                    cb = corr_stats(x, positive, negative)
                    rows.append(dict(experiment=experiment, K=K, L=L,
                                     bootstrap=boot_i, atom=atom, region=region,
                                     n_positive=len(positive), n_negative=len(negative),
                                     positive_corr=cp, negative_corr=cn,
                                     cross_sign_corr=cb))
    details = pd.DataFrame(rows)
    details['mean_within_corr'] = details[['positive_corr', 'negative_corr']].mean(axis=1)
    details['sign_separation'] = details['mean_within_corr'] - details['cross_sign_corr']
    summary = (details.groupby(['experiment','region'], as_index=False)
               .agg(K=('K','first'), L=('L','first'),
                    n_atoms=('atom','count'),
                    valid_pairs=('cross_sign_corr','count'),
                    median_positive=('positive_corr','median'),
                    median_negative=('negative_corr','median'),
                    median_between=('cross_sign_corr','median'),
                    median_separation=('sign_separation','median')))
    save = root / 'outputs' / 'region_adaptive_diagnostics'
    save.mkdir(parents=True, exist_ok=True)
    details.to_csv(save / 'sign_coherence_details.csv', index=False)
    summary.to_csv(save / 'sign_coherence_summary.csv', index=False)
    print('\nSIGN-AWARE SUMMARY\n')
    print(summary.round(4).to_string(index=False))
    print('\nSaved to:', save)
    print('Caution: within-recording descriptive correlations; not independent validation.')

if __name__ == '__main__':
    main()
