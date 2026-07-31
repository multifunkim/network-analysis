#!/usr/bin/env python3
import os, argparse, logging
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.cluster import KMeans

def setup_logger(out_mat):
    d = os.path.dirname(out_mat) or '.'
    os.makedirs(d, exist_ok=True)
    logf = os.path.join(d, 'step5_clustering.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[step5] %(asctime)s - INFO - %(message)s',
        handlers=[logging.FileHandler(logf), logging.StreamHandler()]
    )

def main():
    p = argparse.ArgumentParser(description="Step 5 – Cluster spatial maps across bootstraps")
    p.add_argument('--dicts', nargs='+', required=True,
                   help='list of *_dict.mat files from Step 4')
    p.add_argument('--scale', required=True, help='scale.mat from Step 2')
    p.add_argument('--out',   required=True, help='output clusters.mat')
    args = p.parse_args()

    setup_logger(args.out)
    # load bestK
    sc = loadmat(args.scale)
    K  = int(sc['bestK'].squeeze())
    logging.info(f"Clustering into K={K} spatial networks")

    # load & stack |A| from each bootstrap
    maps = []
    for dfile in args.dicts:
        M = loadmat(dfile)['A']        # (K, V)
        maps.append(np.abs(M))         # absolute value
    ConX = np.vstack(maps)            # (B*K, V)
    logging.info(f"Stacked maps array shape: {ConX.shape}")

    # k-means
    km = KMeans(n_clusters=K, random_state=0)
    labels    = km.fit_predict(ConX)           # (B*K,)
    centroids = km.cluster_centers_            # (K, V)
    logging.info("K-means clustering converged")

    # save
    savemat(args.out, {
        'labels':    labels,
        'centroids': centroids
    })
    logging.info(f"Saved clusters → {args.out}")

if __name__ == '__main__':
    main()
