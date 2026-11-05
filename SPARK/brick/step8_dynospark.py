#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 8 — dynoSPARK

Phase 1:
    Discover the existing SPARK output layout and verify
    the inputs required for temporal reconstruction.

No temporal reconstruction is performed yet.

Supported layouts
-----------------
- NIfTI / GIFTI standard SPARK:
      subject/
          tseries.mat
          clusters.mat
          dict/

- CIFTI whole:
      subject/
          tseries.mat
          clusters.mat
          dict/

- CIFTI regional:
      subject/
          cortex/
          noncortex/

  including:
      cortex only
      noncortex only
      both
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from dyno_signed_atoms import save_signed_atoms
from dyno_kappa import reconstruct_unit
from scipy.io import savemat
import json


# ============================================================
# HELPERS
# ============================================================

def matlab_string(value):

    if value is None:
        return None

    while isinstance(value, np.ndarray):

        if value.size == 0:
            return None

        if value.dtype.kind in ("U", "S"):
            return str(value.ravel()[0])

        value = value.ravel()[0]

    if isinstance(value, bytes):
        value = value.decode()

    return str(value)


def is_analysis_unit(path):
    """
    A completed SPARK analysis unit must contain:
        clusters.mat
        dict/boot_000_dict.mat
    """

    return (
        (path / "clusters.mat").is_file()
        and
        (path / "dict" / "boot_000_dict.mat").is_file()
    )


def discover_units(subject_dir, input_format):

    # --------------------------------------------------------
    # Root-level analysis:
    # NIfTI, GIFTI, or CIFTI whole
    # --------------------------------------------------------

    if is_analysis_unit(subject_dir):

        name = (
            "whole"
            if input_format == "cifti"
            else "standard"
        )

        return [
            {
                "name": name,
                "dir": subject_dir,
                "tseries": subject_dir / "tseries.mat",
            }
        ]

    # --------------------------------------------------------
    # Regional CIFTI
    # --------------------------------------------------------

    units = []

    for region in (
        "cortex",
        "noncortex",
    ):

        region_dir = (
            subject_dir / region
        )

        if not is_analysis_unit(
            region_dir
        ):
            continue

        region_tseries = (
            region_dir / "tseries.mat"
        )

        if not region_tseries.is_file():

            # Useful for possible single-compartment layouts.
            region_tseries = (
                subject_dir / "tseries.mat"
            )

        units.append(
            {
                "name": region,
                "dir": region_dir,
                "tseries": region_tseries,
            }
        )

    if not units:

        raise RuntimeError(
            "Could not identify a valid SPARK analysis layout."
        )

    return units


# ============================================================
# VALIDATE ONE ANALYSIS UNIT
# ============================================================

def inspect_unit(unit):

    name = unit["name"]
    unit_dir = unit["dir"]
    tseries_path = unit["tseries"]

    dict_path = (
        unit_dir
        / "dict"
        / "boot_000_dict.mat"
    )

    clusters_path = (
        unit_dir
        / "clusters.mat"
    )

    if not tseries_path.is_file():
        raise FileNotFoundError(
            tseries_path
        )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    ts = loadmat(
        tseries_path,
        variable_names=[
            "tseries_full",
        ],
    )

    if "tseries_full" not in ts:
        raise KeyError(
            f"{tseries_path}: missing tseries_full"
        )

    X = np.asarray(
        ts["tseries_full"]
    )

    # --------------------------------------------------------
    # Original chronological dictionary
    # --------------------------------------------------------

    original = loadmat(
        dict_path,
        variable_names=[
            "D",
            "A",
        ],
    )

    D = np.asarray(
        original["D"]
    )

    A0 = np.asarray(
        original["A"]
    )

    # --------------------------------------------------------
    # Final Step-5 clusters
    # --------------------------------------------------------

    clusters = loadmat(
        clusters_path,
        variable_names=[
            "labels",
            "centroids",
        ],
    )

    labels = np.asarray(
        clusters["labels"]
    ).ravel()

    centroids = np.asarray(
        clusters["centroids"]
    )

    # --------------------------------------------------------
    # Dimension checks
    # --------------------------------------------------------

    if X.ndim != 2:
        raise ValueError(
            f"{name}: X is not T×V: {X.shape}"
        )

    if D.ndim != 2:
        raise ValueError(
            f"{name}: D is not T×K: {D.shape}"
        )

    if A0.ndim != 2:
        raise ValueError(
            f"{name}: A is not K×V: {A0.shape}"
        )

    if centroids.ndim != 2:
        raise ValueError(
            f"{name}: centroids are not K×V: "
            f"{centroids.shape}"
        )

    if X.shape[0] != D.shape[0]:
        raise ValueError(
            f"{name}: X and D have different T."
        )

    if X.shape[1] != A0.shape[1]:
        raise ValueError(
            f"{name}: X and original A have different V."
        )

    if X.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"{name}: X and final centroids have different V."
        )

    if labels.size % A0.shape[0] != 0:
        raise ValueError(
            f"{name}: Step-5 labels do not match "
            "the Step-4 dictionary size."
        )

    n_boot = (
        labels.size
        //
        A0.shape[0]
    )

    print()
    print("--------------------------------------------------")
    print(f"Analysis unit : {name}")
    print(f"Directory     : {unit_dir}")
    print(f"X             : {X.shape}")
    print(f"D_origin      : {D.shape}")
    print(f"A_origin      : {A0.shape}")
    print(f"Final clusters: {centroids.shape}")
    print(f"Cluster labels: {labels.shape}")
    print(f"Bootstraps    : {n_boot}")
    print("Status        : OK")
    print("--------------------------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "dynoSPARK Step 8 — input/layout validation"
        )
    )

    parser.add_argument(
        "--subject_dir",
        required=True,
    )

    parser.add_argument(
        "--signed-atoms",
        action="store_true",
        help="Build signed final SPARK atoms.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dyno signed-atom outputs.",
    )

    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Estimate kappa and reconstruct signed atom time series.",
    )

    parser.add_argument(
        "--lambda-ratio",
        type=float,
        default=0.05,
        help="L1 penalty relative to max |D.T X A.T|.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
    )

    args = parser.parse_args()

    subject_dir = Path(
        args.subject_dir
    ).resolve()

    if not subject_dir.is_dir():
        raise FileNotFoundError(
            subject_dir
        )

    root_tseries = (
        subject_dir / "tseries.mat"
    )

    if not root_tseries.is_file():
        raise FileNotFoundError(
            root_tseries
        )

    metadata = loadmat(
        root_tseries,
        variable_names=[
            "input_format",
        ],
    )

    input_format = matlab_string(
        metadata.get(
            "input_format"
        )
    )

    if input_format not in (
        "nifti",
        "gifti",
        "cifti",
    ):

        raise ValueError(
            f"Unknown input format: {input_format}"
        )

    print()
    print("==============================================")
    print("dynoSPARK Step 8")
    print("Phase 1 — layout/input validation")
    print("==============================================")
    print(f"Subject : {subject_dir.name}")
    print(f"Format  : {input_format}")

    units = discover_units(
        subject_dir,
        input_format,
    )

    print(
        "Units   : "
        +
        ", ".join(
            unit["name"]
            for unit in units
        )
    )

    for unit in units:

        inspect_unit(
            unit
        )

    print()
    print("All dynoSPARK input checks passed.")

    if args.signed_atoms:

        dyno_dir = subject_dir / "dyno"

        print()
        print("Phase 2 — signed final spatial atoms")

        for unit in units:

            name = unit["name"]

            output = dyno_dir / f"signed_atoms_{name}.mat"

            result = save_signed_atoms(
                unit_dir=unit["dir"],
                output_path=output,
                overwrite=args.overwrite,
            )

            print()
            print(f"Unit             : {name}")
            print(f"A_signed         : {result['A_signed'].shape}")
            print(f"Cluster members  : {result['member_counts'].sum()}")
            print(f"Sign flips       : {result['flip_counts'].sum()}")
            print(f"Mean |cosine|    : {result['mean_abs_cosine'].mean():.4f}")
            print(f"Saved            : {output}")

        print()
        print("Phase 2 completed.")



    if args.reconstruct:

        dyno_dir = subject_dir / "dyno"

        # Verify all inputs and protect existing results
        # before beginning an expensive reconstruction.
        outputs = [
            dyno_dir / f"dyno_{unit['name']}.mat"
            for unit in units
        ]

        outputs += [
            dyno_dir / "dyno_timeseries.mat",
            dyno_dir / "reconstruction_metrics.json",
            dyno_dir / "dyno_metadata.json",
        ]

        for unit in units:
            signed_path = (
                dyno_dir / f"signed_atoms_{unit['name']}.mat"
            )
            if not signed_path.is_file():
                raise FileNotFoundError(
                    f"Missing {signed_path}. Run --signed-atoms first."
                )

        if not args.overwrite:
            for output in outputs:
                if output.exists():
                    raise FileExistsError(
                        f"{output} exists; use --overwrite."
                    )

        print()
        print("Phase 3 — temporal reconstruction")

        results = []
        metrics = {}

        for unit in units:

            name = unit["name"]
            signed_path = dyno_dir / f"signed_atoms_{name}.mat"

            D_final, kappa, info = reconstruct_unit(
                unit,
                signed_path,
                lambda_ratio=args.lambda_ratio,
                max_iter=args.max_iter,
                tol=args.tol,
            )

            results.append((name, D_final, kappa))
            metrics[name] = info

            print()
            print(f"Unit         : {name}")
            print(f"D_final      : {D_final.shape}")
            print(f"kappa        : {kappa.shape}")
            print(f"Iterations   : {info['iterations']}")
            print(f"Converged    : {info['converged']}")
            print(f"KKT residual : {info['kkt_scaled_inf']:.3e}")
            print(
                "Energy explained: "
                f"{info['fraction_energy_explained']:.4f}"
            )

        dyno_dir.mkdir(parents=True, exist_ok=True)

        # Save unit-specific outputs.
        for name, D_final, kappa in results:
            savemat(
                dyno_dir / f"dyno_{name}.mat",
                {
                    "D_all_clusters": D_final,
                    "kappa": kappa,
                    "qc_status": "unreviewed",
                },
            )

        # Concatenate units along the atom dimension.
        combined = np.hstack([
            result[1] for result in results
        ])

        unit_ranges = []
        offset = 0

        for name, D_final, _ in results:
            end = offset + D_final.shape[1]
            unit_ranges.append((offset, end))
            offset = end

        savemat(
            dyno_dir / "dyno_timeseries.mat",
            {
                "D_all_clusters": combined,
                "D_timeseries": combined,
                "unit_names": np.asarray(
                    [r[0] for r in results],
                    dtype=object,
                ),
                "unit_ranges_zero_based": np.asarray(
                    unit_ranges,
                    dtype=np.int64,
                ),
                "qc_status": "unreviewed",
            },
        )

        with open(
            dyno_dir / "reconstruction_metrics.json",
            "w",
        ) as handle:
            json.dump(metrics, handle, indent=2)

        with open(
            dyno_dir / "dyno_metadata.json",
            "w",
        ) as handle:
            json.dump(
                {
                    "input_format": input_format,
                    "units": [r[0] for r in results],
                    "method": "FISTA_L1",
                    "lambda_ratio": args.lambda_ratio,
                    "source_dictionary": "boot_000",
                    "temporal_order": "original",
                    "signed_timecourses": True,
                    "qc_status": "unreviewed",
                },
                handle,
                indent=2,
            )

        print()
        print(f"Combined D : {combined.shape}")
        print(f"Saved to   : {dyno_dir}")
        print("Phase 3 completed.")


if __name__ == "__main__":
    main()
