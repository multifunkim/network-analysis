#!/usr/bin/env python3
"""dynoSPARK — sparse temporal reconstruction using FISTA."""

import numpy as np
from scipy.io import loadmat


def solve_kappa(Gd, Ga, B, lambda_ratio=0.05,
                max_iter=5000, tol=1e-6):
    """
    Minimize:
        0.5 ||X - D @ kappa @ A||_F^2
        + lambda ||kappa||_1

    Gd = D.T @ D
    Ga = A @ A.T
    B  = D.T @ X @ A.T

    Only small sufficient-statistic matrices are used
    during optimization.
    """
    if not 0 <= lambda_ratio < 1:
        raise ValueError("lambda_ratio must be in [0, 1).")
    if max_iter < 1 or tol <= 0:
        raise ValueError("Invalid optimizer settings.")

    L = (
        np.linalg.eigvalsh(Gd)[-1]
        * np.linalg.eigvalsh(Ga)[-1]
    )

    if not np.isfinite(L) or L <= 0:
        raise ValueError("Invalid Lipschitz constant.")

    lam = lambda_ratio * np.max(np.abs(B))

    K = np.zeros_like(B, dtype=np.float64)
    Y = K.copy()
    momentum = 1.0

    converged = False
    kkt_error = np.inf

    for iteration in range(1, max_iter + 1):

        gradient = Gd @ Y @ Ga - B
        Z = Y - gradient / L

        K_new = (
            np.sign(Z)
            * np.maximum(np.abs(Z) - lam / L, 0.0)
        )

        next_momentum = (
            1.0 + np.sqrt(1.0 + 4.0 * momentum**2)
        ) / 2.0

        Y = (
            K_new
            + (momentum - 1.0) / next_momentum
            * (K_new - K)
        )

        K = K_new
        momentum = next_momentum

        # Normalized KKT optimality residual.
        gradient = Gd @ K @ Ga - B
        active = np.abs(K) > 1e-10

        residual = np.where(
            active,
            np.abs(gradient + lam * np.sign(K)),
            np.maximum(np.abs(gradient) - lam, 0.0),
        )

        kkt_error = (
            float(np.max(residual))
            / max(1.0, float(np.max(np.abs(B))))
        )

        if kkt_error < tol:
            converged = True
            break

    return K, {
        "iterations": iteration,
        "converged": converged,
        "kkt_scaled_inf": kkt_error,
        "lambda_ratio": float(lambda_ratio),
        "lambda_absolute": float(lam),
    }


def reconstruct_unit(unit, signed_path, lambda_ratio=0.05,
                     max_iter=5000, tol=1e-6):
    """Fit every final atom using the original chronological data."""

    unit_dir = unit["dir"]

    X = np.asarray(
        loadmat(
            unit["tseries"],
            variable_names=["tseries_full"],
        )["tseries_full"]
    )

    D = np.asarray(
        loadmat(
            unit_dir / "dict" / "boot_000_dict.mat",
            variable_names=["D"],
        )["D"],
        dtype=np.float64,
    )

    A = np.asarray(
        loadmat(
            signed_path,
            variable_names=["A_signed"],
        )["A_signed"],
        dtype=np.float64,
    )

    if X.ndim != 2 or D.ndim != 2 or A.ndim != 2:
        raise ValueError("Expected 2D X, D, and A.")

    T, V = X.shape

    if D.shape[0] != T or A.shape[1] != V:
        raise ValueError("Incompatible X, D, A dimensions.")

    if not np.isfinite(D).all() or not np.isfinite(A).all():
        raise ValueError("Nonfinite dictionary or signed atom maps.")

    Gd = D.T @ D
    Ga = A @ A.T
    B = np.zeros((D.shape[1], A.shape[0]), dtype=np.float64)

    x_energy = 0.0

    # Time-block processing prevents a second full-size
    # float64 copy of the fMRI data.
    for start in range(0, T, 96):
        end = min(start + 96, T)

        block = np.asarray(
            X[start:end],
            dtype=np.float64,
        )

        if not np.isfinite(block).all():
            raise ValueError("Nonfinite input fMRI samples.")

        B += D[start:end].T @ (block @ A.T)

        x_energy += float(
            np.einsum("ij,ij->", block, block)
        )

    if x_energy <= 0:
        raise ValueError("Input data have zero energy.")

    kappa, solver = solve_kappa(
        Gd, Ga, B,
        lambda_ratio=lambda_ratio,
        max_iter=max_iter,
        tol=tol,
    )

    # Signed final temporal expression.
    D_final = D @ kappa

    # Reconstruction error without forming the huge X_hat.
    fitted_energy = float(
        np.sum(kappa * (Gd @ kappa @ Ga))
    )

    cross_term = float(np.sum(kappa * B))

    residual_energy = max(
        0.0,
        x_energy - 2.0 * cross_term + fitted_energy,
    )

    metrics = {
        **solver,
        "T": int(T),
        "V": int(V),
        "K_origin": int(D.shape[1]),
        "K_final": int(A.shape[0]),
        "relative_reconstruction_error": float(
            np.sqrt(residual_energy / x_energy)
        ),
        "fraction_energy_explained": float(
            1.0 - residual_energy / x_energy
        ),
        "nonzero_kappa": int(
            np.count_nonzero(np.abs(kappa) > 1e-8)
        ),
    }

    return D_final, kappa, metrics
