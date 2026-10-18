"""Deterministic feasibility checks for normalized-score calibration dynamics.

No training, observed market data, fitted parameter, bootstrap or novel empirical
claim is involved. The fixed exogenous error trajectories have identical value
multisets and different temporal order. This is not a deployable forecaster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "compatibility_math_probe"
ALPHA = 0.1
Z = float(norm.ppf(1 - ALPHA / 2))
N_PHASES = 256
WINDOW = 32
FREQUENCIES = (1, 17)  # Both coprime with 256: exactly the same value multiset.
AMPLITUDES = (0.2, 0.1, 0.05, 0.025, 0.0125)


def abs_cdf(x):
    return 2 * ndtr(x) - 1


def risk(halfwidth):
    """Exact E[IS_alpha] for y ~ N(0, 1), symmetric zero-centered interval."""
    a = np.asarray(halfwidth)
    return 2 * a + 4 / ALPHA * (norm.pdf(a) - a * norm.sf(a))


def mixture_quantile(archived_errors):
    """Population quantile, not E[the finite-window empirical quantile]."""
    low = Z * np.exp(-np.max(archived_errors))
    high = Z * np.exp(-np.min(archived_errors))
    if high - low < 1e-14:
        return float((low + high) / 2)
    return float(brentq(
        lambda q: np.mean(abs_cdf(q * np.exp(archived_errors))) - (1 - ALPHA),
        low, high, xtol=1e-13, rtol=1e-14,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="New/empty output directory. Existing results are never overwritten.")
    out = parser.parse_args().output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; use --output with a new/empty directory.")
    out.mkdir(parents=True, exist_ok=True)
    density = 2 * norm.pdf(Z)  # Density of |epsilon|, not of epsilon.
    a_coefficient = 1 - Z**2  # 1 + z f'(z) / f(z).
    log_risk_coefficient = Z**2 * density / ALPHA
    phases = np.arange(N_PHASES)
    ages = np.arange(1, WINDOW + 1)
    rows, arrays = [], {}
    max_root_residual = 0.0
    native_tie_max_error = 0.0
    for amplitude in AMPLITUDES:
        native_metrics = []
        for frequency in FREQUENCIES:
            # The explicit modulo makes the matching value multisets exact.
            delta = amplitude * np.sin(2 * np.pi * ((frequency * phases) % N_PHASES) / N_PHASES)
            archive = delta[(phases[:, None] - ages[None, :]) % N_PHASES]
            mu = archive.mean(axis=1)
            variance = archive.var(axis=1)
            q = np.array([mixture_quantile(row) for row in archive])
            xi = delta + np.log(q / Z)
            first = delta - mu
            second = first - 0.5 * a_coefficient * variance
            halfwidth = Z * np.exp(xi)
            native_halfwidth = Z * np.exp(delta)
            residual = np.max(np.abs(np.mean(abs_cdf(q[:, None] * np.exp(archive)), axis=1) - (1 - ALPHA)))
            max_root_residual = max(max_root_residual, float(residual))
            native_metrics.append(np.array([
                risk(native_halfwidth).mean(), abs_cdf(native_halfwidth).mean(),
                (2 * native_halfwidth).mean(),
                (delta + 0.5 * np.exp(-2 * delta)).mean(),  # NLL without common constant.
            ]))
            rows.append({
                "amplitude": amplitude, "frequency_cycles_per_256": frequency,
                "window": WINDOW, "native_expected_interval_score": float(risk(native_halfwidth).mean()),
                "calibrated_expected_interval_score": float(risk(halfwidth).mean()),
                "oracle_expected_interval_score": float(risk(Z)),
                "native_expected_coverage": float(abs_cdf(native_halfwidth).mean()),
                "calibrated_expected_coverage": float(abs_cdf(halfwidth).mean()),
                "native_gaussian_nll_without_constant": float(native_metrics[-1][-1]),
                "first_order_max_log_halfwidth_error": float(np.max(np.abs(xi - first))),
                "second_order_max_log_halfwidth_error": float(np.max(np.abs(xi - second))),
                "exact_calibrated_excess_risk": float(risk(halfwidth).mean() - risk(Z)),
                "quadratic_calibrated_excess_risk": float(log_risk_coefficient * np.mean(first**2)),
                "exact_native_excess_risk": float(risk(native_halfwidth).mean() - risk(Z)),
                "quadratic_native_excess_risk": float(log_risk_coefficient * np.mean(delta**2)),
                "first_order_mean_square_error_ratio": float(np.mean(first**2) / np.mean(delta**2)),
                "population_cdf_root_max_residual": float(residual),
            })
            arrays[f"amplitude_{amplitude}_frequency_{frequency}"] = np.column_stack((delta, mu, variance, q, xi, first, second))
        tie_error = float(np.max(np.abs(native_metrics[0] - native_metrics[1])))
        native_tie_max_error = max(native_tie_max_error, tie_error)
        np.testing.assert_allclose(native_metrics[0], native_metrics[1], rtol=0, atol=2e-14)

    orders = {}
    for frequency in FREQUENCIES:
        selected = [row for row in rows if row["frequency_cycles_per_256"] == frequency]
        first_order = float(np.log2(selected[-2]["first_order_max_log_halfwidth_error"] / selected[-1]["first_order_max_log_halfwidth_error"]))
        second_order = float(np.log2(selected[-2]["second_order_max_log_halfwidth_error"] / selected[-1]["second_order_max_log_halfwidth_error"]))
        orders[str(frequency)] = {"first_approximation_error_order": first_order, "second_approximation_error_order": second_order}
        assert 1.8 < first_order < 2.2
        # A third-order remainder can decay faster at finite amplitudes (or when
        # its leading coefficient cancels); an upper-order cutoff is not valid.
        assert second_order > 2.8

    cancellation_errors = []
    for constant in (-0.7, 0.0, 0.7):
        q = mixture_quantile(np.full(WINDOW, constant))
        cancellation_errors.append(abs(np.exp(constant) * q - Z))
    # Finite empirical constant rescaling also cancels exactly, for every rank.
    base_s = np.exp(0.1 * np.cos(np.arange(WINDOW)))
    abs_y = 0.1 + np.abs(np.sin(np.arange(WINDOW) + 0.3))
    current_s = 1.17
    base_h = current_s * np.sort(abs_y / base_s)
    for multiplier in (0.3, 1.0, 3.0):
        cancellation_errors.extend(np.abs(current_s * multiplier * np.sort(abs_y / (base_s * multiplier)) - base_h))
    assert max(cancellation_errors) < 2e-14
    assert max_root_residual < 1e-12

    # Independent finite differences verify the |epsilon|-density factor of two.
    cdf_step = 1e-5
    numerical_density = (abs_cdf(Z + cdf_step) - abs_cdf(Z - cdf_step)) / (2 * cdf_step)
    log_step = 1e-4
    numerical_log_risk_coefficient = (risk(Z * np.exp(log_step)) + risk(Z * np.exp(-log_step)) - 2 * risk(Z)) / (2 * log_step**2)
    np.testing.assert_allclose(numerical_density, density, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(numerical_log_risk_coefficient, log_risk_coefficient, rtol=1e-6, atol=1e-6)

    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "trajectories.npz", **arrays)
    manifest = {
        "description": "Deterministic analytic feasibility; no learned model or new empirical confirmation.",
        "study_status": "Exploratory algebra demonstration; frequencies and amplitudes were not preregistered and novelty is not established.",
        "alpha": ALPHA, "z": Z, "abs_innovation_density_at_z": float(density),
        "coefficient_1_plus_z_fprime_over_f": float(a_coefficient),
        "second_order_variance_correction_coefficient": float(-a_coefficient / 2),
        "log_halfwidth_quadratic_risk_coefficient": float(log_risk_coefficient),
        "numerical_log_risk_coefficient": float(numerical_log_risk_coefficient),
        "amplitudes": AMPLITUDES, "frequencies_cycles_per_256": FREQUENCIES,
        "phases": N_PHASES, "window": WINDOW, "periodic_past": True,
        "arrays_column_order": ["delta_current", "delta_archive_mean", "delta_archive_variance", "q_population", "log_relative_halfwidth_exact", "first_order", "second_order"],
        "native_risk_coverage_width_nll_tie_max_error": native_tie_max_error,
        "constant_rescaling_max_error": float(max(cancellation_errors)),
        "population_root_max_cdf_residual": max_root_residual,
        "observed_asymptotic_orders_final_two_amplitudes": orders,
        "checks_passed": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ("summary.csv", "trajectories.npz")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
