"""Audit saved finite-mechanism artifacts without regenerating study outcomes.

Bootstrap intervals are independently recomputed from saved path averages using
multinomial multiplicity matrices, rather than the runner's indexed-array means.
No innovations, trajectories, mixture roots, or models are regenerated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta, norm, t

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("coverage", "width", "interval_score")
CASE = ("distribution", "template", "amplitude", "span", "frequency")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_weights(n, alpha, rule):
    target = (1 - alpha) * (n + 1)
    lo, hi = int(np.floor(target)), int(np.ceil(target))
    return [(hi, 1.)] if rule == "ceiling" or lo == hi else [(lo, hi-target), (hi, target-lo)]


def distribution(name):
    return norm if name == "gaussian" else t(5, scale=np.sqrt(3/5))


def scalar_risk(a, name, alpha):
    if name == "gaussian":
        tail = norm.pdf(a) - a * norm.sf(a)
    else:
        scale = np.sqrt(3/5)
        x = a / scale
        tail = scale * (5 + x*x) / 4 * t.pdf(x, 5) - a * t.sf(x, 5)
    return 2*a + 4/alpha*tail


def grid(frame, keys):
    return set(frame[list(keys)].itertuples(index=False, name=None))


def independent_bootstrap(values, repetitions, seed, levels):
    """One multiplicity vector per bootstrap replicate, paired across cases."""
    ncases, npaths, nmetrics = values.shape
    rng = np.random.default_rng(seed)
    flattened = values.transpose(1, 0, 2).reshape(npaths, -1)
    means = np.empty((repetitions, ncases, nmetrics))
    for begin in range(0, repetitions, 64):
        count = min(64, repetitions - begin)
        draw = rng.integers(npaths, size=(count, npaths))
        multiplicities = np.vstack([np.bincount(indices, minlength=npaths) for indices in draw])
        means[begin:begin+count] = (multiplicities @ flattened / npaths).reshape(count, ncases, nmetrics)
    return {level: np.quantile(means, [(1-level)/2, 1-(1-level)/2], axis=0) for level in levels}


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    p = manifest["protocol"]
    source_checks = artifact_checks = 0
    for relative, expected in manifest["sources"].items():
        assert sha(ROOT / relative) == expected, relative
        frozen = subprocess.check_output(["git", "show", f"{manifest['freeze_commit']}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest() == expected, relative
        source_checks += 1
    assert json.loads((ROOT / "configs/finite_mechanism_protocol.json").read_text()) == p
    for filename, expected in manifest["outputs"].items():
        assert sha(directory / filename) == expected, filename
        artifact_checks += 1
    frames = {name: pd.read_csv(directory / f"{name}.csv") for name in (
        "summary", "population_phases", "quadrature_checks", "primary_contrasts",
        "endogenous_boundary", "oracle_controls", "path_index")}
    summary, phases, numerics = (frames[name] for name in ("summary", "population_phases", "quadrature_checks"))
    primary, boundary, controls, index = (frames[name] for name in ("primary_contrasts", "endogenous_boundary", "oracle_controls", "path_index"))
    expected_cases = set(itertools.product(p["distributions"], p["templates"], p["amplitudes"], p["history_spans"], p["frequencies"]))
    assert len(expected_cases) == 48
    for source in ("native", "population", "first_order_population", "second_order_population"):
        chosen = summary[summary.source == source]
        assert len(chosen) == 48 and grid(chosen, CASE) == expected_cases
        assert set(chosen.n) == {0} and set(chosen.rule) == {"population"}
    for source in ("iid_mixture", "contiguous", "first_order_plus_noise"):
        chosen = summary[summary.source == source]
        expected = {(*case, n, rule) for case in expected_cases for n in (p["iid_bank_counts"] if source == "iid_mixture" else [case[3]]) for rule in p["rules"]}
        assert len(chosen) == len(expected) and grid(chosen, (*CASE, "n", "rule")) == expected
    assert len(summary) == 576 and not summary.duplicated([*CASE, "source", "n", "rule"]).any()
    assert len(phases) == 3072 and grid(phases, CASE) == expected_cases
    assert set(phases.groupby(list(CASE)).size()) == {64}
    np.testing.assert_allclose(phases.first_order_log_error, phases.current_delta-phases.archive_mean, atol=1e-15, rtol=0)
    for name in p["distributions"]:
        selected = phases[phases.distribution == name]
        z = distribution(name).ppf(1-p["alpha"]/2)
        slope = -z if name == "gaussian" else -6*z/(3+z*z)
        np.testing.assert_allclose(selected.second_order_log_error, selected.first_order_log_error-.5*(1+z*slope)*selected.archive_variance, atol=1e-14, rtol=0)
        oracle = scalar_risk(z, name, p["alpha"])
        assert summary.loc[(summary.distribution == name) & (summary.source != "first_order_plus_noise"), "interval_score"].min() >= oracle - 1e-10
    for _, frame in summary[summary.source == "native"].groupby(["distribution", "template", "amplitude", "span"]):
        assert len(frame) == 2
        np.testing.assert_allclose(frame[list(METRICS)].iloc[0], frame[list(METRICS)].iloc[1], atol=1e-12, rtol=0)
    assert len(numerics) == manifest["numerical_cases"] == 200
    assert (numerics.root_cdf_max_residual < p["numerics"]["root_tolerance"]).all()
    assert (numerics.maximum_phase_metric_change <= p["numerics"]["maximum_phase_metric_difference"]).all()
    assert set(numerics.quadrature_nodes).issubset({64, 128, 256})
    assert numerics.maximum_phase_metric_change.max() == manifest["maximum_quadrature_metric_change"]
    assert manifest["maximum_constant_error"] < 1e-11
    assert manifest["checks"]["numerical_engine"]["passed"]

    arrays = np.load(directory / "path_metrics.npz", allow_pickle=False)
    assert tuple(arrays["metrics"]) == METRICS
    expected_shapes = {"values": (96, p["paths"], 3), "primary_contrasts": (8, p["paths"], 3),
                       "endogenous_contrasts": (4, p["paths"], 3), "oracle_controls": (8, p["paths"], 3)}
    for key, shape in expected_shapes.items():
        assert arrays[key].shape == shape and np.isfinite(arrays[key]).all(), key
    assert len(index) == 96 and not index.duplicated([*CASE, "rule"]).any()
    for i, row in index.iterrows():
        selected = summary[summary.source == "contiguous"]
        for key in (*CASE, "rule"):
            selected = selected[selected[key] == row[key]]
        assert len(selected) == 1
        np.testing.assert_allclose(arrays["values"][i].mean(axis=0), selected[list(METRICS)].iloc[0], atol=2e-13, rtol=0)

    assert len(primary) == 8 and grid(primary, ("distribution", "template", "span")) == set(itertools.product(p["distributions"], p["templates"], p["history_spans"]))
    assert set(primary.amplitude) == {p["primary"]["amplitude"]}
    predicted = []
    for i, row in primary.iterrows():
        matched = index[(index.distribution == row.distribution) & (index.template == row.template) & (index.amplitude == row.amplitude) & (index.span == row.span) & (index.rule == p["primary"]["rule"])]
        slow = matched[matched.frequency == 1].index.item()
        fast = matched[matched.frequency == 17].index.item()
        np.testing.assert_array_equal(arrays["primary_contrasts"][i], arrays["values"][fast]-arrays["values"][slow])
        selected = phases[(phases.distribution == row.distribution) & (phases.template == row.template) & (phases.amplitude == row.amplitude) & (phases.span == row.span)]
        z = distribution(row.distribution).ppf(1-p["alpha"]/2)
        coefficient = z*z*2*distribution(row.distribution).pdf(z)/p["alpha"]
        mse = selected.groupby("frequency").first_order_log_error.apply(lambda value: np.mean(value**2))
        predicted.append(coefficient*(mse[17]-mse[1]))
    np.testing.assert_allclose(primary.predicted_is_contrast, predicted, atol=1e-11, rtol=0)
    all_values = np.concatenate([arrays["primary_contrasts"], arrays["endogenous_contrasts"], arrays["oracle_controls"]])
    intervals = independent_bootstrap(all_values, p["primary"]["bootstrap_replicates"], p["seeds"]["bootstrap"], p["primary"]["intervals"])
    maximum_ci_error = 0.
    for frame, offset, is_primary in ((primary, 0, True), (boundary, 8, False), (controls, 12, False)):
        for i, row in frame.iterrows():
            for mi, metric in enumerate(METRICS):
                name = f"observed_{metric}_contrast" if is_primary else metric
                np.testing.assert_allclose(row[name], all_values[offset+i, :, mi].mean(), atol=2e-13, rtol=0)
                for level in p["primary"]["intervals"] if is_primary else [.95]:
                    for bound_index, bound in enumerate(("low", "high")):
                        column = f"{metric}_{bound}_{level}" if is_primary else f"{metric}_{bound}95"
                        discrepancy = abs(row[column]-intervals[level][bound_index, offset+i, mi])
                        maximum_ci_error = max(maximum_ci_error, discrepancy)
                        assert discrepancy < 2e-12, column
    for metric in METRICS:
        np.testing.assert_allclose(boundary[f"actual_{metric}"]-boundary[f"redraw_{metric}"], boundary[metric], atol=2e-13, rtol=0)

    oracle_checks = []
    for _, row in controls.iterrows():
        ranks = rank_weights(int(row.span), p["alpha"], row.rule)
        expected_coverage = sum(weight*k/(row.span+1) for k, weight in ranks)
        np.testing.assert_allclose(row.expected_coverage, expected_coverage, atol=1e-14, rtol=0)
        exact_risk = 0.
        for k, weight in ranks:
            beta_law = beta(k, row.span+1-k)
            exact_risk += weight*quad(lambda u: scalar_risk(distribution(row.distribution).isf((1-u)/2), row.distribution, p["alpha"])*beta_law.pdf(u), 0, 1, epsabs=1e-9, epsrel=1e-10, limit=200)[0]
        assert abs(exact_risk-row.expected_interval_score) <= p["numerics"]["maximum_phase_metric_difference"]
        coverage_ok = bool(row.coverage_low95 <= expected_coverage <= row.coverage_high95)
        risk_ok = bool(row.interval_score_low95 <= exact_risk <= row.interval_score_high95)
        oracle_checks.append({"distribution": row.distribution, "span": int(row.span), "rule": row.rule,
                              "coverage_inside_reported_mc_interval": coverage_ok, "risk_inside_reported_mc_interval": risk_ok,
                              "independent_expected_risk_error": abs(exact_risk-row.expected_interval_score)})
    assert all(item["coverage_inside_reported_mc_interval"] and item["risk_inside_reported_mc_interval"] for item in oracle_checks)

    gate = p["primary"]["neural_bridge_gate"]
    observed = arrays["primary_contrasts"][:, :, 2].mean(axis=1)
    predicted = np.asarray(predicted)
    indeterminate = np.abs(predicted) < gate["predicted_absolute_contrast_indeterminate_below"]
    correct = (~indeterminate) & (predicted*observed > 0)
    low, high = intervals[.99375][:, :8, 2]
    supported = correct & (((predicted > 0) & (low > 0)) | ((predicted < 0) & (high < 0)))
    error = np.abs(predicted-observed)/np.maximum(np.abs(observed), gate["relative_error_denominator_floor"])
    for column, values in (("indeterminate_prediction", indeterminate), ("correct_direction", correct), ("supported_direction", supported)):
        np.testing.assert_array_equal(primary[column], values)
    np.testing.assert_allclose(primary.relative_contrast_error, error, atol=2e-11, rtol=0)
    counts_distribution = {name: int(supported[np.asarray(primary.distribution == name)].sum()) for name in p["distributions"]}
    counts_span = {str(span): int(supported[np.asarray(primary.span == span)].sum()) for span in p["history_spans"]}
    passed = bool(correct.sum() >= gate["minimum_correct_directions"] and supported.sum() >= gate["minimum_correct_directions_with_simultaneous_interval_excluding_zero"] and min(counts_distribution.values()) >= gate["minimum_supported_directions_per_distribution"] and min(counts_span.values()) >= gate["minimum_supported_directions_per_span"] and np.median(error) <= gate["maximum_median_relative_contrast_error"])
    verdict = {"correct_directions": int(correct.sum()), "supported_directions": int(supported.sum()), "supported_by_distribution": counts_distribution, "supported_by_span": counts_span, "median_relative_contrast_error_all8": float(np.median(error)), "passed": passed}
    for key in verdict:
        if key == "median_relative_contrast_error_all8":
            np.testing.assert_allclose(verdict[key], manifest["neural_bridge_gate"][key], atol=2e-11, rtol=0)
        else:
            assert verdict[key] == manifest["neural_bridge_gate"][key]
    return {"passed": True, "freeze_commit": manifest["freeze_commit"], "source_and_freeze_hash_checks": source_checks,
            "artifact_hash_checks": artifact_checks, "grid_rows": {key: len(value) for key, value in frames.items()},
            "independent_bootstrap_max_interval_error": maximum_ci_error, "oracle_control_checks": oracle_checks,
            "independently_recomputed_gate": verdict,
            "scope": "Saved data, formula, bootstrap, and provenance audit; no regeneration or new scientific outcome.",
            "verification_script_sha256": sha(__file__), "input_manifest_sha256": sha(directory / "manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/finite_mechanism")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are refused.")
    args = parser.parse_args()
    result = audit(args.directory)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")
