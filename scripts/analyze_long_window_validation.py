"""Analyze the frozen independent-pool long-window validation on Slurm.

No observations are generated here. The bootstrap unit is the independent pool;
absolute values are applied only after pooling. ``--self-check`` uses tiny toy
arrays and is the only execution mode permitted outside a Slurm job.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/long_window_validation_protocol.json"
METHODS = ("P", "A", "D", "X", "D_first32", "D_last32")
METRICS = ("risk", "half", "coverage", "tangent", "convex", "eta", "eta2",
           "cdf_deficit", "recent_deficit", "event_deficit")
STATISTICS = ("G_phase", "G_endpoint", "C", "cancellation_P", "cancellation_D")
CONTRASTS = (("A-P", 1, 0), ("D-P", 2, 0), ("X-P", 3, 0),
             ("A-D", 1, 2), ("A-X", 1, 3))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(directory, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts,
            f"Expected a relative artifact path: {relative}")
    resolved = (directory / path).resolve()
    require(resolved.is_relative_to(directory.resolve()),
            f"Artifact escapes input directory: {relative}")
    require(resolved.is_file(), f"Missing artifact: {relative}")
    return resolved


def committed_sources():
    """Require current analysis and frozen design bytes in the current commit."""
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {}
    for path in (Path(__file__).resolve(), PROTOCOL):
        relative = path.relative_to(ROOT).as_posix()
        saved = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                               check=False, capture_output=True)
        require(saved.returncode == 0 and saved.stdout == path.read_bytes(),
                f"Commit {relative} before production analysis")
        result[relative] = sha256(path)
    return revision, result


def estimands(means):
    """Input [..., fit, endpoint, method]; output [..., statistic]."""
    ep = means[..., 0] - means[..., 1]
    ed = means[..., 2] - means[..., 1]
    phase_p = np.abs(ep.mean(axis=-1)).mean(axis=-1)
    phase_d = np.abs(ed.mean(axis=-1)).mean(axis=-1)
    endpoint_p = np.abs(ep).mean(axis=(-2, -1))
    endpoint_d = np.abs(ed).mean(axis=(-2, -1))
    coupling = (means[..., 1] - means[..., 3]).mean(axis=(-2, -1))
    return np.stack((phase_p - phase_d, endpoint_p - endpoint_d, coupling,
                     endpoint_p - phase_p, endpoint_d - phase_d), axis=-1)


def counts_from_indices(indices, categories):
    counts = np.zeros(indices.shape[:-1] + (categories,), dtype=np.uint8)
    flat_counts = counts.reshape(-1, categories)
    flat_indices = indices.reshape(-1, indices.shape[-1])
    for row, selected in zip(flat_counts, flat_indices):
        row[:] = np.bincount(selected, minlength=categories)
    return counts


def pool_bootstrap(primary_risk, repetitions, seed):
    pools = len(primary_risk)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, pools, size=(repetitions, pools), dtype=np.int64)
    counts = counts_from_indices(indices, pools)
    pooled = np.einsum("bp,pftm->bftm", counts.astype(float) / pools,
                       primary_risk, optimize=True)
    return counts, estimands(pooled)


def replica_bootstrap(primary_risk, replicas, repetitions, seed):
    """Independent resampling in each pool, shared across its fits/endpoints."""
    pools, _, replica_count, _ = replicas.shape
    rng = np.random.default_rng(seed)
    counts = np.empty((repetitions, pools, replica_count), dtype=np.uint8)
    draws = np.empty((repetitions, 2), dtype=np.float64)
    fixed = primary_risk.mean(axis=0)
    # This is the same RNG stream as one C-order [repetitions,pool,replica]
    # integers call with dtype=int64, without retaining a large index array.
    for start in range(0, repetitions, 128):
        stop = min(start + 128, repetitions)
        selected = rng.integers(0, replica_count,
                               size=(stop - start, pools, replica_count),
                               dtype=np.int64)
        counts[start:stop] = counts_from_indices(selected, replica_count)
        d = np.einsum("bpr,pfrt->bpft",
                      counts[start:stop].astype(float) / replica_count,
                      replicas, optimize=True).mean(axis=1)
        means = np.broadcast_to(fixed, (stop - start,) + fixed.shape).copy()
        means[..., 2] = d
        draws[start:stop] = estimands(means)[..., :2]
    return counts, draws


def direction_pass(value, direction):
    return bool(value < 0 if direction == "negative" else value > 0)


def hypothesis_summary(point, draws, mc_draws, first, last, architecture_points,
                       primary_risk, protocol):
    inference = protocol["primary"]["inference"]
    precision = protocol["mc_precision"]
    quantiles = inference["quantiles"]
    intervals = np.quantile(draws[:, :3], quantiles, axis=0).T
    pool_c = (primary_risk[..., 1] - primary_risk[..., 3]).mean(axis=(1, 2))
    df = len(pool_c) - 1
    se_c = float(pool_c.std(ddof=1) / np.sqrt(len(pool_c)))
    critical = float(student_t.ppf(quantiles[1], df))
    t_interval = [float(pool_c.mean() - critical * se_c),
                  float(pool_c.mean() + critical * se_c)]
    mc_se = mc_draws.std(axis=0, ddof=1)
    result = {}
    for i, hypothesis in enumerate(protocol["primary"]["hypotheses"]):
        direction = hypothesis["predicted_direction"]
        interval_pass = direction_pass(
            intervals[i, 1] if direction == "negative" else intervals[i, 0], direction)
        architecture_estimates = {
            architecture: float(values[i])
            for architecture, values in architecture_points.items()}
        architecture_pass = all(direction_pass(value, direction)
                                for value in architecture_estimates.values())
        row = {"statistic": STATISTICS[i], "predicted_direction": direction,
               "estimate": float(point[i]), "bootstrap_interval": intervals[i].tolist(),
               "architecture_estimates": architecture_estimates,
               "interval_direction_pass": interval_pass,
               "architecture_direction_pass": architecture_pass}
        supporting = interval_pass and architecture_pass
        if i < 2:
            denominator = max(abs(float(point[i])), precision["denominator_floor"])
            difference = abs(float(first[i] - last[i]))
            se_fraction = float(mc_se[i] / denominator)
            difference_fraction = difference / denominator
            se_pass = se_fraction <= precision["maximum_se_fraction"]
            split_pass = (difference_fraction <=
                          precision["maximum_first32_last32_difference_fraction"])
            passed = se_pass and split_pass
            row["mc_precision"] = {
                "standard_error": float(mc_se[i]), "denominator": denominator,
                "se_fraction": se_fraction, "first32_estimate": float(first[i]),
                "last32_estimate": float(last[i]),
                "first32_last32_difference": difference,
                "difference_fraction": difference_fraction, "se_pass": se_pass,
                "split_pass": split_pass, "passed": passed}
            row["supported"] = bool(supporting and passed)
            row["status"] = ("precision-inconclusive" if not passed else
                             "supported" if supporting else "unsupported")
        else:
            t_pass = t_interval[1] < 0
            row["student_t"] = {
                "df": df, "confidence": inference["individual_two_sided_confidence"],
                "standard_error": se_c, "critical_value": critical,
                "interval": t_interval, "upper_negative": t_pass}
            row["supported"] = bool(supporting and t_pass)
            row["status"] = "supported" if row["supported"] else "unsupported"
        result[hypothesis["name"]] = row
    return result, pool_c


def load_inputs(input_dir, protocol):
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "complete", "Evaluation manifest must be complete")
    require(manifest.get("protocol_sha256") == sha256(PROTOCOL),
            "Evaluation protocol hash does not match the frozen design")
    outputs = manifest.get("outputs", {})
    require(isinstance(outputs, dict) and "index.csv" in outputs,
            "Manifest must hash index.csv and every output")
    for relative, expected in outputs.items():
        require(sha256(safe_file(input_dir, relative)) == expected,
                f"Evaluation artifact hash mismatch: {relative}")
    with (input_dir / "index.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    evaluation = protocol["evaluation"]
    require(len(records) == evaluation["neural_groups"], "Unexpected evaluation group count")
    for row in records:
        for field in ("pool", "panel", "seed", "context", "window"):
            row[field] = int(row[field])
    fits = sorted({(r["panel"], r["architecture"], r["seed"]) for r in records})
    require(len(fits) == protocol["models"]["count"], "Expected all twelve frozen fits")
    require({fit[0] for fit in fits} == {1, 2, 3}, "Expected three training panels")
    require({fit[1] for fit in fits} == {"gru", "tcn"}, "Expected GRU and TCN")
    for panel in (1, 2, 3):
        for architecture in ("gru", "tcn"):
            require(sum(p == panel and a == architecture for p, a, _ in fits) == 2,
                    "Each training panel must retain two fits per architecture")
    conditions = list(itertools.product(
        [world["name"] for world in evaluation["worlds"]],
        protocol["models"]["contexts"], evaluation["windows"]))
    pools = evaluation["independent_pools"]
    paths = evaluation["paths_per_pool"]
    times = np.asarray(evaluation["times"])
    phase_indices = list(evaluation["phases"].values())
    method_count, metric_count = len(METHODS), len(METRICS)
    endpoint = np.empty((pools, len(fits), len(conditions), method_count,
                         len(times), metric_count), dtype=np.float64)
    slope_means = np.empty((pools, len(fits), len(conditions), len(times)))
    primary = protocol["primary"]
    primary_condition = (primary["world"], primary["context"], primary["window"])
    primary_index = conditions.index(primary_condition)
    entry = evaluation["phases"][primary["phase"]]
    replica_count = evaluation["D_replicas"]
    primary_replicas = np.empty((pools, len(fits), replica_count, len(entry)))
    seen = set()
    maximum_identity_error = 0.0
    maximum_replica_error = 0.0
    maximum_pair_error = 0.0
    for row in records:
        fit = (row["panel"], row["architecture"], row["seed"])
        condition = (row["world"], row["context"], row["window"])
        key = (row["pool"], fit, condition)
        require(1 <= row["pool"] <= pools and condition in conditions,
                f"Unexpected group: {key}")
        require(key not in seen, f"Duplicate group: {key}")
        seen.add(key)
        require(outputs.get(row["file"]) == row["sha256"],
                f"Group checksum absent or inconsistent in manifest: {row['file']}")
        pi, fi, ci = row["pool"] - 1, fits.index(fit), conditions.index(condition)
        with np.load(safe_file(input_dir, row["file"]), allow_pickle=False) as saved:
            require(tuple(saved["methods"].tolist()) == METHODS, "Method order mismatch")
            require(tuple(saved["metric_names"].tolist()) == METRICS, "Metric order mismatch")
            np.testing.assert_array_equal(saved["times"], times)
            metrics = np.asarray(saved["metrics"], dtype=float)
            slope = np.asarray(saved["slope_P"], dtype=float)
            replica_endpoint = np.asarray(saved["D_replica_endpoint"], dtype=float)
            replica_phase = np.asarray(saved["D_replica_phase"], dtype=float)
            require(metrics.shape == (method_count, paths, len(times), metric_count),
                    f"Incorrect metrics shape in {row['file']}")
            require(slope.shape == (paths, len(times)), "Incorrect slope shape")
            require(replica_endpoint.shape == (replica_count, len(times), metric_count),
                    "Incorrect endpoint replica shape")
            require(replica_phase.shape == (replica_count, len(phase_indices), metric_count),
                    "Incorrect phase replica shape")
            require(all(np.isfinite(array).all() for array in
                        (metrics, slope, replica_endpoint, replica_phase)),
                    f"Nonfinite evaluation values in {row['file']}")
            expected_tangent = slope[None] * (metrics[..., 1] - metrics[0, ..., 1])
            residual = metrics[..., 0] - metrics[0, ..., 0] - metrics[..., 3] - metrics[..., 4]
            maximum_identity_error = max(maximum_identity_error,
                                         float(np.max(np.abs(residual))),
                                         float(np.max(np.abs(metrics[..., 3] - expected_tangent))))
            np.testing.assert_allclose(metrics[..., 3], expected_tangent, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-9)
            require(float(metrics[..., 4].min()) >= -1e-10, "Negative convex remainder")
            means = metrics.mean(axis=1)
            endpoint[pi, fi, ci] = means
            slope_means[pi, fi, ci] = slope.mean(axis=0)
            for method, selected in ((2, slice(None)), (4, slice(0, 32)),
                                     (5, slice(32, None))):
                average = replica_endpoint[selected].mean(axis=0)
                maximum_replica_error = max(maximum_replica_error,
                                            float(np.max(np.abs(average - means[method]))))
                np.testing.assert_allclose(average, means[method], rtol=1e-10, atol=1e-10)
            expected_phase = np.stack([replica_endpoint[:, ids].mean(axis=1)
                                       for ids in phase_indices], axis=1)
            np.testing.assert_allclose(replica_phase, expected_phase, rtol=1e-10, atol=1e-10)
            if ci == primary_index:
                primary_replicas[pi, fi] = replica_endpoint[:, entry, 0]
            needs_pairs = condition[0] in ("gaussian_flat", "gaussian_up2") and condition[2] == 128
            require(("pair_risk_phase" in saved.files) == needs_pairs,
                    "Unexpected presence/absence of ordered-pair phase matrices")
            if needs_pairs:
                pair = saved["pair_risk_phase"]
                require(pair.shape == (len(phase_indices), paths, paths) and np.isfinite(pair).all(),
                        "Incorrect ordered-pair phase matrices")
                diagonal = np.diagonal(pair, axis1=1, axis2=2)
                off_diagonal = (pair.sum(axis=2) - diagonal) / (paths - 1)
                for phase_id, ids in enumerate(phase_indices):
                    a = metrics[1][:, ids, 0].mean(axis=1)
                    x = metrics[3][:, ids, 0].mean(axis=1)
                    maximum_pair_error = max(maximum_pair_error,
                                             float(np.max(np.abs(diagonal[phase_id] - a))),
                                             float(np.max(np.abs(off_diagonal[phase_id] - x))))
                    np.testing.assert_allclose(diagonal[phase_id], a, rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(off_diagonal[phase_id], x, rtol=1e-10, atol=1e-10)
    expected_keys = set(itertools.product(range(1, pools + 1), fits, conditions))
    require(seen == expected_keys, "Incomplete pool/fit/condition Cartesian product")
    primary_risk = endpoint[:, :, primary_index, :, :, 0].transpose(0, 1, 3, 2)[:, :, entry]
    return {"endpoint": endpoint, "slope_means": slope_means, "fits": fits,
            "conditions": conditions, "primary_risk": primary_risk,
            "primary_replicas": primary_replicas, "manifest": manifest,
            "input_checks": {"groups": len(records), "maximum_identity_error": maximum_identity_error,
                             "maximum_replica_error": maximum_replica_error,
                             "maximum_pair_error": maximum_pair_error}}


def scopes(fits, pools):
    all_pools, all_fits = list(range(pools)), list(range(len(fits)))
    yield {"scope": "all", "scope_id": "all", "panel": "", "architecture": "", "seed": "", "pool": ""}, all_pools, all_fits
    for panel in sorted({fit[0] for fit in fits}):
        yield {"scope": "panel", "scope_id": str(panel), "panel": panel, "architecture": "", "seed": "", "pool": ""}, all_pools, [i for i, fit in enumerate(fits) if fit[0] == panel]
    for i, (panel, architecture, seed) in enumerate(fits):
        yield {"scope": "fit", "scope_id": str(i), "panel": panel, "architecture": architecture, "seed": seed, "pool": ""}, all_pools, [i]
    for pool in all_pools:
        yield {"scope": "pool", "scope_id": str(pool + 1), "panel": "", "architecture": "", "seed": "", "pool": pool + 1}, [pool], all_fits


def write_csv(path, rows):
    iterator = iter(rows)
    first = next(iterator)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def secondary_tables(data, protocol, output):
    endpoints, slopes = data["endpoint"], data["slope_means"]
    phases = protocol["evaluation"]["phases"]
    times = protocol["evaluation"]["times"]
    entry = phases["entry"]
    grid_rows, metric_rows, target_rows, decomposition_rows, event_rows = [], [], [], [], []
    for scope, pool_ids, fit_ids in scopes(data["fits"], len(endpoints)):
        # First pool paths (already averaged in endpoints), then selected pools.
        means = endpoints[pool_ids][:, fit_ids].mean(axis=0)
        slope_mean = slopes[pool_ids][:, fit_ids].mean(axis=0)
        for ci, (world, context, window) in enumerate(data["conditions"]):
            metadata = {**scope, "world": world, "context": context, "window": window}
            risks = means[:, ci, :, :, 0].transpose(0, 2, 1)
            for phase, ids in phases.items():
                point = estimands(risks[:, ids])
                require(abs(float(point[0] - point[1] + point[3] - point[4])) < 1e-10,
                        "Cancellation identity failed")
                # Take explicitly to retain fit/method/time/metric axis order.
                phase_means = np.take(means[:, ci], ids, axis=2).mean(axis=(0, 2))
                row = {**metadata, "phase": phase,
                       **dict(zip(STATISTICS, map(float, point))),
                       "P_minus_A_signed": float((risks[:, ids, 0] - risks[:, ids, 1]).mean()),
                       "D_minus_A_signed": float((risks[:, ids, 2] - risks[:, ids, 1]).mean())}
                for metric_id, metric in enumerate(METRICS):
                    row[f"D_minus_X_{metric}"] = float(phase_means[2, metric_id] - phase_means[3, metric_id])
                grid_rows.append(row)
                for mi, method in enumerate(METHODS):
                    metric_rows.append({**metadata, "phase": phase, "method": method,
                                        **dict(zip(METRICS, map(float, phase_means[mi])))})
                local_slope = float(slope_mean[:, ci, ids].mean())
                for name, left, right in CONTRASTS:
                    gap = phase_means[left] - phase_means[right]
                    product = local_slope * float(gap[1])
                    covariance = float(gap[3]) - product
                    require(abs(float(gap[0] - gap[3] - gap[4])) < 1e-9,
                            "Aggregated exact decomposition failed")
                    decomposition_rows.append({**metadata, "phase": phase, "contrast": name,
                        "risk_gap": float(gap[0]), "exact_tangent": float(gap[3]),
                        "convex_gap": float(gap[4]), "slope_mean": local_slope,
                        "halfwidth_gap": float(gap[1]), "mean_product": float(gap[3]),
                        "product_of_means": product, "covariance": covariance})
            targets = [(f"entry_endpoint_{j + 1}", [ti]) for j, ti in enumerate(entry)]
            targets += [("entry_post_onset3", entry[1:])]
            for target, ids in targets:
                point = estimands(risks[:, ids])
                target_rows.append({**metadata, "target": target,
                    "times": ";".join(str(times[i]) for i in ids),
                    **dict(zip(STATISTICS, map(float, point))),
                    "P_minus_A_signed": float((risks[:, ids, 0] - risks[:, ids, 1]).mean()),
                    "D_minus_A_signed": float((risks[:, ids, 2] - risks[:, ids, 1]).mean()),
                    "A_minus_X_signed": float((risks[:, ids, 1] - risks[:, ids, 3]).mean())})
            if world.startswith("gaussian_") and world != "gaussian_flat":
                flat_index = data["conditions"].index(("gaussian_flat", context, window))
                flat_risk = means[:, flat_index, :, :, 0].transpose(0, 2, 1)
                for phase, ids in phases.items():
                    event_c = float((risks[:, ids, 1] - risks[:, ids, 3]).mean())
                    flat_c = float((flat_risk[:, ids, 1] - flat_risk[:, ids, 3]).mean())
                    event_rows.append({**metadata, "phase": phase, "event_A_minus_X": event_c,
                                       "flat_A_minus_X": flat_c,
                                       "event_minus_flat_A_minus_X": event_c - flat_c})
    tables = {"phase_grid.csv": grid_rows, "phase_metrics.csv": metric_rows,
              "entry_targets.csv": target_rows, "decompositions.csv": decomposition_rows,
              "event_minus_flat.csv": event_rows}
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    return {filename: len(rows) for filename, rows in tables.items()}


def run(input_dir, output):
    require(bool(os.environ.get("SLURM_JOB_ID")),
            "Production analysis is HPC-only; submit it to Slurm. Use --self-check locally.")
    require(not output.exists(), f"Output already exists; refusing to overwrite: {output}")
    revision, sources = committed_sources()
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol.get("frozen") is True, "Validation protocol must be frozen")
    require(protocol["evaluation"]["independent_pools"] == 16 and
            protocol["evaluation"]["D_replicas"] == 64, "Unexpected frozen design")
    data = load_inputs(input_dir, protocol)
    inference = protocol["primary"]["inference"]
    precision = protocol["mc_precision"]
    primary_risk = data["primary_risk"]
    point = estimands(primary_risk.mean(axis=0))
    bootstrap_counts, bootstrap_draws = pool_bootstrap(
        primary_risk, inference["replicates"], inference["seed"])
    mc_counts, mc_draws = replica_bootstrap(
        primary_risk, data["primary_replicas"], precision["bootstrap_replicates"], precision["seed"])
    architectures = ("gru", "tcn")
    architecture_points = {
        architecture: estimands(primary_risk[:, [i for i, fit in enumerate(data["fits"])
                                                 if fit[1] == architecture]].mean(axis=0))
        for architecture in architectures}
    leave_one_out = np.stack([estimands(np.delete(primary_risk, i, axis=0).mean(axis=0))
                              for i in range(len(primary_risk))])
    splits = []
    for mi in (4, 5):
        split_means = primary_risk.mean(axis=0).copy()
        split_means[..., 2] = split_means[..., mi]
        splits.append(estimands(split_means))
    primary, pool_c = hypothesis_summary(point, bootstrap_draws, mc_draws,
                                         splits[0], splits[1], architecture_points,
                                         primary_risk, protocol)
    np.testing.assert_allclose(bootstrap_draws[:, 0] - bootstrap_draws[:, 1] +
                               bootstrap_draws[:, 3] - bootstrap_draws[:, 4], 0,
                               rtol=0, atol=1e-10)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "inference.npz",
        primary_risk=primary_risk, primary_D_replica_risk=data["primary_replicas"],
        bootstrap_counts=bootstrap_counts, bootstrap_draws=bootstrap_draws,
        mc_counts=mc_counts, mc_draws=mc_draws, point=point,
        architecture_points=np.stack([architecture_points[a] for a in architectures]),
        architecture_names=np.asarray(architectures), leave_one_pool_out=leave_one_out,
        first32_point=splits[0], last32_point=splits[1], pool_C=pool_c,
        statistic_names=np.asarray(STATISTICS), methods=np.asarray(METHODS),
        fit_panel=np.asarray([f[0] for f in data["fits"]]),
        fit_architecture=np.asarray([f[1] for f in data["fits"]]),
        fit_seed=np.asarray([f[2] for f in data["fits"]]),
        times=np.asarray(protocol["evaluation"]["times"])[protocol["evaluation"]["phases"]["entry"]])
    np.savez_compressed(output / "endpoints.npz", means=data["endpoint"],
        slope_P_mean=data["slope_means"], methods=np.asarray(METHODS),
        metric_names=np.asarray(METRICS),
        times=np.asarray(protocol["evaluation"]["times"]),
        fit_panel=np.asarray([f[0] for f in data["fits"]]),
        fit_architecture=np.asarray([f[1] for f in data["fits"]]),
        fit_seed=np.asarray([f[2] for f in data["fits"]]),
        condition_world=np.asarray([c[0] for c in data["conditions"]]),
        condition_context=np.asarray([c[1] for c in data["conditions"]]),
        condition_window=np.asarray([c[2] for c in data["conditions"]]))
    table_counts = secondary_tables(data, protocol, output)
    write_csv(output / "leave_one_pool_out.csv", (
        {"omitted_pool": i + 1, **dict(zip(STATISTICS, map(float, values)))}
        for i, values in enumerate(leave_one_out)))
    summary = {"status": "complete", "protocol_version": protocol["version"],
        "protocol_sha256": sha256(PROTOCOL), "primary": primary,
        "point": dict(zip(STATISTICS, map(float, point))),
        "bootstrap": {"unit": "whole independent pool", "replicates": inference["replicates"],
                      "seed": inference["seed"], "quantiles": inference["quantiles"],
                      "confidence": inference["individual_two_sided_confidence"],
                      "shared_across_fits_methods_conditions_endpoints": True,
                      "absolute_values_after_pool_averaging": True},
        "mc": {"replicates": precision["bootstrap_replicates"], "seed": precision["seed"],
               "independent_resampling_per_pool": True,
               "shared_across_fits_conditions_within_pool": True,
               "fixed_original_paths_P_A": True},
        "input_checks": data["input_checks"], "secondary_table_rows": table_counts,
        "scope": {"independent_pools": 16, "paths_per_pool": 64, "fixed_fits": 12,
                  "training_panels": 3, "new_training": False},
        "interpretation": {
            "aggregation_specific_reversal_supported": bool(
                primary["H1_phase_failure"]["supported"] and
                primary["H2_endpoint_improvement"]["supported"]),
            "pairing_gap_supported": primary["H3_same_pool_pairing"]["supported"],
            "qualification": inference["qualification"],
            "fit_uncertainty": inference["fit_uncertainty"],
            "secondary_results": "Descriptive; no additional confirmatory family claim.",
            "pairing": protocol["primary"]["hypotheses"][2]["interpretation"],
            "cancellation": "The cancellation identity is algebra, not empirical confirmation.",
            "precision": "Any failed H1/H2 MC gate is precision-inconclusive; do not append replicas.",
            "stop_and_decide": protocol["stop_and_decide"]}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    manifest = {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL), "git_commit": revision,
        "sources": sources, "input_directory": str(input_dir),
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "input_outputs": data["manifest"]["outputs"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "bootstrap_seed": inference["seed"], "bootstrap_repetitions": inference["replicates"],
        "mc_seed": precision["seed"], "mc_repetitions": precision["bootstrap_replicates"],
        "array_axes": {"endpoints.means": ["pool", "fit", "condition", "method", "time", "metric"],
                       "endpoints.slope_P_mean": ["pool", "fit", "condition", "time"],
                       "inference.primary_risk": ["pool", "fit", "entry_time", "method"],
                       "inference.primary_D_replica_risk": ["pool", "fit", "replica", "entry_time"],
                       "inference.bootstrap_counts": ["draw", "pool"],
                       "inference.mc_counts": ["draw", "pool", "replica"]},
        "outputs": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(output),
                      "hypotheses": {key: value["status"] for key, value in primary.items()}}))


def self_check():
    """Tiny deterministic arithmetic checks; never read or create study data."""
    risk = np.full((3, 2, 4, 6), 5.0)
    risk[..., 0] += np.asarray([-2.0, 2.0, -2.0, 2.0])
    risk[..., 2] += 0.6
    risk[..., 3] += 0.2
    risk[..., 4] = risk[..., 2] - 0.01
    risk[..., 5] = risk[..., 2] + 0.01
    np.testing.assert_allclose(estimands(risk.mean(axis=0)), [-0.6, 1.4, -0.2, 2.0, 0.0], atol=1e-14)
    risk[0, :, :, 0] += 0.8
    risk[1, :, :, 0] -= 0.8
    risk[0, :, :, 2] -= 0.25
    risk[2, :, :, 2] += 0.25
    point = estimands(risk.mean(axis=0))
    require(not np.allclose(point, estimands(risk).mean(axis=0)),
            "Toy must distinguish pooling before absolute values")
    counts, draws = pool_bootstrap(risk, 19, 123)
    np.testing.assert_array_equal(counts.sum(axis=1), 3)
    for count, draw in zip(counts, draws):
        repeated = np.repeat(risk, count, axis=0).mean(axis=0)
        ep, ed = repeated[..., 0] - repeated[..., 1], repeated[..., 2] - repeated[..., 1]
        independent = [np.abs(ep.mean(1)).mean() - np.abs(ed.mean(1)).mean(),
                       np.abs(ep).mean() - np.abs(ed).mean(),
                       (repeated[..., 1] - repeated[..., 3]).mean(),
                       np.abs(ep).mean() - np.abs(ep.mean(1)).mean(),
                       np.abs(ed).mean() - np.abs(ed.mean(1)).mean()]
        np.testing.assert_allclose(draw, independent, atol=1e-14)
    offsets = np.linspace(-0.03, 0.03, 6)
    replicas = risk[..., 2][:, :, None, :] + offsets[None, None, :, None]
    mc_counts, mc_draws = replica_bootstrap(risk, replicas, 131, 456)
    indices = np.random.default_rng(456).integers(0, 6, size=(131, 3, 6), dtype=np.int64)
    np.testing.assert_array_equal(mc_counts, counts_from_indices(indices, 6))
    np.testing.assert_array_equal(mc_counts.sum(axis=2), 6)
    require(np.any(mc_counts[:, 0] != mc_counts[:, 1]), "Pool replica draws must be independent")
    for bi in (0, 128, 130):
        means = risk.mean(axis=0).copy()
        means[..., 2] = np.stack([np.take(replicas[pi], indices[bi, pi], axis=1).mean(axis=1)
                                  for pi in range(3)]).mean(axis=0)
        np.testing.assert_allclose(mc_draws[bi], estimands(means)[:2], atol=1e-14)
    protocol = json.loads(PROTOCOL.read_text())
    first, last = point.copy(), point.copy()
    first[:2] += 0.001
    last[:2] -= 0.001
    architecture = {"gru": point, "tcn": point}
    result, _ = hypothesis_summary(point, np.tile(point, (19, 1)),
                                    np.tile(point[:2], (13, 1)), first, last,
                                    architecture, risk, protocol)
    require(all(row["supported"] for row in result.values()), "Expected toy directional support")
    failed, _ = hypothesis_summary(point, np.tile(point, (19, 1)),
                                    np.tile(point[:2], (13, 1)), first, last + 100,
                                    architecture, risk, protocol)
    require(failed["H1_phase_failure"]["status"] == "precision-inconclusive" and
            failed["H2_endpoint_improvement"]["status"] == "precision-inconclusive",
            "Precision gate must prevent support")
    require(failed["H3_same_pool_pairing"]["supported"], "H3 must not depend on D MC")
    wrong_arch = {"gru": point, "tcn": -point}
    rejected, _ = hypothesis_summary(point, np.tile(point, (19, 1)),
                                      np.tile(point[:2], (13, 1)), first, last,
                                      wrong_arch, risk, protocol)
    require(not any(row["supported"] for row in rejected.values()), "Architecture sign gate failed")
    np.testing.assert_allclose(draws[:, 0] - draws[:, 1] + draws[:, 3] - draws[:, 4], 0, atol=1e-14)
    # Exercise secondary reductions on two toy conditions and one four-time phase.
    import tempfile
    toy_means = np.zeros((3, 2, 2, 6, 4, 10))
    toy_means[..., 0] = risk.transpose(0, 1, 3, 2)[:, :, None]
    toy_means[..., 1] = toy_means[..., 0]
    toy_means[..., 3] = toy_means[..., 0] - toy_means[:, :, :, :1, :, 0]
    toy = {"endpoint": toy_means, "slope_means": np.ones((3, 2, 2, 4)),
           "fits": [(1, "gru", 1), (1, "tcn", 2)],
           "conditions": [("gaussian_flat", 64, 128), ("gaussian_up2", 64, 128)]}
    toy_protocol = {"evaluation": {"phases": {"entry": [0, 1, 2, 3]},
                                    "times": [0, 1, 2, 3]}}
    with tempfile.TemporaryDirectory(prefix="long-window-analysis-selfcheck-") as tmp:
        counts_by_file = secondary_tables(toy, toy_protocol, Path(tmp))
        require(counts_by_file["phase_grid.csv"] == 14, "Toy secondary scope count mismatch")
        with (Path(tmp) / "phase_grid.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            require(abs(float(row["G_phase"]) - float(row["G_endpoint"]) +
                        float(row["cancellation_P"]) - float(row["cancellation_D"])) < 1e-12,
                    "Toy secondary cancellation identity failed")
        with (Path(tmp) / "event_minus_flat.csv").open(newline="") as handle:
            require(all(abs(float(row["event_minus_flat_A_minus_X"])) < 1e-12
                        for row in csv.DictReader(handle)), "Matched toy event subtraction failed")
    print(json.dumps({"self_check": "passed", "toy_pools": 3, "toy_fits": 2,
                      "pool_draws": 19, "mc_draws": 131,
                      "study_data_read_or_generated": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/long_window_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/long_window_validation_analysis")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        run(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
