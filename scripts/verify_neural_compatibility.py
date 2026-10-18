"""Independently verify saved neural compatibility artifacts, without data draws.

Does not import the evaluator, analyzer, numerical engine, or training driver.
Saved actual intervals are rebuilt by sorting issued historical scores; risk
uses independent SciPy tail formulas. Bootstrap randomness only resamples
saved arrays and uses multiplicity counts rather than the analyzer's indexing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"configs/neural_compatibility_protocol.json"
EVAL = ROOT/"outputs/neural_compatibility_evaluation"
ANALYSIS = ROOT/"outputs/neural_compatibility_analysis"
TRAIN = ROOT/"outputs/neural_compatibility_training"
OUT = ROOT/"outputs/neural_compatibility_verification.json"
PRIMARY = ("P", "A", "D", "D_first32", "D_last32")
MOMENTS = ("xi_mean", "xi_second_moment", "eta_mean", "eta_second_moment", "xiP_eta_mean", "xiP_mean", "xiP_second_moment")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def independent_distribution(family):
    return norm if family == "gaussian" else t(5, scale=np.sqrt(3/5))


def independent_metrics(half, sigma, family, alpha):
    """half is relative to true sigma; do not import the numerical engine."""
    half = np.asarray(half, dtype=np.float64)
    distribution = independent_distribution(family)
    if family == "gaussian":
        stop_loss = 2*(norm.pdf(half)-half*norm.sf(half))
    else:
        scale = np.sqrt(3/5)
        unit = half/scale
        first_tail = scale*(5+unit*unit)/4*t.pdf(unit, 5)
        stop_loss = 2*(first_tail-half*t.sf(unit, 5))
    risk = 2*half+(2/alpha)*stop_loss
    return np.stack((2*distribution.cdf(half)-1, 2*half, risk, 2*half*sigma, risk*sigma), axis=-1)


def rank_weights(n, alpha):
    target = (1-alpha)*(n+1)
    low, high = int(np.floor(target)), int(np.ceil(target))
    return [(low, 1.)] if low == high else [(low, high-target), (high, target-low)]


def independently_resample(values, repetitions, seed, redraw=False, fixed=None):
    """Use multinomial multiplicities induced by the exact frozen index stream.

    Match the prescribed random number sequence, but replace indexed averaging
    with count-matrix multiplication and separate scalar loss calculations.
    """
    rng = np.random.default_rng(seed)
    paths = values.shape[1]
    result = np.empty((repetitions, 3))
    for start in range(0, repetitions, 128):
        batch = min(128, repetitions-start)
        draws = rng.integers(paths, size=(batch, paths))
        counts = np.zeros((batch, paths), dtype=np.float64)
        np.add.at(counts, (np.repeat(np.arange(batch), paths), draws.ravel()), 1.)
        weights = counts/paths
        if redraw:
            redrawn = np.einsum("bp,cp->bc", weights, values, optimize=False)
            population = np.full(batch, np.mean(np.abs(fixed[:, 0]-fixed[:, 1])))
            finite = np.mean(np.abs(redrawn-fixed[None, :, 1]), axis=1)
        else:
            averaged = np.einsum("bp,cpm->bcm", weights, values, optimize=False)
            population = np.mean(np.abs(averaged[:, :, 0]-averaged[:, :, 1]), axis=1)
            finite = np.mean(np.abs(averaged[:, :, 2]-averaged[:, :, 1]), axis=1)
        result[start:start+batch] = np.column_stack((population, finite, population-finite))
    return result


def self_check():
    values = np.arange(3*7*5, dtype=np.float64).reshape(3, 7, 5)/17
    got = independently_resample(values, 9, 91872)
    rng = np.random.default_rng(91872)
    draws = rng.integers(7, size=(9, 7))
    direct = []
    for selected in draws:
        means = values[:, selected, :].mean(axis=1)
        p = np.abs(means[:, 0]-means[:, 1]).mean()
        d = np.abs(means[:, 2]-means[:, 1]).mean()
        direct.append([p, d, p-d])
    np.testing.assert_allclose(got, direct, rtol=0, atol=1e-14)
    return {"independent_multiplicity_bootstrap": True}


def run(check_only=False):
    started = time.time()
    p = load_json(PROTOCOL)
    e, a = p["evaluation"], p["analysis"]
    em, am, tm = [load_json(folder/"manifest.json") for folder in (EVAL, ANALYSIS, TRAIN)]
    assert all(m["status"] == "complete" for m in (em, am, tm)), "Wait for all three complete manifests"
    assert check_only or not OUT.exists(), "Refusing to overwrite verification results; use --check-only for a read-only audit"
    assert p["frozen"] is True
    assert em["protocol_sha256"] == am["protocol_sha256"] == tm["protocol_sha256"] == sha(PROTOCOL)
    assert em["source_sha256"] == sha(ROOT/"scripts/evaluate_neural_compatibility.py")
    assert am["script_sha256"] == em["analysis_source_sha256"] == sha(ROOT/"scripts/analyze_neural_compatibility.py")
    assert em["training_manifest_sha256"] == sha(TRAIN/"manifest.json")
    assert tm["driver_sha256"] == sha(ROOT/"scripts/train_neural_compatibility.py")
    assert tm["test_data_accessed"] is False and tm["completed_models"] == 12
    assert em["groups"] == 192 and em["models"] == 12
    assert em["self_check"]["passed"] is True
    # Verify the exact evaluator and analyzer bytes predate new test generation.
    freeze = em["freeze_commit"]
    for relative in ("configs/neural_compatibility_protocol.json", "scripts/evaluate_neural_compatibility.py", "scripts/analyze_neural_compatibility.py"):
        saved_source = subprocess.check_output(["git", "show", f"{freeze}:{relative}"], cwd=ROOT)
        assert hashlib.sha256(saved_source).hexdigest() == sha(ROOT/relative), relative
    verified_hashes = 0
    for mapping, base in [(em["training_artifact_sha256"], ROOT), (em["evaluation_dependencies"], ROOT),
                          (am["sources"], ROOT), (em["outputs"], EVAL), (am["outputs"], ANALYSIS)]:
        for name, digest in mapping.items():
            path = (base/name).resolve()
            assert path.is_relative_to(base), name
            assert sha(path) == digest, name
            verified_hashes += 1
    for mapping in (p["sources"], p["evaluation_sources"]):
        for name, digest in mapping.items():
            assert sha(ROOT/"scripts"/name) == digest, name
            verified_hashes += 1
    index = pd.read_csv(EVAL/"index.csv")
    keys = ["panel", "architecture", "seed", "world", "context", "window"]
    expected = {(panel["panel"], architecture, seed, world["name"], context, window)
                for panel in p["panels"] for architecture in panel["training_config"]["architectures"]
                for seed in panel["training_config"]["model_seeds"] for world in e["worlds"]
                for context in e["contexts"] for window in e["windows"]}
    assert len(index) == len(expected) == 192 and not index.duplicated(keys).any()
    assert set(index[keys].itertuples(index=False, name=None)) == expected
    assert len(set(index.file)) == 192
    metrics_names, methods = e["metrics"], e["methods"]
    times = np.asarray(e["times"])
    world_data, scale_cache = {}, {}
    for world in e["worlds"]:
        with np.load(EVAL/f"test_{world['name']}.npz", allow_pickle=False) as saved:
            world_data[world["name"]] = {k:saved[k] for k in saved.files}
        data = world_data[world["name"]]
        np.testing.assert_array_equal(data["observations"], data["true_scale"]*data["innovations"])
        expected_sigma = np.ones((e["paths"], e["length"]))
        expected_sigma[:, e["onset"]:e["onset"]+e["duration"]] = world["amplitude"]
        np.testing.assert_array_equal(data["true_scale"], expected_sigma)
    np.testing.assert_array_equal(world_data["gaussian_flat"]["innovations"], world_data["gaussian_up2"]["innovations"])
    np.testing.assert_array_equal(world_data["gaussian_flat"]["innovations"], world_data["gaussian_down_half"]["innovations"])
    maxima = dict(actual_risk_replay=0., population_risk_replay=0., population_cdf=0., log_moments=0.,
                  decomposition=0., raw_units=0., redraw_aggregation=0., first_order=0., second_order=0.)
    primary_groups, primary_arrays, replica_arrays = [], [], []
    independent_cells, independent_moments, independent_approximation = [], [], []
    for row in index.itertuples(index=False):
        metadata = {key:getattr(row, key) for key in keys}
        world = next(w for w in e["worlds"] if w["name"] == row.world)
        family = world["distribution"]
        distribution = independent_distribution(family)
        z = float(distribution.ppf(1-e["alpha"]/2))
        coefficient = z*z*(2*distribution.pdf(z))/e["alpha"]
        slope = -z if family == "gaussian" else -6*z/(3+z*z)
        second_coefficient = -.5*(1+z*slope)
        data = world_data[row.world]
        if row.scale_file not in scale_cache:
            with np.load(ROOT/row.scale_file, allow_pickle=False) as saved:
                scale_cache[row.scale_file] = saved["scales"].astype(np.float64)
            assert sha(ROOT/row.scale_file) == row.scale_sha256
        scales = scale_cache[row.scale_file]
        y, sigma = data["observations"], data["true_scale"]
        current_sigma = sigma[:, times]
        delta = np.log(scales/sigma)
        current_delta = delta[:, times]
        ages = times[:, None]-np.arange(row.window, 0, -1)
        archive = delta[:, ages]
        file = EVAL/row.file
        assert sha(file) == row.sha256
        with np.load(file, allow_pickle=False) as saved:
            values = {k:saved[k] for k in saved.files}
        metrics, moments, replicas = values["metrics"], values["moments"], values["D_replicas_phase"]
        assert metrics.shape == (7, 64, 16, 5) and moments.shape == (3, 64, 16, 7) and replicas.shape == (64, 4, 5)
        assert values["methods"].tolist() == methods and values["metric_names"].tolist() == metrics_names
        assert values["moment_methods"].tolist() == ["P", "A", "D"] and values["moment_names"].tolist() == list(MOMENTS)
        assert values["phase_names"].tolist() == list(e["phases"]) and values["times"].tolist() == e["times"]
        assert np.isfinite(metrics).all() and np.isfinite(moments).all() and np.isfinite(replicas).all()
        np.testing.assert_allclose(values["delta"], current_delta, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(values["true_scale"], current_sigma)
        mu, variance = archive.mean(axis=-1), archive.var(axis=-1)
        np.testing.assert_allclose(values["archive_delta_mean"], mu, rtol=0, atol=1e-13)
        np.testing.assert_allclose(values["archive_delta_variance"], variance, rtol=0, atol=1e-13)
        maxima["first_order"] = max(maxima["first_order"], float(np.max(np.abs(values["first_order_log_error"]-(current_delta-mu)))))
        maxima["second_order"] = max(maxima["second_order"], float(np.max(np.abs(values["second_order_log_error"]-(current_delta-mu+second_coefficient*variance)))))
        np.testing.assert_allclose(values["first_order_log_error"], current_delta-mu, rtol=0, atol=1e-13)
        np.testing.assert_allclose(values["second_order_log_error"], current_delta-mu+second_coefficient*variance, rtol=0, atol=1e-12)
        np.testing.assert_allclose(values["risk_quadratic_coefficient"], coefficient, rtol=0, atol=1e-12)
        expected_nll = np.log(scales[:, times])+.5*(current_sigma/scales[:, times])**2+.5*np.log(2*np.pi)
        np.testing.assert_allclose(values["expected_gaussian_nll"], expected_nll, rtol=0, atol=1e-12)
        np.testing.assert_allclose(values["lag1_delta"], delta[:, times-1], rtol=0, atol=1e-13)
        oracle = independent_metrics(np.full_like(current_delta, z), current_sigma, family, e["alpha"])
        native = independent_metrics(z*scales[:, times]/current_sigma, current_sigma, family, e["alpha"])
        np.testing.assert_allclose(metrics[methods.index("oracle")], oracle, rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(metrics[methods.index("native")], native, rtol=1e-11, atol=1e-10)
        xi_p = moments[0, ..., 0]
        qp = z*np.exp(xi_p-current_delta)
        population_cdf = (2*distribution.cdf(qp[..., None]*np.exp(archive))-1).mean(axis=-1)
        root_error = float(np.max(np.abs(population_cdf-(1-e["alpha"]))))
        assert root_error < e["root_cdf_tolerance"]
        maxima["population_cdf"] = max(maxima["population_cdf"], root_error)
        population = independent_metrics(z*np.exp(xi_p), current_sigma, family, e["alpha"])
        maxima["population_risk_replay"] = max(maxima["population_risk_replay"], float(np.max(np.abs(population-metrics[methods.index("P")]))))
        np.testing.assert_allclose(metrics[methods.index("P")], population, rtol=1e-11, atol=1e-10)
        actual = np.zeros_like(population); actual_moments = np.zeros_like(moments[1])
        # Independent full sorting of every archive; original evaluator uses partition.
        sorted_scores = np.sort(np.abs(y[:, ages])/scales[:, ages], axis=-1)
        for rank, probability in rank_weights(row.window, e["alpha"]):
            q = sorted_scores[..., rank-1]
            relative_half = (scales[:, times]/current_sigma)*q
            actual += probability*independent_metrics(relative_half, current_sigma, family, e["alpha"])
            eta = np.log(q/qp)
            xi = np.log(relative_half/z)
            actual_moments += probability*np.stack((xi, xi*xi, eta, eta*eta, xi_p*eta, xi_p, xi_p*xi_p), axis=-1)
        maxima["actual_risk_replay"] = max(maxima["actual_risk_replay"], float(np.max(np.abs(actual-metrics[methods.index("A")]))))
        np.testing.assert_allclose(metrics[methods.index("A")], actual, rtol=1e-11, atol=1e-10)
        maxima["log_moments"] = max(maxima["log_moments"], float(np.max(np.abs(actual_moments-moments[1]))))
        np.testing.assert_allclose(moments[1], actual_moments, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(moments[0, ..., 1], xi_p*xi_p, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(moments[0, ..., 2:5], np.zeros_like(moments[0, ..., 2:5]))
        np.testing.assert_allclose(moments[..., 5], np.broadcast_to(xi_p, moments[..., 5].shape), rtol=0, atol=1e-12)
        np.testing.assert_allclose(moments[..., 6], np.broadcast_to(xi_p*xi_p, moments[..., 6].shape), rtol=0, atol=1e-12)
        assert np.min(moments[..., 1]-moments[..., 0]**2) >= -1e-10
        assert np.min(moments[..., 3]-moments[..., 2]**2) >= -1e-10
        identity = moments[..., 1]-moments[..., 6]-moments[..., 3]-2*moments[..., 4]
        maxima["decomposition"] = max(maxima["decomposition"], float(np.max(np.abs(identity))))
        assert np.max(np.abs(identity)) < 1e-9
        assert np.min(metrics[..., 2]-oracle[None, ..., 2]) >= -1e-9
        for raw, standardized in ((3, 1), (4, 2)):
            raw_error = float(np.max(np.abs(metrics[..., raw]-metrics[..., standardized]*current_sigma)))
            maxima["raw_units"] = max(maxima["raw_units"], raw_error)
            assert raw_error < 1e-9
        np.testing.assert_allclose(metrics[methods.index("D")], (metrics[methods.index("D_first32")]+metrics[methods.index("D_last32")])/2, rtol=0, atol=1e-10)
        for pi, (phase, selected) in enumerate(e["phases"].items()):
            phase_metrics = np.take(metrics, selected, axis=2).mean(axis=2)
            for mi, method in enumerate(methods):
                independent_cells.append({**metadata, "phase":phase, "method":method, **dict(zip(metrics_names, phase_metrics[mi].mean(axis=0)))})
            for method, subset in (("D", slice(None)), ("D_first32", slice(0,32)), ("D_last32", slice(32,64))):
                replicated = replicas[subset, pi].mean(axis=0)
                difference = float(np.max(np.abs(replicated-phase_metrics[methods.index(method)].mean(axis=0))))
                maxima["redraw_aggregation"] = max(maxima["redraw_aggregation"], difference)
                assert difference < 1e-9
            for mi, method in enumerate(("P", "A", "D")):
                mean = np.take(moments[mi], selected, axis=1).mean(axis=(0,1))
                independent_moments.append({**metadata, "phase":phase, "method":method, **dict(zip(MOMENTS, mean)),
                    "cross_covariance":mean[4]-mean[5]*mean[2], "cross_product_of_means":mean[5]*mean[2],
                    "identity_residual":mean[1]-mean[6]-mean[3]-2*mean[4]})
            independent_approximation.append({**metadata, "phase":phase,
                "first_order_log_mae":float(np.abs((current_delta-mu)[:, selected]-xi_p[:, selected]).mean()),
                "second_order_log_mae":float(np.abs((current_delta-mu+second_coefficient*variance)[:, selected]-xi_p[:, selected]).mean()),
                "mean_absolute_delta":float(np.abs(current_delta[:, selected]).mean()),
                **{f"{method}_quadratic_excess_risk":float(coefficient*moments[mi][:,selected,1].mean()) for mi,method in enumerate(("P","A","D"))},
                **{f"{method}_exact_excess_risk":float(phase_metrics[methods.index(method),:,2].mean()-oracle[...,2].mean()) for method in ("P","A","D")}})
            if row.world == a["primary_world"] and row.context == a["primary_context"] and row.window in a["primary_windows"] and phase in a["primary_phases"]:
                primary_groups.append({**metadata, "phase":phase})
                primary_arrays.append(np.stack([phase_metrics[methods.index(method), :, 2] for method in PRIMARY], axis=-1))
                replica_arrays.append(replicas[:, pi, 2])
    for filename, records in (("cell_metrics.csv", independent_cells), ("log_moments.csv", independent_moments), ("local_approximation.csv", independent_approximation)):
        recorded = pd.read_csv(ANALYSIS/filename)
        rebuilt = pd.DataFrame(records)
        assert list(recorded.columns) == list(rebuilt.columns) and len(recorded) == len(rebuilt)
        for column in recorded.columns:
            if recorded[column].dtype.kind in "fi":
                np.testing.assert_allclose(recorded[column], rebuilt[column], rtol=1e-9, atol=1e-9)
            else:
                assert recorded[column].tolist() == rebuilt[column].tolist(), column
    primary = np.stack(primary_arrays); replica = np.stack(replica_arrays)
    assert primary.shape == (48,64,5) and replica.shape == (48,64)
    groups = pd.DataFrame(primary_groups)
    pd.testing.assert_frame_equal(groups, pd.read_csv(ANALYSIS/"primary_groups.csv"), check_dtype=False)
    means = primary.mean(axis=1)
    pmae = float(np.abs(means[:,0]-means[:,1]).mean())
    dmae = float(np.abs(means[:,2]-means[:,1]).mean())
    gain = pmae-dmae
    path_draws = independently_resample(primary, a["bootstrap_replicates"], a["bootstrap_seed"])
    replica_draws = independently_resample(replica, a["bootstrap_replicates"], a["mc_bootstrap_seed"], redraw=True, fixed=means)
    with np.load(ANALYSIS/"primary_arrays.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["path_metrics"], primary)
        np.testing.assert_array_equal(saved["redraw_replica_risk"], replica)
        np.testing.assert_allclose(saved["path_bootstrap"], path_draws, rtol=0, atol=1e-12)
        np.testing.assert_allclose(saved["redraw_bootstrap"], replica_draws, rtol=0, atol=1e-12)
        bootstrap_max_error = float(np.max(np.abs(saved["path_bootstrap"]-path_draws)))
        replica_max_error = float(np.max(np.abs(saved["redraw_bootstrap"]-replica_draws)))
    path_ci = np.quantile(path_draws[:,2], [.025,.975])
    mc_se = float(np.std(replica_draws[:,2], ddof=1))
    first = pmae-float(np.abs(means[:,3]-means[:,1]).mean())
    last = pmae-float(np.abs(means[:,4]-means[:,1]).mean())
    fits = []
    for (panel, architecture, seed), rows in groups.groupby(["panel","architecture","seed"]):
        selected = means[rows.index]
        pp = float(np.abs(selected[:,0]-selected[:,1]).mean())
        dd = float(np.abs(selected[:,2]-selected[:,1]).mean())
        fits.append({"panel":int(panel), "architecture":architecture, "seed":int(seed), "population_mae":pp, "redraw_mae":dd, "gain":pp-dd})
    fit_frame = pd.DataFrame(fits)
    pd.testing.assert_frame_equal(fit_frame, pd.read_csv(ANALYSIS/"primary_fits.csv"), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    architecture_gains = fit_frame.groupby("architecture").gain.mean().to_dict()
    panel_gains = {str(k):float(v) for k,v in fit_frame.groupby("panel").gain.mean().items()}
    relative = gain/pmae if pmae > 1e-12 else None
    mc_fraction, half_fraction = mc_se/max(abs(gain),1e-12), abs(first-last)/max(abs(gain),1e-12)
    specification = a["refinement_gate"]
    gate = {"gain_positive":bool(gain>0), "path_interval_positive":bool(path_ci[0]>0),
        "relative_reduction_sufficient":bool(relative is not None and relative>=specification["minimum_relative_MAE_reduction"]),
        "both_architectures_positive":bool(sum(x>0 for x in architecture_gains.values())>=specification["positive_architectures"]),
        "panels_positive":bool(sum(x>0 for x in panel_gains.values())>=specification["minimum_positive_panels"]),
        "mc_se_small":bool(mc_fraction<=specification["maximum_mc_se_fraction_of_gain"]),
        "half_split_stable":bool(half_fraction<=specification["maximum_first32_last32_gain_difference_fraction"])}
    gate["passed"] = all(gate.values())
    expected_summary = {"population_mae":pmae, "redraw_mae":dmae, "gain":gain, "relative_mae_reduction":relative,
        "path95_gain":path_ci.tolist(), "redraw_mc_se_gain":mc_se, "redraw_mc95_gain":np.quantile(replica_draws[:,2],[.025,.975]).tolist(),
        "first32_gain":first, "last32_gain":last, "mc_se_fraction_of_gain":mc_fraction, "first_last_difference_fraction":half_fraction,
        "gain_by_architecture":architecture_gains, "gain_by_panel":panel_gains, "refinement_gate":gate}
    summary = load_json(ANALYSIS/"primary_summary.json")
    assert summary == am["primary_summary"]
    for key, value in expected_summary.items():
        if isinstance(value, dict):
            if key == "refinement_gate":
                assert summary[key] == value
            else:
                assert set(summary[key]) == set(value)
                for name in value:
                    np.testing.assert_allclose(summary[key][name], value[name], rtol=0, atol=1e-12)
        elif value is None:
            assert summary[key] is None
        else:
            np.testing.assert_allclose(summary[key], value, rtol=0, atol=1e-12)
    result = {"status":"complete", "passed":True, "verifier_source_sha256":sha(__file__),
        "protocol_sha256":sha(PROTOCOL), "evaluation_manifest_sha256":sha(EVAL/"manifest.json"),
        "analysis_manifest_sha256":sha(ANALYSIS/"manifest.json"), "verified_hashes":verified_hashes,
        "verified_groups":len(index), "actual_intervals_replayed":len(index)*e["paths"]*len(times),
        "primary_cells":48, "path_bootstrap_replicates":len(path_draws), "replica_bootstrap_replicates":len(replica_draws),
        "maximum_path_bootstrap_difference":bootstrap_max_error, "maximum_replica_bootstrap_difference":replica_max_error,
        "maximum_errors":maxima, "independent_primary_summary":expected_summary,
        "checks":self_check(), "new_observations_generated":False, "wall_seconds":time.time()-started,
        "limitations":"Saved D replicas audited through their means, splits, moments and phase aggregates; not redrawn or re-evaluated. No new training or test observations."}
    if not check_only:
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--check-only", action="store_true", help="Run all artifact checks and print results without writing or replacing the verification record")
    args = parser.parse_args()
    print(json.dumps(self_check(), indent=2)) if args.self_check else run(check_only=args.check_only)
