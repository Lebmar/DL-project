"""Frozen neural bridge: population, issued-history and independent-redraw banks.

No model is trained here. True scales are used only in offline diagnostics;
redrawn observations never enter a forecaster. All current targets are
integrated analytically and all archived scores strictly precede the target.
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
import torch

import finite_mechanism_math as fm
from evaluate_stage2 import predict_scales
from evaluate_stage3 import FAST_ATOL, FAST_RTOL, issued_scales
from stage2_models import GaussianTCN, load_checkpoint, make_model
from train_stage2 import array_sha256, generate_paths

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/neural_compatibility_protocol.json"
TRAIN = ROOT / "outputs/neural_compatibility_training"
OUT = ROOT / "outputs/neural_compatibility_evaluation"
METHODS = ("oracle", "native", "P", "A", "D", "D_first32", "D_last32")
METRICS = ("coverage", "width_standardized", "interval_score_standardized", "width_raw", "interval_score_raw")
MOMENT_OBJECTS = ("P", "A", "D")
MOMENTS = ("xi_mean", "xi_second_moment", "eta_mean", "eta_second_moment", "xiP_eta_mean", "xiP_mean", "xiP_second_moment")
PHASES = ("pre", "entry", "late", "recovery")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    expected = hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)).hexdigest()
    assert expected == sha(path), f"Commit protocol and sources before test access: {relative}"


def point_metrics(relative_halfwidth, sigma, distribution, alpha=.1):
    a = np.asarray(relative_halfwidth, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(sigma, dtype=np.float64), a.shape)
    risk = fm.expected_interval_score(a, distribution, alpha)
    return np.stack((fm.absolute_cdf(a, distribution), 2*a, risk, 2*a*scale, risk*scale), axis=-1)


def log_moments(q, qp, current_delta, distribution, alpha=.1):
    z = float(fm.absolute_ppf(1-alpha, distribution))
    xi_p = np.asarray(current_delta) + np.log(np.asarray(qp)/z)
    eta = np.log(np.asarray(q)/np.asarray(qp))
    xi = xi_p + eta
    xi_p = np.broadcast_to(xi_p, xi.shape)
    return np.stack((xi, xi*xi, eta, eta*eta, xi_p*eta, xi_p, xi_p*xi_p), axis=-1)


def bank_metrics(absolute_innovations, archive_delta, current_delta, qp, sigma, distribution, alpha=.1):
    """Score every adjacent-rank candidate before integrating the rank coin.

    Last axis is historical age. Leading axes may be [path,time] or
    [replica,path,time]; the supplied issued deltas and scales broadcast.
    """
    scores = np.asarray(absolute_innovations)*np.exp(-np.asarray(archive_delta))
    ranks = fm.rank_mixture(scores.shape[-1], alpha, "randomized")
    scores.partition([rank-1 for rank, _ in ranks], axis=-1)
    result, moments = None, None
    for rank, probability in ranks:
        q = scores[..., rank-1]
        scored = probability*point_metrics(np.exp(current_delta)*q, sigma, distribution, alpha)
        component = probability*log_moments(q, qp, current_delta, distribution, alpha)
        result = scored if result is None else result+scored
        moments = component if moments is None else moments+component
    return result, moments


def decomposition_error(moments):
    return float(np.max(np.abs(moments[..., 1] - moments[..., 6] - moments[..., 3] - 2*moments[..., 4])))


def phase_means(values):
    """[replica,path,16 endpoints,metric] -> [replica,4 phases,metric]."""
    assert values.shape[-2] == 16
    return values.reshape(values.shape[0], values.shape[1], 4, 4, values.shape[-1]).mean(axis=(1, 3))


def self_check():
    """Small independent toy checks; no study seeds or fitted checkpoints."""
    rng = np.random.default_rng(99173)
    y = rng.normal(size=(3, 180))
    delta = rng.uniform(-.2, .2, size=y.shape)
    times = np.array([129, 130, 150, 160])
    max_replay_error = max_identity_error = max_rescale_error = 0.
    for distribution in ("gaussian", "t5"):
        for window in (32, 128):
            ages = times[:, None]-np.arange(window, 0, -1)
            archive = delta[:, ages]
            current = delta[:, times]
            qp = fm.population_mixture_quantile(archive.reshape(-1, window), distribution=distribution).reshape(current.shape)
            values, moments = bank_metrics(np.abs(y[:, ages]), archive, current, qp, np.ones(current.shape), distribution)
            reference = np.zeros_like(values)
            for path in range(len(y)):
                for ti, tt in enumerate(times):
                    sorted_scores = np.sort(np.abs(y[path, tt-window:tt])*np.exp(-delta[path, tt-window:tt]))
                    for rank, weight in fm.rank_mixture(window, .1, "randomized"):
                        reference[path, ti] += weight*point_metrics(np.exp(delta[path, tt])*sorted_scores[rank-1], 1, distribution)
            np.testing.assert_allclose(values, reference, rtol=0, atol=1e-12)
            max_replay_error = max(max_replay_error, float(np.max(np.abs(values-reference))))
            max_identity_error = max(max_identity_error, decomposition_error(moments))
            shifted, _ = bank_metrics(np.abs(y[:, ages]), archive+.4, current+.4, qp*np.exp(-.4), np.ones(current.shape), distribution)
            np.testing.assert_allclose(values, shifted, rtol=0, atol=1e-12)
            max_rescale_error = max(max_rescale_error, float(np.max(np.abs(values-shifted))))
            changed = y.copy(); changed[:, 129:] = 1e5
            future, _ = bank_metrics(np.abs(changed[:, ages]), archive, current, qp, np.ones(current.shape), distribution)
            np.testing.assert_array_equal(values[:, 0], future[:, 0])
            replicas = np.abs(rng.normal(size=(5, 3, 180)))
            batched, batched_moments = bank_metrics(replicas[:, :, ages], archive, current, qp, np.ones(current.shape), distribution)
            independent = [bank_metrics(replica[:, ages], archive, current, qp, np.ones(current.shape), distribution) for replica in replicas]
            np.testing.assert_array_equal(batched, np.stack([value[0] for value in independent]))
            np.testing.assert_array_equal(batched_moments, np.stack([value[1] for value in independent]))
    assert max_identity_error < 1e-12
    torch.manual_seed(99173)
    maximum_fast_error = 0.
    for architecture in ("gru", "tcn"):
        model = make_model(architecture)
        for context in (16, 64):
            issued = issued_scales(model, y, context)
            reference = predict_scales(model, y, context)
            np.testing.assert_allclose(issued, reference, rtol=FAST_RTOL, atol=FAST_ATOL)
            changed = y.copy(); changed[:, 129:] = 1e3
            future = issued_scales(model, changed, context)
            np.testing.assert_allclose(issued[:, :130], future[:, :130], rtol=FAST_RTOL, atol=FAST_ATOL)
            maximum_fast_error = max(maximum_fast_error, float(np.max(np.abs(issued-reference))))
    test = rng.normal(size=(5, 3, 16, 5))
    np.testing.assert_allclose(phase_means(test), np.stack([test[:, :, i:i+4].mean(axis=(1, 2)) for i in range(0, 16, 4)], axis=1), atol=1e-15)
    return {"passed": True, "independent_scalar_rank_replay_error": max_replay_error,
            "log_second_moment_identity_error": max_identity_error, "uniform_scale_invariance_error": max_rescale_error,
            "strict_past_quantiles": True, "strict_past_issued_scales": True, "fast_vs_reset_error": maximum_fast_error,
            "phase_and_replica_axis_audit": True}


def verified_inputs(protocol):
    assert protocol["frozen"] is True
    for path in (PROTOCOL, Path(__file__).resolve(), ROOT/"scripts/analyze_neural_compatibility.py"):
        committed(path)
    for name, digest in protocol["sources"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    for name, digest in protocol["evaluation_dependencies"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in protocol["evaluation_sources"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    training = json.loads((TRAIN / "manifest.json").read_text())
    assert training["status"] == "complete" and training["completed_models"] == 12
    assert training["protocol_sha256"] == sha(PROTOCOL)
    assert training["test_data_accessed"] is False
    assert training["driver_sha256"] == sha(ROOT / "scripts/train_neural_compatibility.py")
    assert training["sources"] == protocol["sources"]
    models, checks, artifacts = [], [], {}
    for specification in protocol["panels"]:
        panel = specification["panel"]
        record = next(row for row in training["panels"] if row["panel"] == panel)
        manifest_path = ROOT / record["manifest"]
        assert sha(manifest_path) == record["manifest_sha256"]
        fitted = json.loads(manifest_path.read_text())
        assert fitted["status"] == "complete" and fitted["completed_models"] == 4
        assert fitted["matching_minibatches_verified"] and fitted["training_config"] == specification["training_config"]
        panel_protocol = manifest_path.parent / "input_protocol.json"
        frozen_protocol = manifest_path.parent / "frozen_protocol.json"
        assert fitted["protocol_sha256"] == sha(panel_protocol) == sha(frozen_protocol)
        panel_payload = json.loads(panel_protocol.read_text())
        assert panel_payload["frozen"] is True and panel_payload["neural_compatibility_panel"] == panel
        assert panel_payload["parent_protocol_sha256"] == sha(PROTOCOL)
        assert panel_payload["training_config"] == specification["training_config"]
        assert fitted["source_sha256"] == protocol["sources"]
        tc = fitted["training_config"]
        val_y = generate_paths(tc["validation_data"], tc["generator"])
        assert array_sha256(val_y) == fitted["data_sha256"]["validation"]
        assert {(m["architecture"], m["model_seed"]) for m in fitted["models"]} == {(a, s) for a in tc["architectures"] for s in tc["model_seeds"]}
        for entry in fitted["models"]:
            assert sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
            assert entry["causal_audit_selected_checkpoint"]["passed"]
            model, payload = load_checkpoint(ROOT / entry["checkpoint"])
            assert payload["protocol_sha256"] == fitted["protocol_sha256"]
            assert payload["architecture"] == entry["architecture"] and payload["model_seed"] == entry["model_seed"]
            assert payload["model_config"] == tc["model"] and payload["contexts"] == tc["contexts"]
            fit = {"panel": panel, "architecture": entry["architecture"], "seed": entry["model_seed"], "checkpoint": entry["checkpoint"], "checkpoint_sha256": entry["checkpoint_sha256"]}
            if isinstance(model, GaussianTCN):
                optimized = issued_scales(model, val_y[:3, :512], 64)
                original = predict_scales(model, val_y[:3, :512], 64)
                accepted = bool(np.allclose(optimized, original, rtol=FAST_RTOL, atol=FAST_ATOL))
                model._stage3_fast_context64_verified = accepted
                checks.append({**fit, "context": 64, "validation_paths": 3, "validation_length": 512,
                    "maximum_absolute_error": float(np.max(np.abs(optimized-original))), "optimized_inference_used": accepted,
                    "failure_action": "original reset-window fallback"})
            models.append((fit, model))
        for path in manifest_path.parent.rglob("*"):
            if path.is_file():
                artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
        if "artifact_sha256" in record:
            for name, digest in record["artifact_sha256"].items():
                candidate = ROOT/name
                if not candidate.is_file():
                    candidate = manifest_path.parent/name
                assert sha(candidate) == digest, name
    assert len(models) == 12
    return training, models, checks, artifacts


def generate_innovations(distribution, shape, seed):
    rng = np.random.default_rng(seed)
    if distribution == "gaussian":
        return rng.standard_normal(shape)
    assert distribution == "t5"
    return rng.standard_t(5, shape)*np.sqrt(3/5)


def run():
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())
    assert not OUT.exists(), "Refusing to overwrite evaluation outputs"
    training, models, validation_checks, training_artifacts = verified_inputs(protocol)
    check = self_check()
    config = protocol["evaluation"]
    paths, length, replicas = config["paths"], config["length"], config["redraw_replicas"]
    times = np.asarray(config["times"], dtype=np.int64)
    alpha = config["alpha"]
    assert paths == 64 and length == 640 and replicas == 64 and len(times) == 16
    assert list(config["contexts"]) == [16, 64] and list(config["windows"]) == [32, 128]
    assert times.min() >= max(config["windows"])+max(config["contexts"])
    assert tuple(config["methods"]) == METHODS and tuple(config["metrics"]) == METRICS
    assert config["rank_rule"] == "randomized"
    assert config["phases"] == {phase:list(range(4*i,4*i+4)) for i,phase in enumerate(PHASES)}
    OUT.mkdir(parents=True)
    (OUT/"groups").mkdir(); (OUT/"issued_scales").mkdir()
    manifest = {"status": "evaluating", "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(__file__),
        "freeze_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "training_manifest_sha256": sha(TRAIN/"manifest.json"), "training_artifact_sha256": training_artifacts,
        "analysis_source_sha256": sha(ROOT/"scripts/analyze_neural_compatibility.py"),
        "sources": protocol["sources"], "evaluation_dependencies": protocol["evaluation_dependencies"],
        "self_check": check, "validation_fast_inference_checks": validation_checks,
        "metric_names": METRICS, "method_names": METHODS, "moment_names": MOMENTS, "moment_objects": MOMENT_OBJECTS,
        "phase_names": PHASES, "evaluation_times": times.tolist(), "new_test_generated": False,
        "replica_uncertainty": "core D means and two32replica means retained per path; replica phase means retained, full replica-by-path arrays not retained",
        "scope": "new frozen fits; offline true-scale diagnostic, fixed endpoint means; no deployable diagnostic or neural-population claim",
        "innovation_array_sha256": {}, "redraw_array_sha256": {}, "world_array_sha256": {}, "completed_groups": 0}
    def save_manifest():
        (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    save_manifest()
    innovations, redraws = {}, {}
    for family in ("gaussian", "t5"):
        innovations[family] = generate_innovations(family, (paths, length), config["seeds"][family])
        redraws[family] = generate_innovations(family, (replicas, paths, length), config["seeds"]["redraw_"+family])
        manifest["innovation_array_sha256"][family] = array_sha256(innovations[family])
        manifest["redraw_array_sha256"][family] = array_sha256(redraws[family])
    worlds = {}
    for item in config["worlds"]:
        name, family = item["name"], item["distribution"]
        sigma = np.ones((paths, length))
        sigma[:, config["onset"]:config["onset"]+config["duration"]] = item["amplitude"]
        y = sigma*innovations[family]
        worlds[name] = (family, y, sigma)
        manifest["world_array_sha256"][name] = {"observations": array_sha256(y), "true_scale": array_sha256(sigma)}
        np.savez_compressed(OUT/f"test_{name}.npz", observations=y, true_scale=sigma, innovations=innovations[family])
    manifest["new_test_generated"] = True
    save_manifest()
    index, summaries = [], []
    maximum_decomposition = maximum_root_residual = maximum_scalar_replay = 0.
    for fit, model in models:
        for world, (family, y, sigma) in worlds.items():
            z = float(fm.absolute_ppf(1-alpha, family))
            coefficient = z*z*float(fm.absolute_pdf(z, family))/alpha
            second_coefficient = -.5*(1+z*float(fm.absolute_pdf_derivative(z, family))/float(fm.absolute_pdf(z, family)))
            for context in config["contexts"]:
                scales = issued_scales(model, y, context)
                assert scales.shape == y.shape and np.isfinite(scales).all() and np.all(scales > 0)
                key = f"panel{fit['panel']}_{fit['architecture']}_seed{fit['seed']}_{world}_c{context}"
                compact_scales = scales.astype(np.float32)
                np.testing.assert_array_equal(compact_scales.astype(np.float64), scales)
                scale_path = OUT/"issued_scales"/(key+".npz")
                np.savez_compressed(scale_path, scales=compact_scales)
                delta = np.log(scales/sigma)
                current = delta[:, times]
                current_sigma = sigma[:, times]
                native = point_metrics(z*np.exp(current), current_sigma, family, alpha)
                oracle = point_metrics(np.full(current.shape, z), current_sigma, family, alpha)
                for window in config["windows"]:
                    ages = times[:, None]-np.arange(window, 0, -1)
                    archive = delta[:, ages]
                    qp = fm.population_mixture_quantile(archive.reshape(-1, window), 1-alpha, family).reshape(current.shape)
                    root_error = float(np.max(np.abs(fm.mixture_cdf(qp.reshape(-1), archive.reshape(-1, window), family)-(1-alpha))))
                    assert root_error < config["root_cdf_tolerance"]
                    maximum_root_residual = max(maximum_root_residual, root_error)
                    population = point_metrics(np.exp(current)*qp, current_sigma, family, alpha)
                    population_moments = log_moments(qp, qp, current, family, alpha)
                    actual, actual_moments = bank_metrics(np.abs(innovations[family][:, ages]), archive, current, qp, current_sigma, family, alpha)
                    # Independent sorted replay of every actual interval.
                    sorted_actual = np.sort(np.abs(y[:, ages])/scales[:, ages], axis=-1)
                    replay = sum(prob*point_metrics(scales[:, times]/current_sigma*sorted_actual[...,rank-1], current_sigma, family, alpha)
                                 for rank, prob in fm.rank_mixture(window, alpha, "randomized"))
                    replay_error = float(np.max(np.abs(actual-replay)))
                    np.testing.assert_allclose(actual, replay, rtol=1e-12, atol=1e-11)
                    maximum_scalar_replay = max(maximum_scalar_replay, replay_error)
                    redraw_sum = np.zeros_like(actual); first_sum = np.zeros_like(actual); last_sum = np.zeros_like(actual)
                    redraw_moments = np.zeros_like(actual_moments)
                    replica_phase = np.empty((replicas, 4, len(METRICS)))
                    for start in range(0, replicas, 8):
                        stop = min(start+8, replicas)
                        values, moments = bank_metrics(np.abs(redraws[family][start:stop, :, ages]), archive, current, qp, current_sigma, family, alpha)
                        redraw_sum += values.sum(axis=0)
                        redraw_moments += moments.sum(axis=0)
                        if stop <= replicas//2:
                            first_sum += values.sum(axis=0)
                        else:
                            assert start >= replicas//2
                            last_sum += values.sum(axis=0)
                        replica_phase[start:stop] = phase_means(values)
                    redraw_mean = redraw_sum/replicas
                    first_mean, last_mean = first_sum/(replicas//2), last_sum/(replicas//2)
                    redraw_moments /= replicas
                    all_metrics = np.stack((oracle, native, population, actual, redraw_mean, first_mean, last_mean))
                    all_moments = np.stack((population_moments, actual_moments, redraw_moments))
                    identity_error = decomposition_error(all_moments)
                    assert identity_error < 1e-10
                    assert np.min(all_metrics[..., 2]-oracle[None, ..., 2]) >= -1e-10
                    maximum_decomposition = max(maximum_decomposition, identity_error)
                    mu, variance = archive.mean(axis=-1), archive.var(axis=-1)
                    group = {**fit, "world": world, "distribution": family, "context": context, "window": window,
                             "scale_file": scale_path.relative_to(ROOT).as_posix(), "scale_sha256": sha(scale_path)}
                    output = OUT/"groups"/(key+f"_w{window}.npz")
                    np.savez_compressed(output, metrics=all_metrics, methods=np.array(METHODS), metric_names=np.array(METRICS),
                        moments=all_moments, moment_methods=np.array(MOMENT_OBJECTS), moment_names=np.array(MOMENTS),
                        D_replicas_phase=replica_phase, phase_names=np.array(PHASES), times=times, true_scale=current_sigma,
                        delta=current, archive_delta_mean=mu, archive_delta_variance=variance,
                        first_order_log_error=current-mu, second_order_log_error=current-mu+second_coefficient*variance,
                        expected_gaussian_nll=np.log(scales[:, times])+.5*np.exp(-2*current)+.5*np.log(2*np.pi),
                        lag1_delta=delta[:, times-1], risk_quadratic_coefficient=np.array(coefficient))
                    group.update(file=output.relative_to(OUT).as_posix(), sha256=sha(output), root_cdf_residual=root_error,
                                 log_decomposition_error=identity_error, scalar_replay_error=replay_error)
                    index.append(group)
                    for method_index, method in enumerate(METHODS):
                        phased = all_metrics[method_index].reshape(paths, 4, 4, len(METRICS)).mean(axis=(0, 2))
                        for phase_index, phase in enumerate(PHASES):
                            summaries.append({**{k: group[k] for k in ("panel", "architecture", "seed", "world", "distribution", "context", "window")},
                                "method": method, "phase": phase, **dict(zip(METRICS, phased[phase_index]))})
                    manifest["completed_groups"] += 1
                save_manifest()
                print(json.dumps({"completed": key, "groups": manifest["completed_groups"]}), flush=True)
    assert len(index) == 192
    # Verify training and frozen inputs were not mutated by inference or output.
    for name, digest in training_artifacts.items():
        assert sha(ROOT/name) == digest, name
    for name, digest in protocol["sources"].items():
        assert sha(ROOT/"scripts"/name) == digest, name
    for name, digest in protocol["evaluation_dependencies"].items():
        assert sha(ROOT/name) == digest, name
    assert sha(PROTOCOL) == manifest["protocol_sha256"]
    assert sha(__file__) == manifest["source_sha256"]
    pd.DataFrame(index).to_csv(OUT/"index.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT/"summary.csv", index=False)
    manifest.update(status="complete", wall_seconds=time.time()-started, groups=len(index), models=len(models),
        maximum_population_cdf_residual=maximum_root_residual, maximum_log_decomposition_error=maximum_decomposition,
        maximum_actual_scalar_replay_error=maximum_scalar_replay,
        outputs={path.relative_to(OUT).as_posix():sha(path) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name!="manifest.json"})
    save_manifest()
    print(json.dumps({"status":"complete", "groups":len(index), "wall_seconds":manifest["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print(json.dumps(self_check(), indent=2)) if args.self_check else run()
