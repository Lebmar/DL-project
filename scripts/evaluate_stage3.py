"""Evaluate all frozen Stage 3 training repeats on the new replication paths.

This is a Stage 2-informed replication, not a newly selected method. No model
or parameter fitting occurs. Current context, issued-score archive context and
rolling window form the complete 2 x 2 x 2 factorial for every original fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

from evaluate_stage2 import (METRICS, ROOT, generate_base, metric_arrays, phases,
                            predict_scales, rolling_half_width)
from stage2_models import GaussianTCN, load_checkpoint, make_model
from train_stage2 import array_sha256, generate_paths

PROTOCOL = ROOT / "configs/stage3_replication_protocol.json"
TRAIN = ROOT / "outputs/stage3_training"
OUT = ROOT / "outputs/stage3_evaluation"
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
COMPONENTS = ("total", "current_at_archive64", "archive_at_current16",
              "current_at_archive16", "archive_at_current64", "third_order")
FAST_RTOL, FAST_ATOL = 1e-6, 2e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_source(path):
    relative = str(Path(path).resolve().relative_to(ROOT))
    content = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    assert hashlib.sha256(content).hexdigest() == sha(path), f"commit before new test access: {relative}"


def verified_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    training = json.loads((TRAIN / "manifest.json").read_text())
    assert protocol["frozen"] is True and training["status"] == "complete"
    assert training["completed_models"] == 30 and len(training["panels"]) == 3
    assert training["protocol_sha256"] == sha(PROTOCOL)
    assert training["test_data_accessed"] is False
    assert sha(ROOT / "scripts/train_stage3_repeats.py") == training["driver_sha256"]
    assert training["sources"] == protocol["sources"]
    for name, value in protocol["sources"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    panels, artifact_hashes = {}, {}
    for specification in protocol["panels"]:
        panel = specification["panel"]
        record = next(p for p in training["panels"] if p["panel"] == panel)
        panel_path = ROOT / record["manifest"]
        assert sha(panel_path) == record["manifest_sha256"]
        fitted = json.loads(panel_path.read_text())
        assert fitted["status"] == "complete" and fitted["completed_models"] == 10
        assert fitted["matching_minibatches_verified"] and len(fitted["models"]) == 10
        assert fitted["training_config"] == specification["training_config"]
        panel_protocol = panel_path.parent / "input_protocol.json"
        frozen_copy = panel_path.parent / "frozen_protocol.json"
        assert fitted["protocol_sha256"] == sha(panel_protocol) == sha(frozen_copy)
        frozen = json.loads(panel_protocol.read_text())
        assert frozen["frozen"] and frozen["stage3_panel"] == panel
        assert frozen["parent_protocol_sha256"] == sha(PROTOCOL)
        assert frozen["training_config"] == specification["training_config"]
        for name, value in fitted["source_sha256"].items():
            assert value == protocol["sources"][name] == sha(ROOT / "scripts" / name)
        tc = fitted["training_config"]
        assert {(m["architecture"], m["model_seed"]) for m in fitted["models"]} == {
            (a, s) for a in tc["architectures"] for s in tc["model_seeds"]}
        for model in fitted["models"]:
            assert sha(ROOT / model["checkpoint"]) == model["checkpoint_sha256"]
            assert model["causal_audit_selected_checkpoint"]["passed"]
        panels[panel] = fitted
        artifact_hashes[panel] = {str(p.relative_to(ROOT)): sha(p)
                                 for p in panel_path.parent.rglob("*") if p.is_file()}
    assert set(panels) == {1, 2, 3}
    return protocol, training, panels, artifact_hashes


def issued_scales(model, y, context, start=0, stop=None):
    """Each scale excludes y[t]; exact receptive field permits TCN64 batching.

    A TCN with receptive field 64 gives the same context64 output when all
    endpoints share one longer forward pass. First targets still get the same
    raw zero padding. Context16 and all GRUs use the original reset-window code.
    """
    y = np.asarray(y, dtype=np.float64)
    stop = y.shape[1] if stop is None else stop
    if (not isinstance(model, GaussianTCN) or context != model.receptive_field
            or not getattr(model, "_stage3_fast_context64_verified", True)):
        return predict_scales(model, y, context, start, stop)
    padded = np.pad(y, ((0, 0), (context, 0)))
    result = np.empty((len(y), stop-start), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for p in range(0, len(y), 32):
            x = torch.tensor(padded[p:p+32, start:stop+context-1], dtype=torch.float32)
            result[p:p+32] = model.forward_sequence(x)[:, context-1:].exp().numpy()
    return result


def past_quantiles(y, archive_scale, window, alpha, warmup):
    """Order statistic uses issued scores before t, with b-specific warmup."""
    scores = np.abs(y)/archive_scale
    warm = np.broadcast_to(np.asarray(warmup)[None, :], (len(y), len(warmup)))
    assert warm.shape[1] >= window
    history = np.concatenate((warm, scores), axis=1)
    views = np.lib.stride_tricks.sliding_window_view(history, window, axis=1)
    views = views[:, len(warmup)-window:len(warmup)-window+y.shape[1]]
    rank = min(window, int(np.ceil((window+1)*(1-alpha))))-1
    return np.partition(views, rank, axis=-1)[:, :, rank]


def decomposition(effect):
    d = lambda a, b: effect[(a, b, 128)]-effect[(a, b, 32)]
    total = d(64, 64)-d(16, 16)
    values = {"total": total,
              "current_at_archive64": d(64, 64)-d(16, 64),
              "archive_at_current16": d(16, 64)-d(16, 16),
              "current_at_archive16": d(64, 16)-d(16, 16),
              "archive_at_current64": d(64, 64)-d(64, 16)}
    values["third_order"] = values["current_at_archive64"]-values["current_at_archive16"]
    np.testing.assert_allclose(total, values[COMPONENTS[1]]+values[COMPONENTS[2]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(total, values[COMPONENTS[3]]+values[COMPONENTS[4]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(values["third_order"], values[COMPONENTS[4]]-values[COMPONENTS[2]], rtol=0, atol=1e-12)
    return values


def tiny_self_check():
    rng = np.random.default_rng(89931)
    y = rng.normal(size=(3, 320))
    sigma_a = np.exp(rng.normal(0, .3, y.shape))
    sigma_b = np.exp(rng.normal(0, .3, y.shape))
    warm = rng.uniform(.1, 2, 128)
    changed = y.copy()
    changed[:, 170:] += 50
    pulse = y.copy()
    pulse[:, 90:123] *= 3
    scalar_error = 0.
    for window in WINDOWS:
        q = past_quantiles(y, sigma_b, window, .1, warm)
        history = np.concatenate((np.broadcast_to(warm, (3, 128)), np.abs(y)/sigma_b), axis=1)
        rank = min(window, int(np.ceil((window+1)*.9)))-1
        expected = np.stack([np.sort(history[:, 128+t-window:128+t], axis=1)[:, rank]
                             for t in range(y.shape[1])], axis=1)
        np.testing.assert_array_equal(q, expected)
        np.testing.assert_array_equal(q[:, :171], past_quantiles(changed, sigma_b, window, .1, warm)[:, :171])
        diagonal = sigma_b*q
        np.testing.assert_array_equal(diagonal, rolling_half_width(y, sigma_b, window, .1, warm))
        scalar_error = max(scalar_error, float(np.max(np.abs(q-expected))))
        # q_b cancels from current-context x W in log width.
    values = {(a, b, w): rng.normal(size=(3, 6)) for a in CONTEXTS for b in CONTEXTS for w in WINDOWS}
    decomposition(values)
    torch.manual_seed(89931)
    fast_error = causal_error = 0.
    for architecture in ("gru", "tcn"):
        model = make_model(architecture)
        for context in CONTEXTS:
            result = issued_scales(model, y, context)
            reference = predict_scales(model, y, context)
            np.testing.assert_allclose(result, reference, rtol=FAST_RTOL, atol=FAST_ATOL)
            altered = issued_scales(model, changed, context)
            np.testing.assert_allclose(result[:, :171], altered[:, :171], rtol=FAST_RTOL, atol=FAST_ATOL)
            subset = issued_scales(model, y, context, 100, 220)
            np.testing.assert_allclose(result[:, 100:220], subset, rtol=FAST_RTOL, atol=FAST_ATOL)
            pulse_result = issued_scales(model, pulse, context)
            pulse_reference = predict_scales(model, pulse, context)
            np.testing.assert_allclose(pulse_result, pulse_reference, rtol=FAST_RTOL, atol=FAST_ATOL)
            np.testing.assert_allclose(result[:, :91], pulse_result[:, :91], rtol=FAST_RTOL, atol=FAST_ATOL)
            fast_error = max(fast_error, float(np.max(np.abs(result-reference))))
            fast_error = max(fast_error, float(np.max(np.abs(pulse_result-pulse_reference))))
            causal_error = max(causal_error, float(np.max(np.abs(result[:, :171]-altered[:, :171]))))
        if isinstance(model, GaussianTCN):
            model._stage3_fast_context64_verified = False
            np.testing.assert_array_equal(issued_scales(model, y, 64), predict_scales(model, y, 64))
    return {"passed": True, "scalar_past_quantile_error": scalar_error,
            "diagonal_half_width_bitwise_matches_original_quantile": True,
            "both_decomposition_paths_and_third_order_verified": True,
            "current_and_future_outcomes_excluded": True,
            "TCN_receptive_field_full_pass_matches_reset_context64": True,
            "all_toy_endpoints_including_initial_and_pulse_boundaries_checked": True,
            "disabled_fast_path_bitwise_matches_original_fallback": True,
            "maximum_fast_inference_scale_error": fast_error,
            "maximum_causal_prefix_scale_error": causal_error}


def save_manifest(manifest):
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.self_check:
        print(json.dumps(tiny_self_check()))
        return
    # All checks precede generation of any new replication test observation.
    committed_source(__file__)
    committed_source(PROTOCOL)
    protocol, training, panels, panel_hashes = verified_inputs()
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Existing Stage3 evaluation manifest; refusing silent partial or complete rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {"status": "evaluating", "design_status": protocol["purpose"],
                "claim_scope": "new Stage2-informed replication; no new selected method or population architecture claim",
                "protocol_sha256": sha(PROTOCOL), "source_sha256": sha(__file__),
                "code_commit_before_new_test_access": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "training_manifest_sha256": sha(TRAIN / "manifest.json"),
                "training_driver_sha256": training["driver_sha256"],
                "frozen_dependency_sha256": protocol["sources"], "panel_artifact_sha256": panel_hashes,
                "models": 30, "test": protocol["test"], "model_exclusions": "none",
                "new_test_generated": False,
                "self_check": tiny_self_check(), "validation_fast_inference_checks": [],
                "test_array_sha256": {}, "event_array_sha256": {},
                "cache": "one nochange forecast per family/fit/context; event targets [onset+1,exit+L) recomputed only",
                "inference": "TCN RF64 full causal pass only for context64; all other contexts reset separately; validation equivalence checked",
                "fast_inference_policy": {"relative_tolerance": FAST_RTOL, "absolute_tolerance": FAST_ATOL,
                    "validation_failure_action": "automatically use original reset-window predictor for that checkpoint before test access",
                    "mathematical_target": "same past64 observations and original raw-zero padding; numerical execution may differ below threshold"},
                "event_labels_and_true_scale_use": "generation, phase/oracle evaluation, and equivalent-context cache boundaries only",
                "component_definition": "J=D64,64-D16,16; C@b64+A@a16=C@b16+A@a64=J; third_order=C@b64-C@b16=A@a64-A@a16"}
    save_manifest(manifest)
    config = protocol["test"]
    assert config["contexts"] == list(CONTEXTS) and config["rolling_windows"] == list(WINDOWS)
    assert protocol["evaluation"]["metrics"] == METRICS
    onset, length, alpha = config["onset"], config["length"], config["alpha"]
    native_q = norm.ppf(1-alpha/2)
    models, warmups = {}, {}
    for panel, fitted in panels.items():
        tc = fitted["training_config"]
        train_y = generate_paths(tc["training_data"], tc["generator"])
        val_y = generate_paths(tc["validation_data"], tc["generator"])
        assert array_sha256(train_y) == fitted["data_sha256"]["train"]
        assert array_sha256(val_y) == fitted["data_sha256"]["validation"]
        with np.load(TRAIN / f"panel{panel}/validation_target_pairs.npz") as pairs:
            assert array_sha256(pairs["paths"]) == fitted["data_sha256"]["validation_pair_paths"]
            assert array_sha256(pairs["targets"]) == fitted["data_sha256"]["validation_pair_targets"]
        for entry in fitted["models"]:
            key = (panel, entry["architecture"], entry["model_seed"])
            model, payload = load_checkpoint(ROOT / entry["checkpoint"])
            assert payload["protocol_sha256"] == fitted["protocol_sha256"]
            assert payload["architecture"] == key[1] and payload["model_seed"] == key[2]
            assert payload["model_config"] == tc["model"] and payload["contexts"] == tc["contexts"]
            models[key] = model
            if isinstance(model, GaussianTCN):
                # Validation only: test is not generated until all 15 checks pass.
                optimized = issued_scales(model, val_y[:3, :512], 64)
                original = predict_scales(model, val_y[:3, :512], 64)
                error = float(np.max(np.abs(optimized-original)))
                accepted = bool(np.allclose(optimized, original, rtol=FAST_RTOL, atol=FAST_ATOL))
                model._stage3_fast_context64_verified = accepted
                manifest["validation_fast_inference_checks"].append({"panel": panel,
                    "architecture": key[1], "seed": key[2], "context": 64,
                    "validation_paths": 3, "targets_per_path": 512, "maximum_absolute_scale_error": error,
                    "optimized_inference_used": accepted,
                    "action": "optimized full-sequence" if accepted else "original reset-window fallback"})
            for context in CONTEXTS:
                vs = issued_scales(model, val_y[:1], context, val_y.shape[1]-128)
                warmups[(*key, context)] = np.abs(val_y[0, -128:])/vs[0]
    save_manifest(manifest)
    performance, factorial_performance, interactions, components, negative, diagonal_checks = [], [], [], [], [], []
    maximum_logwidth_identity = 0.
    for family_index, family in enumerate(config["families"]):
        manifest["new_test_generated"] = True
        control_y, control_true = generate_base(family, config, family_index)
        manifest["test_array_sha256"][family] = {"observations": array_sha256(control_y),
                                               "true_scale": array_sha256(control_true)}
        for key, model in models.items():
            panel, architecture, seed = key
            control_scales = {c: issued_scales(model, control_y, c) for c in CONTEXTS}
            control_q = {(b, w): past_quantiles(control_y, control_scales[b], w, alpha, warmups[(*key, b)])
                         for b in CONTEXTS for w in WINDOWS}
            control_halves = {(a, b, w): control_scales[a]*control_q[(b, w)]
                              for a in CONTEXTS for b in CONTEXTS for w in WINDOWS}
            control_arrays = {cell: metric_arrays(control_y, h, control_true, family, config)
                              for cell, h in control_halves.items()}
            control_native = {c: metric_arrays(control_y, control_scales[c]*native_q, control_true, family, config)
                              for c in CONTEXTS}
            for duration in config["durations"]:
                exit_time = onset+duration
                for amplitude in config["amplitudes"]:
                    y, true = control_y.copy(), control_true.copy()
                    y[:, onset:exit_time] *= amplitude
                    true[:, onset:exit_time] *= amplitude
                    case_name = f"{family}_D{duration}_A{amplitude:g}"
                    manifest["event_array_sha256"][case_name] = {"observations": array_sha256(y), "true_scale": array_sha256(true)}
                    slices = phases(onset, duration, length)
                    assert list(slices) == config["phases"]
                    sigma = {}
                    for context in CONTEXTS:
                        sigma[context] = control_scales[context].copy()
                        sigma[context][:, onset+1:exit_time+context] = issued_scales(model, y, context, onset+1, exit_time+context)
                        for target in (onset, exit_time+context):
                            independent = predict_scales(model, y[:2], context, target, target+1)
                            np.testing.assert_allclose(sigma[context][:2, target:target+1], independent, rtol=1e-6, atol=2e-6)
                    event_q = {(b, w): past_quantiles(y, sigma[b], w, alpha, warmups[(*key, b)])
                               for b in CONTEXTS for w in WINDOWS}
                    common = {"panel": panel, "family": family, "duration": duration, "amplitude": amplitude,
                              "architecture": architecture, "seed": seed}
                    phase_effects = {phase: {} for phase in slices}
                    logwidth_effects = {phase: {} for phase in slices}
                    for a in CONTEXTS:
                        event_native = metric_arrays(y, sigma[a]*native_q, true, family, config)
                        for phase, (lo, hi) in slices.items():
                            for world, arrays in (("event", event_native), ("control", control_native[a])):
                                performance.append({**common, "context": a, "window": 0, "calibrator": "native_gaussian",
                                    "phase": phase, "world": world, "paths": len(y), "steps": hi-lo,
                                    **{m: float(arrays[m][:, lo:hi].mean()) for m in METRICS}})
                        for b in CONTEXTS:
                            for window in WINDOWS:
                                cell = (a, b, window)
                                half = sigma[a]*event_q[(b, window)]
                                control_half = control_halves[cell]
                                np.testing.assert_allclose(half[:, :onset+1], control_half[:, :onset+1], rtol=1e-6, atol=2e-6)
                                bound = exit_time+max(a, b+window)
                                error = float(np.max(np.abs(half[:, bound:]-control_half[:, bound:])))
                                assert error <= 2e-6
                                negative.append({**common, "current_context": a, "archive_context": b, "window": window,
                                                 "forgetting_bound": bound, "half_width_max_abs_error_after_bound": error})
                                if a == b:
                                    event_direct = rolling_half_width(y, sigma[a], window, alpha, warmups[(*key, a)])
                                    control_direct = rolling_half_width(control_y, control_scales[a], window, alpha, warmups[(*key, a)])
                                    event_error = float(np.max(np.abs(half-event_direct)))
                                    control_error = float(np.max(np.abs(control_half-control_direct)))
                                    assert max(event_error, control_error) <= 1e-10
                                event_metrics = metric_arrays(y, half, true, family, config)
                                if a == b:
                                    event_original = metric_arrays(y, event_direct, true, family, config)
                                    control_original = metric_arrays(control_y, control_direct, control_true, family, config)
                                    metric_error = max(float(np.max(np.abs(event_metrics[m]-event_original[m])))
                                                       for m in METRICS)
                                    metric_error = max(metric_error, max(float(np.max(np.abs(control_arrays[cell][m]-control_original[m])))
                                                                         for m in METRICS))
                                    assert metric_error <= 1e-10
                                    diagonal_checks.append({**common, "context": a, "window": window,
                                        "event_half_width_error": event_error, "control_half_width_error": control_error,
                                        "all_five_metric_max_abs_error": metric_error})
                                for phase, (lo, hi) in slices.items():
                                    phase_means = {}
                                    for world, arrays in (("event", event_metrics), ("control", control_arrays[cell])):
                                        pp = np.stack([arrays[m][:, lo:hi].mean(axis=1) for m in METRICS], axis=-1)
                                        phase_means[world] = pp
                                        row = {**common, "current_context": a, "archive_context": b, "window": window,
                                               "phase": phase, "world": world, "paths": len(y), "steps": hi-lo,
                                               **dict(zip(METRICS, pp.mean(axis=0)))}
                                        factorial_performance.append(row)
                                        if a == b:
                                            performance.append({**common, "context": a, "window": window, "calibrator": f"rolling{window}",
                                                "phase": phase, "world": world, "paths": len(y), "steps": hi-lo,
                                                **dict(zip(METRICS, pp.mean(axis=0)))})
                                    phase_effects[phase][cell] = phase_means["event"]-phase_means["control"]
                                    logwidth_effects[phase][cell] = np.log(2*half[:, lo:hi]).mean(axis=1)-np.log(2*control_half[:, lo:hi]).mean(axis=1)
                    for phase in slices:
                        parts = decomposition(phase_effects[phase])
                        direct_j = (phase_effects[phase][(64, 64, 128)]-phase_effects[phase][(16, 16, 128)]
                                    -phase_effects[phase][(64, 64, 32)]+phase_effects[phase][(16, 16, 32)])
                        np.testing.assert_allclose(parts["total"], direct_j, rtol=0, atol=1e-12)
                        log_parts = decomposition(logwidth_effects[phase])
                        for component in ("current_at_archive64", "current_at_archive16", "third_order"):
                            err = float(np.max(np.abs(log_parts[component])))
                            maximum_logwidth_identity = max(maximum_logwidth_identity, err)
                            assert err <= 1e-12
                        for path, values in enumerate(direct_j):
                            interactions.append({**common, "phase": phase, "path": path, **dict(zip(METRICS, values))})
                        for component, values in parts.items():
                            for path, row in enumerate(values):
                                components.append({**common, "phase": phase, "component": component, "path": path,
                                                   **dict(zip(METRICS, row))})
                    print(json.dumps({**common, "elapsed_seconds": round(time.time()-started, 2)}), flush=True)
        save_manifest(manifest)
    datasets = {"performance_by_seed.csv": pd.DataFrame(performance),
                "crossed_performance_by_seed.csv": pd.DataFrame(factorial_performance),
                "paired_interactions.csv.gz": pd.DataFrame(interactions),
                "paired_component_effects.csv.gz": pd.DataFrame(components),
                "negative_controls.csv": pd.DataFrame(negative),
                "diagonal_reproduction.csv": pd.DataFrame(diagonal_checks)}
    for name, data in datasets.items():
        data.to_csv(OUT / name, index=False, compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None)
    assert manifest["source_sha256"] == sha(__file__)
    _, _, _, final_hashes = verified_inputs()
    assert final_hashes == panel_hashes, "training artifacts changed during evaluation"
    assert manifest["training_manifest_sha256"] == sha(TRAIN / "manifest.json")
    assert len(models) == 30 and len(negative) == 960 and len(diagonal_checks) == 480
    assert len(interactions) == 61440 and len(components) == 368640
    assert len(performance) == 5760 and len(factorial_performance) == 7680
    manifest.update(status="complete", cases=4, models_evaluated=len(models),
                    elapsed_seconds=time.time()-started, rows={n: len(d) for n, d in datasets.items()},
                    diagonal_check_count=len(diagonal_checks),
                    maximum_diagonal_half_width_error=max(max(r["event_half_width_error"], r["control_half_width_error"]) for r in diagonal_checks),
                    maximum_diagonal_all_five_metric_error=max(r["all_five_metric_max_abs_error"] for r in diagonal_checks),
                    negative_control_check_count=len(negative),
                    maximum_crossed_half_width_error_after_bound=max(r["half_width_max_abs_error_after_bound"] for r in negative),
                    maximum_fixed_archive_current_window_logwidth_identity_error=maximum_logwidth_identity,
                    maximum_validation_fast_inference_scale_error=max(r["maximum_absolute_scale_error"] for r in manifest["validation_fast_inference_checks"]),
                    artifacts={n: sha(OUT / n) for n in datasets})
    save_manifest(manifest)
    print(json.dumps({"status": "complete", "models": len(models), "cases": 4,
                      "elapsed_seconds": manifest["elapsed_seconds"], "rows": manifest["rows"]}), flush=True)


if __name__ == "__main__":
    main()
