"""Exploratory crossed issued-score archive diagnostic of frozen Stage 2 fits.

The Stage 2 outcomes were viewed before this diagnostic was specified. There
is no new blind test, fitting, model selection or calibration parameter search.
Both contexts use each original checkpoint; event labels are evaluation only.
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

from evaluate_stage2 import (METRICS, ROOT, generate_base, metric_arrays, phases,
                            predict_scales, rolling_half_width, verified_inputs)
from stage2_models import load_checkpoint
from train_stage2 import array_sha256, generate_paths

OUT = ROOT / "outputs/stage2_archive_mechanism"
EVAL = ROOT / "outputs/stage2_evaluation"
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
COMPONENTS = ("total", "current_at_archive64", "archive_at_current16",
              "current_at_archive16", "archive_at_current64")
ALL_METRICS = (*METRICS, "log_width")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_original_inputs():
    protocol, training, controls = verified_inputs()
    evaluation_path = EVAL / "manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["status"] == "complete" and evaluation["cases"] == 30
    assert sha(ROOT / "configs/stage2_protocol.json") == evaluation["protocol_sha256"]
    assert sha(ROOT / "outputs/stage2_training/manifest.json") == evaluation["training_manifest_sha256"]
    assert sha(ROOT / "outputs/stage2_controls/models.json") == evaluation["controls_sha256"]
    for name, expected in evaluation["sources"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    for name, expected in evaluation["artifacts"].items():
        assert sha(EVAL / name) == expected, name
    return protocol, training, controls, evaluation


def crossed_half_width(y, current_scale, archive_scale, window, alpha, archive_warmup):
    """Current scale times past issued scores with the chosen archive scale."""
    archive_half = rolling_half_width(y, archive_scale, window, alpha, archive_warmup)
    return current_scale * (archive_half / archive_scale)


def decompose(effect):
    """Two exact paths from the diagonal interaction, without unique mediation."""
    d = lambda a, b: effect[(a, b, 128)] - effect[(a, b, 32)]
    total = d(64, 64)-d(16, 16)
    values = {"total": total,
              "current_at_archive64": d(64, 64)-d(16, 64),
              "archive_at_current16": d(16, 64)-d(16, 16),
              "current_at_archive16": d(64, 16)-d(16, 16),
              "archive_at_current64": d(64, 64)-d(64, 16)}
    np.testing.assert_allclose(total, values[COMPONENTS[1]]+values[COMPONENTS[2]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(total, values[COMPONENTS[3]]+values[COMPONENTS[4]], rtol=0, atol=1e-12)
    return values


def tiny_self_check():
    rng = np.random.default_rng(89031)
    y = rng.normal(size=(3, 300))
    current = np.exp(rng.normal(0, .3, y.shape))
    archive = np.exp(rng.normal(0, .3, y.shape))
    warm = rng.uniform(.1, 2, 128)
    changed = y.copy()
    changed[:, 170:] += 50
    errors = []
    for window in WINDOWS:
        actual = crossed_half_width(y, current, archive, window, .1, warm)
        history = np.concatenate((np.broadcast_to(warm, (3, 128)), np.abs(y)/archive), axis=1)
        rank = min(window, int(np.ceil((window+1)*.9)))-1
        expected = np.empty_like(y)
        for t in range(y.shape[1]):
            expected[:, t] = current[:, t]*np.sort(history[:, 128+t-window:128+t], axis=1)[:, rank]
        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)
        diagonal = crossed_half_width(y, archive, archive, window, .1, warm)
        native_rolling = rolling_half_width(y, archive, window, .1, warm)
        np.testing.assert_allclose(diagonal, native_rolling, rtol=1e-15, atol=1e-15)
        causal = crossed_half_width(changed, current, archive, window, .1, warm)
        np.testing.assert_array_equal(actual[:, :171], causal[:, :171])
        errors.append(float(np.max(np.abs(actual-expected))))
    toy = {(a, b, w): rng.normal(size=(3, 6)) for a in CONTEXTS for b in CONTEXTS for w in WINDOWS}
    decompose(toy)
    # Q_b,W cancels from current-context x window log-width interaction.
    logs = {(a, b, w): np.log(2*crossed_half_width(y, s, archive, w, .1, warm))
            for a, s in ((16, current), (64, current*1.3)) for b in (16,) for w in WINDOWS}
    identity = logs[(64, 16, 128)]-logs[(16, 16, 128)]-logs[(64, 16, 32)]+logs[(16, 16, 32)]
    np.testing.assert_allclose(identity, 0, rtol=0, atol=1e-14)
    return {"passed": True, "cross_quantile_matches_scalar_past_score_definition": True,
            "diagonal_matches_original_rolling_definition": True,
            "current_and_future_outcomes_excluded_from_archive_quantile": True,
            "both_decomposition_paths_are_algebraically_exact": True,
            "fixed_archive_logwidth_current_by_window_zero_is_identity": True,
            "maximum_scalar_quantile_error": max(errors)}


def bootstrap_weights(paths, repetitions=4000, seed=610023):
    rng = np.random.default_rng(seed)
    sampled = rng.integers(paths, size=(repetitions, paths))
    weights = np.zeros((repetitions, paths))
    np.add.at(weights, (np.arange(repetitions)[:, None], sampled), 1/paths)
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-14)
    return weights


def summarize(effects, seeds, paths):
    weights = bootstrap_weights(paths)
    rows, seed_rows = [], []
    keys = ["family", "architecture", "phase", "component"]
    indexed = {}
    expected = pd.MultiIndex.from_product([seeds, range(paths)], names=["seed", "path"])
    for key, group in effects.groupby(keys, sort=True):
        assert not group.duplicated(["seed", "path"]).any()
        g = group.set_index(["seed", "path"])
        assert set(g.index) == set(expected)
        indexed[key] = g.reindex(expected)[list(ALL_METRICS)].to_numpy().reshape(len(seeds), paths, len(ALL_METRICS))
    # Preserve seed/path pairing across architectures for a descriptive contrast.
    for family in effects.family.unique():
        for phase in effects.phase.unique():
            for component in COMPONENTS:
                indexed[(family, "gru-minus-tcn", phase, component)] = (
                    indexed[(family, "gru", phase, component)]-indexed[(family, "tcn", phase, component)])
    for key, values in sorted(indexed.items()):
        estimates = values.mean(axis=(0, 1))
        bounds = np.quantile(weights @ values.mean(axis=0), [.025, .975], axis=0)
        by_seed = values.mean(axis=1)
        seed_samples = (weights @ values.transpose(1, 0, 2).reshape(paths, -1)).reshape(
            len(weights), len(seeds), len(ALL_METRICS))
        seed_bounds = np.quantile(seed_samples, [.025, .975], axis=0)
        for m, metric in enumerate(ALL_METRICS):
            rows.append({**dict(zip(keys, key)), "metric": metric, "estimate": estimates[m],
                         "ci_low": bounds[0, m], "ci_high": bounds[1, m],
                         "seed_min": by_seed[:, m].min(), "seed_max": by_seed[:, m].max(),
                         "seed_std_ddof1": by_seed[:, m].std(ddof=1), "fixed_fits": len(seeds), "paths": paths,
                         "interval_level": .95, "scope": "exploratory unadjusted descriptive; fixed five paired fits"})
            for j, seed in enumerate(seeds):
                seed_rows.append({**dict(zip(keys, key)), "metric": metric, "seed": seed,
                                  "estimate": by_seed[j, m], "path_ci_low": seed_bounds[0, j, m],
                                  "path_ci_high": seed_bounds[1, j, m]})
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.self_check:
        print(json.dumps(tiny_self_check()))
        return
    protocol, training, _, original = verify_original_inputs()
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                               check=True, capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest() == sha(__file__), "commit new diagnostic before execution"
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Existing diagnostic manifest; refusing silent rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {"status": "running", "design_status": "exploratory; specified after Stage 2 outcomes were viewed",
                "original_protocol_unchanged": True, "new_blind_test": False,
                "model_selection_or_training": "none; all ten original checkpoints retained",
                "code_commit_before_execution": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip(),
                "source_sha256": sha(__file__), "protocol_sha256": original["protocol_sha256"],
                "training_manifest_sha256": original["training_manifest_sha256"],
                "original_evaluation_manifest_sha256": sha(EVAL / "manifest.json"),
                "original_evaluator_and_dependencies_sha256": original["sources"],
                "checkpoint_sha256": {e["checkpoint"]: e["checkpoint_sha256"] for e in training["models"]},
                "cells": "current context a16/64 x archive denominator context b16/64 x W32/128",
                "families": ["gaussian", "student_t5"], "duration": 128, "amplitude": 3,
                "bootstrap": {"repetitions": 4000, "path_seed": 610023, "level": .95,
                              "scope": "unadjusted descriptive paired-path interval conditional on five fixed paired fits"},
                "self_check": tiny_self_check(), "test_array_hashes": {},
                "event_label_use": "path generation, phase evaluation and reuse of mathematically identical contexts only",
                "fixed_archive_logwidth_current_window_identity": "zero by log h=log s_a+log Q_b,W; not empirical novelty"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    tc, config = protocol["training_config"], protocol["test"]
    seeds = tc["model_seeds"]
    val_y = generate_paths(tc["validation_data"], tc["generator"])
    assert array_sha256(val_y) == training["data_sha256"]["validation"]
    models, warmups = {}, {}
    for entry in training["models"]:
        key = (entry["architecture"], entry["model_seed"])
        model, _ = load_checkpoint(ROOT / entry["checkpoint"])
        models[key] = model
        for context in CONTEXTS:
            vs = predict_scales(model, val_y[:1], context, val_y.shape[1]-128)
            warmups[(*key, context)] = np.abs(val_y[0, -128:])/vs[0]
    old_performance = pd.read_csv(EVAL / "performance_by_seed.csv")
    old_interactions = pd.read_csv(EVAL / "paired_interactions.csv.gz")
    old_perf = old_performance.set_index(["family", "duration", "amplitude", "architecture", "seed", "context", "window", "phase", "world"])
    old_effect = old_interactions.set_index(["family", "duration", "amplitude", "architecture", "seed", "phase", "path"])
    performance, path_effects, negative_controls = [], [], []
    diagonal_performance_error = diagonal_effect_error = 0.0
    fixed_archive_logwidth_error = 0.0
    onset, duration, amplitude = config["onset"], 128, 3
    exit_time = onset+duration
    slices = phases(onset, duration, config["length"])
    for family in ("gaussian", "student_t5"):
        family_index = config["families"].index(family)
        yc, sc = generate_base(family, config, family_index)
        hashes = {"observations": array_sha256(yc), "true_scale": array_sha256(sc)}
        assert hashes == original["test_arrays_sha256"][family]
        ye, se = yc.copy(), sc.copy()
        ye[:, onset:exit_time] *= amplitude
        se[:, onset:exit_time] *= amplitude
        manifest["test_array_hashes"][family] = {**hashes, "event_observations": array_sha256(ye),
                                              "event_true_scale": array_sha256(se)}
        with np.load(EVAL / f"reference_{family}.npz") as ref:
            for name, value in (("y_control", yc), ("y_event", ye), ("true_sigma_control", sc), ("true_sigma_event", se)):
                np.testing.assert_array_equal(ref[name], value)
            for architecture in tc["architectures"]:
                for seed in seeds:
                    model = models[(architecture, seed)]
                    scales = {"event": {}, "control": {}}
                    for context in CONTEXTS:
                        if seed == 4101:
                            scales["control"][context] = ref[f"{architecture}_c{context}_control"].copy()
                            scales["event"][context] = ref[f"{architecture}_c{context}_event"].copy()
                        else:
                            control = predict_scales(model, yc, context)
                            event = control.copy()
                            event[:, onset+1:exit_time+context] = predict_scales(model, ye, context, onset+1, exit_time+context)
                            scales["control"][context], scales["event"][context] = control, event
                        for j in (onset, exit_time+context):
                            check = predict_scales(model, ye[:2], context, j, j+1)
                            np.testing.assert_allclose(scales["event"][context][:2, j:j+1], check, rtol=0, atol=2e-6)
                    means = {phase: {"event": {}, "control": {}} for phase in slices}
                    halves = {}
                    for a in CONTEXTS:
                        for b in CONTEXTS:
                            for window in WINDOWS:
                                cell = (a, b, window)
                                for world, y, true in (("event", ye, se), ("control", yc, sc)):
                                    half = crossed_half_width(y, scales[world][a], scales[world][b], window,
                                                              config["alpha"], warmups[(architecture, seed, b)])
                                    halves[(world, *cell)] = half
                                    arrays = metric_arrays(y, half, true, family, config)
                                    arrays["log_width"] = np.log(2*half)
                                    for phase, (lo, hi) in slices.items():
                                        pp = np.stack([arrays[m][:, lo:hi].mean(axis=1) for m in ALL_METRICS], axis=-1)
                                        means[phase][world][cell] = pp
                                        avg = pp.mean(axis=0)
                                        performance.append({"family": family, "architecture": architecture, "seed": seed,
                                                            "current_context": a, "archive_context": b, "window": window,
                                                            "phase": phase, "world": world, "paths": len(y), "steps": hi-lo,
                                                            **dict(zip(ALL_METRICS, avg))})
                                        if a == b:
                                            old = old_perf.loc[(family, duration, amplitude, architecture, seed, a, window, phase, world), METRICS].to_numpy(float)
                                            error = float(np.max(np.abs(avg[:len(METRICS)]-old)))
                                            assert error <= 1e-10, (family, architecture, seed, a, window, phase, world, error)
                                            diagonal_performance_error = max(diagonal_performance_error, error)
                                he, hc = halves[("event", *cell)], halves[("control", *cell)]
                                np.testing.assert_allclose(he[:, :onset+1], hc[:, :onset+1], rtol=0, atol=2e-6)
                                bound = exit_time+max(a, b+window)
                                error = float(np.max(np.abs(he[:, bound:]-hc[:, bound:])))
                                assert error <= 2e-6, (family, architecture, seed, cell, bound, error)
                                negative_controls.append({"family": family, "architecture": architecture, "seed": seed,
                                                          "current_context": a, "archive_context": b, "window": window,
                                                          "forgetting_bound": bound, "half_width_max_abs_error_after_bound": error})
                    for phase in slices:
                        effect = {cell: means[phase]["event"][cell]-means[phase]["control"][cell]
                                  for cell in means[phase]["event"]}
                        parts = decompose(effect)
                        for component, values in parts.items():
                            if component.startswith("current_"):
                                fixed_archive_logwidth_error = max(fixed_archive_logwidth_error, float(np.max(np.abs(values[:, -1]))))
                                np.testing.assert_allclose(values[:, -1], 0, rtol=0, atol=1e-12)
                            for path, row in enumerate(values):
                                path_effects.append({"family": family, "architecture": architecture, "seed": seed,
                                                     "phase": phase, "component": component, "path": path,
                                                     **dict(zip(ALL_METRICS, row))})
                        expected = np.stack([old_effect.loc[(family, duration, amplitude, architecture, seed, phase, p), METRICS].to_numpy(float)
                                             for p in range(config["paths"])])
                        error = float(np.max(np.abs(parts["total"][:, :len(METRICS)]-expected)))
                        assert error <= 1e-10, (family, architecture, seed, phase, error)
                        diagonal_effect_error = max(diagonal_effect_error, error)
                    print(json.dumps({"family": family, "architecture": architecture, "seed": seed,
                                      "elapsed_seconds": round(time.time()-started, 2)}), flush=True)
    perf, effects = pd.DataFrame(performance), pd.DataFrame(path_effects)
    summary, seed_summary = summarize(effects, seeds, config["paths"])
    perf.to_csv(OUT / "crossed_performance_by_seed.csv", index=False)
    effects.to_csv(OUT / "paired_component_effects.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary.to_csv(OUT / "mechanism_summary.csv", index=False)
    seed_summary.to_csv(OUT / "individual_seed_components.csv", index=False)
    pd.DataFrame(negative_controls).to_csv(OUT / "negative_controls.csv", index=False)
    assert manifest["source_sha256"] == sha(__file__), "diagnostic source changed while running"
    verify_original_inputs()
    manifest.update(status="complete", elapsed_seconds=time.time()-started,
                    performance_rows=len(perf), paired_component_rows=len(effects),
                    diagonal_check={"all_original_five_metrics": True, "all_ten_fits": True,
                                    "families": 2, "phases": 4, "performance_rows": 640,
                                    "maximum_performance_absolute_error": diagonal_performance_error,
                                    "interaction_path_rows": 10240,
                                    "maximum_path_interaction_absolute_error": diagonal_effect_error},
                    negative_control_checks=len(negative_controls),
                    maximum_half_width_error_after_each_crossed_bound=float(pd.DataFrame(negative_controls).half_width_max_abs_error_after_bound.max()),
                    maximum_fixed_archive_current_window_logwidth_interaction_absolute_error=fixed_archive_logwidth_error,
                    artifacts={p.name: sha(p) for p in OUT.iterdir() if p.name != "manifest.json"})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"],
                      "diagonal_performance_error": diagonal_performance_error,
                      "diagonal_path_interaction_error": diagonal_effect_error}), flush=True)


if __name__ == "__main__":
    main()
