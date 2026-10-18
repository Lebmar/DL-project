"""Post-hoc finite-rank sensitivity of the frozen Stage 3 models and paths.

No neural training, checkpoint selection, new test claims, or random rank draws.
Randomized-rank metrics are integrated over two candidate intervals exactly.
"""
from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from evaluate_stage2 import METRICS, ROOT, generate_base, metric_arrays, phases, predict_scales
from evaluate_stage3 import (COMPONENTS, FAST_ATOL, FAST_RTOL, decomposition, issued_scales,
                             past_quantiles, pulse_scales, verified_inputs)
from stage2_models import GaussianTCN, load_checkpoint
from train_stage2 import array_sha256, generate_paths

PROTOCOL = ROOT / "configs/rank_sensitivity_protocol.json"
OUT = ROOT / "outputs/rank_sensitivity"
REFERENCE = ROOT / "outputs/stage3_evaluation"
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
RULES = ("ceiling", "randomized90")
PHASES = ("entry", "recovery64")
WORLDS = ("control", "event")
RANKS = ("lower", "upper")
COMMON = ["panel", "family", "duration", "amplitude", "architecture", "seed"]
PART_KEYS = COMMON + ["rule", "phase", "component", "path"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    relative = str(Path(path).resolve().relative_to(ROOT))
    content = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    assert hashlib.sha256(content).hexdigest() == sha(path), f"Commit before sensitivity execution: {relative}"


def load_protocol():
    p = json.loads(PROTOCOL.read_text())
    assert p["frozen"] and p["design_status"] == "post-hoc sensitivity on already-viewed Stage3 paths"
    assert p["test"]["families"] == ["gaussian", "student_t5"]
    assert p["test"]["duration"] == 128 and p["test"]["amplitude"] == 3.0
    assert p["test"]["phases"] == list(PHASES) and p["rules"] == list(RULES)
    assert p["test"]["contexts"] == list(CONTEXTS) and p["test"]["windows"] == list(WINDOWS)
    assert p["test"]["alpha"] == .1 and p["test"]["same_stage3_observations"] is True
    assert p["new_blind_test"] is False and p["original_stage3_primary_analysis_replaced"] is False
    bounds = phases(p["test"]["onset"], p["test"]["duration"], p["test"]["length"])
    assert p["test"]["entry_bounds"] == list(bounds["entry"])
    assert p["test"]["recovery64_bounds"] == list(bounds["recovery64"])
    for window in WINDOWS:
        lower, upper, probability = rank_spec(window, p["test"]["alpha"])
        assert p["rank_rules"][f"window{window}"] == {
            "lower": lower, "upper": upper, "upper_probability": probability}
    assert p["models"] == {"panels": [1, 2, 3], "seeds": [4101, 4102, 4103, 4104, 4105],
                           "architectures": ["gru", "tcn"], "count": 30, "exclusions": "none"}
    for name, value in p["frozen_dependencies"].items():
        assert sha(ROOT / name) == value, name
    return p


def rank_spec(window, alpha=.1):
    """1-based adjacent ranks and upper-rank probability for common ideal level."""
    target = (window + 1) * (1 - Fraction(str(alpha)))
    lower = target.numerator // target.denominator
    upper = (target.numerator + target.denominator - 1) // target.denominator
    assert 1 <= lower <= upper <= window
    weight = float(target - lower) if lower != upper else 0.0
    return lower, upper, weight


def candidate_quantiles(y, archive_scale, window, alpha, warmup):
    lower, upper, _ = rank_spec(window, alpha)
    warmup = np.asarray(warmup)
    assert warmup.ndim == 1 and len(warmup) >= window
    assert y.shape == archive_scale.shape and np.all(archive_scale > 0)
    score = np.abs(y) / archive_scale
    warm = np.broadcast_to(warmup, (len(y), len(warmup)))
    history = np.concatenate((warm, score), axis=1)
    views = np.lib.stride_tricks.sliding_window_view(history, window, axis=1)
    views = views[:, len(warmup)-window:len(warmup)-window+y.shape[1]]
    ordered = np.partition(views, (lower-1, upper-1), axis=-1)
    return np.stack((ordered[..., lower-1], ordered[..., upper-1]), axis=0)


def mix_metrics(candidates, window, rule, rank_axis=0, alpha=.1):
    """Expected METRIC, not a metric evaluated at the expected half-width."""
    assert rule in RULES
    low = np.take(candidates, 0, axis=rank_axis)
    high = np.take(candidates, 1, axis=rank_axis)
    if rule == "ceiling":
        return high
    _, _, weight = rank_spec(window, alpha)
    return (1-weight)*low + weight*high


def tiny_self_check():
    rng = np.random.default_rng(271018)
    y = rng.normal(size=(3, 270))
    scales = np.exp(rng.normal(size=y.shape)*.2)
    warm = rng.uniform(.1, 2.0, 128)
    max_error = 0.0
    for window in WINDOWS:
        lower, upper, p = rank_spec(window)
        np.testing.assert_allclose(((1-p)*lower+p*upper)/(window+1), .9, rtol=0, atol=1e-15)
        actual = candidate_quantiles(y, scales, window, .1, warm)
        history = np.concatenate((np.broadcast_to(warm, (3, 128)), np.abs(y)/scales), axis=1)
        expected = np.stack([np.stack([np.sort(history[:, 128+t-window:128+t], axis=1)[:, k-1]
                                       for t in range(y.shape[1])], axis=1) for k in (lower, upper)])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[1], past_quantiles(y, scales, window, .1, warm))
        changed = y.copy()
        changed[:, 170:] += 100
        # Both normalized score and scale history remain unchanged before t=170.
        changed_scale = scales.copy()
        changed_scale[:, 170:] *= 5
        np.testing.assert_array_equal(actual[:, :, :171],
            candidate_quantiles(changed, changed_scale, window, .1, warm)[:, :, :171])
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        values = rng.normal(size=(2, 3, 7, len(METRICS)))
        expected_mix = (1-p)*values[0]+p*values[1]
        np.testing.assert_allclose(mix_metrics(values, window, "randomized90"), expected_mix, rtol=0, atol=0)
        np.testing.assert_array_equal(mix_metrics(values, window, "ceiling"), values[1])
        # Enumerate the W+1 possible continuous exchangeable ranks exactly.
        ranks = np.arange(1, window+2)
        expected_coverage = np.mean((1-p)*(ranks <= lower)+p*(ranks <= upper))
        np.testing.assert_allclose(expected_coverage, .9, rtol=0, atol=1e-15)
    # A point crossing the two candidate widths makes the nonlinear-score error visible.
    observed, h0, h1, p = 1.0, .5, 2.0, .7
    score = lambda h: 2*h+20*max(abs(observed)-h, 0)
    true_mean = (1-p)*score(h0)+p*score(h1)
    score_of_mean = score((1-p)*h0+p*h1)
    assert abs(true_mean-score_of_mean) > 1.0
    # Expectation commutes with phase means, not with nonlinear scoring.
    yy = np.array([[.2, 1.0, 2.0]])
    cfg = {"alpha": .1, "student_t_df": 5}
    lows, highs = np.full_like(yy, h0), np.full_like(yy, h1)
    a0, a1 = metric_arrays(yy, lows, np.ones_like(yy), "gaussian", cfg), metric_arrays(yy, highs, np.ones_like(yy), "gaussian", cfg)
    for metric in METRICS:
        np.testing.assert_allclose(((1-p)*a0[metric]+p*a1[metric]).mean(axis=1),
                                   (1-p)*a0[metric].mean(axis=1)+p*a1[metric].mean(axis=1), atol=1e-15)
    effects = {cell: rng.normal(size=(7, len(METRICS))) for cell in product(CONTEXTS, CONTEXTS, WINDOWS)}
    parts = decomposition(effects)
    direct = effects[(64, 64, 128)]-effects[(64, 64, 32)]-effects[(16, 16, 128)]+effects[(16, 16, 32)]
    np.testing.assert_allclose(parts["total"], direct, atol=1e-14)
    # Independent scalar reconstruction audits the multidimensional saved-candidate layout.
    toy_keys = [(1, "gru", 41), (2, "tcn", 42)]
    toy_test = {"families": ["gaussian", "student_t5"], "duration": 128, "amplitude": 3.0}
    toy = rng.uniform(size=(2, 2, 2, 2, 2, 2, 2, 2, 3, len(METRICS)))
    table = component_frame(toy, toy_keys, toy_test).set_index(PART_KEYS)
    cell_table = candidate_frame(toy, toy_keys, toy_test)
    toy_error = 0.0
    for mi, key in enumerate(toy_keys):
        for fi, family in enumerate(toy_test["families"]):
            for rule in RULES:
                for pi, phase in enumerate(PHASES):
                    d = {}
                    for ai, a in enumerate(CONTEXTS):
                        for bi, b in enumerate(CONTEXTS):
                            effects = []
                            for zi, window in enumerate(WINDOWS):
                                weight = 1.0 if rule == "ceiling" else rank_spec(window)[2]
                                control = (1-weight)*toy[mi, fi, 0, ai, bi, zi, 0, pi]+weight*toy[mi, fi, 0, ai, bi, zi, 1, pi]
                                event = (1-weight)*toy[mi, fi, 1, ai, bi, zi, 0, pi]+weight*toy[mi, fi, 1, ai, bi, zi, 1, pi]
                                effects.append(event-control)
                                for world, expected in (("control", control), ("event", event)):
                                    row = cell_table[(cell_table.panel == key[0]) & (cell_table.family == family)
                                        & (cell_table.rule == rule) & (cell_table.phase == phase) & (cell_table.world == world)
                                        & (cell_table.current_context == a) & (cell_table.archive_context == b)
                                        & (cell_table.window == window)]
                                    assert len(row) == 1
                                    np.testing.assert_allclose(row[METRICS].to_numpy()[0], expected.mean(axis=0), atol=1e-15)
                            d[(a, b)] = effects[1]-effects[0]
                    expected_parts = {"total": d[(64, 64)]-d[(16, 16)],
                        "current_at_archive64": d[(64, 64)]-d[(16, 64)],
                        "archive_at_current16": d[(16, 64)]-d[(16, 16)],
                        "current_at_archive16": d[(64, 16)]-d[(16, 16)],
                        "archive_at_current64": d[(64, 64)]-d[(64, 16)],
                        "third_order": d[(64, 64)]-d[(16, 64)]-d[(64, 16)]+d[(16, 16)]}
                    for component, expected in expected_parts.items():
                        for path in range(3):
                            identity = (key[0], family, 128, 3.0, key[1], key[2], rule, phase, component, path)
                            actual = table.loc[identity, METRICS].to_numpy(dtype=float)
                            np.testing.assert_allclose(actual, expected[path], atol=1e-15)
                            toy_error = max(toy_error, float(np.max(np.abs(actual-expected[path]))))
    return {"passed": True, "adjacent_ranks": {str(w): rank_spec(w) for w in WINDOWS},
            "candidate_layout_and_scalar_component_max_error": toy_error,
            "scalar_quantile_max_error": max_error, "ceiling_matches_frozen_quantile_bitwise": True,
            "current_and_future_score_exclusion": True, "ideal_rank_coverage_exact_enumeration": True,
            "metric_expectation_commutes_with_phase_mean": True,
            "nonlinear_counterexample_expected_score": true_mean,
            "nonlinear_counterexample_score_at_expected_width": score_of_mean,
            "both_decomposition_routes": True}


def candidate_frame(values, keys, test):
    """Expose both rules, all eight cells and both worlds, averaging paths only."""
    rows = []
    for mi, (panel, architecture, seed) in enumerate(keys):
        for fi, family in enumerate(test["families"]):
            common = dict(panel=panel, family=family, duration=test["duration"], amplitude=test["amplitude"],
                          architecture=architecture, seed=seed)
            for wi, world in enumerate(WORLDS):
                for ai, a in enumerate(CONTEXTS):
                    for bi, b in enumerate(CONTEXTS):
                        for zi, window in enumerate(WINDOWS):
                            candidates = values[mi, fi, wi, ai, bi, zi]
                            for rule in RULES:
                                mixed = mix_metrics(candidates, window, rule)
                                for pi, phase in enumerate(PHASES):
                                    rows.append({**common, "rule": rule, "world": world, "current_context": a,
                                        "archive_context": b, "window": window, "phase": phase,
                                        **dict(zip(METRICS, mixed[pi].mean(axis=0)))})
    return pd.DataFrame(rows)


def component_frame(values, keys, test):
    rows = []
    for mi, (panel, architecture, seed) in enumerate(keys):
        for fi, family in enumerate(test["families"]):
            common = dict(panel=panel, family=family, duration=test["duration"], amplitude=test["amplitude"],
                          architecture=architecture, seed=seed)
            for rule in RULES:
                for pi, phase in enumerate(PHASES):
                    effect = {}
                    for ai, a in enumerate(CONTEXTS):
                        for bi, b in enumerate(CONTEXTS):
                            for zi, window in enumerate(WINDOWS):
                                worlds = mix_metrics(values[mi, fi, :, ai, bi, zi, :, pi], window, rule, rank_axis=1)
                                effect[(a, b, window)] = worlds[1]-worlds[0]
                    for component, matrix in decomposition(effect).items():
                        for path, row in enumerate(matrix):
                            rows.append({**common, "rule": rule, "phase": phase, "component": component,
                                         "path": path, **dict(zip(METRICS, row))})
    return pd.DataFrame(rows)


def reference_reproduction(parts, cells, evaluation):
    names = ["paired_component_effects.csv.gz", "crossed_performance_by_seed.csv"]
    for name in names:
        assert sha(REFERENCE / name) == evaluation["artifacts"][name]
    ref = pd.read_csv(REFERENCE / names[0])
    ref = ref[(ref.duration == 128) & ref.phase.isin(PHASES)]
    ids = COMMON + ["phase", "component", "path"]
    fresh = parts[parts.rule == "ceiling"]
    assert not ref.duplicated(ids).any() and not fresh.duplicated(ids).any()
    ref = ref.set_index(ids).sort_index()[METRICS]
    fresh = fresh.set_index(ids).sort_index()[METRICS]
    assert ref.index.equals(fresh.index)
    np.testing.assert_allclose(fresh, ref, rtol=0, atol=1e-10)
    part_error = float(np.max(np.abs(fresh.to_numpy()-ref.to_numpy())))
    ref_cells = pd.read_csv(REFERENCE / names[1])
    ref_cells = ref_cells[(ref_cells.duration == 128) & ref_cells.phase.isin(PHASES)]
    cell_ids = COMMON + ["current_context", "archive_context", "window", "phase", "world"]
    new_cells = cells[cells.rule == "ceiling"]
    assert not ref_cells.duplicated(cell_ids).any() and not new_cells.duplicated(cell_ids).any()
    aa = ref_cells.set_index(cell_ids).sort_index()[METRICS]
    bb = new_cells.set_index(cell_ids).sort_index()[METRICS]
    assert aa.index.equals(bb.index)
    np.testing.assert_allclose(aa, bb, rtol=0, atol=1e-10)
    return {"passed": True, "component_path_rows": len(ref), "cell_rows": len(aa),
            "component_path_max_abs_error": part_error,
            "all_cell_metric_max_abs_error": float(np.max(np.abs(aa.to_numpy()-bb.to_numpy()))),
            "reference_hashes": {n: sha(REFERENCE / n) for n in names}, "absolute_tolerance": 1e-10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check(), indent=2))
        return
    for file in (PROTOCOL, Path(__file__), ROOT / "scripts/analyze_rank_sensitivity.py", ROOT / "research/rank_sensitivity_protocol.md"):
        committed(file)
    protocol = load_protocol()
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite any existing rank-sensitivity output directory")
    frozen, training, panels, original_panel_hashes = verified_inputs()
    evaluation = json.loads((REFERENCE / "manifest.json").read_text())
    assert evaluation["status"] == "complete" and evaluation["models_evaluated"] == 30
    assert evaluation["source_sha256"] == sha(ROOT / "scripts/evaluate_stage3.py")
    assert evaluation["training_manifest_sha256"] == sha(ROOT / "outputs/stage3_training/manifest.json")
    assert evaluation["protocol_sha256"] == sha(ROOT / "configs/stage3_replication_protocol.json")
    config = frozen["test"]
    for key in ("paths", "length", "onset", "families", "alpha", "path_seed", "student_t_df"):
        assert protocol["test"][key] == config[key], key
    torch.set_num_threads(protocol["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.time()
    OUT.mkdir(parents=True)
    manifest = {"status": "running", "design_status": protocol["design_status"], "new_blind_test": False,
                "fitting_or_selection": "none", "primary_stage3_analysis_replaced": False,
                "model_exclusions": "none", "protocol_sha256": sha(PROTOCOL),
                "source_sha256": sha(__file__), "analysis_source_sha256": sha(ROOT / "scripts/analyze_rank_sensitivity.py"),
                "protocol_document_sha256": sha(ROOT / "research/rank_sensitivity_protocol.md"),
                "code_commit_before_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "stage3_evaluation_manifest_sha256": sha(REFERENCE / "manifest.json"),
                "training_manifest_sha256": sha(ROOT / "outputs/stage3_training/manifest.json"),
                "self_check": tiny_self_check(), "rank_specs": {str(w): rank_spec(w) for w in WINDOWS},
                "candidate_axes": ["model", "family", "world", "current_context", "archive_context", "window",
                                   "rank_choice", "phase", "path", "metric"],
                "randomization": "exact expectation over adjacent-rank interval metrics; no rank Monte Carlo",
                "coverage_deficit_definition": "expectation of per-candidate positive oracle deficit, not positive part after averaging probability",
                "test_array_sha256": {}, "event_array_sha256": {}, "validation_fast_inference_checks": []}
    write_manifest = lambda: (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    write_manifest()
    keys, models, warmups = [], {}, {}
    for panel in protocol["models"]["panels"]:
        fitted = panels[panel]
        tc = fitted["training_config"]
        val_y = generate_paths(tc["validation_data"], tc["generator"])
        assert array_sha256(val_y) == fitted["data_sha256"]["validation"]
        for entry in sorted(fitted["models"], key=lambda m: (m["architecture"], m["model_seed"])):
            key = (panel, entry["architecture"], entry["model_seed"])
            keys.append(key)
            model, payload = load_checkpoint(ROOT / entry["checkpoint"])
            assert payload["protocol_sha256"] == fitted["protocol_sha256"]
            models[key] = model
            if isinstance(model, GaussianTCN):
                optimized = issued_scales(model, val_y[:3, :512], 64)
                original = predict_scales(model, val_y[:3, :512], 64)
                accepted = bool(np.allclose(optimized, original, rtol=FAST_RTOL, atol=FAST_ATOL))
                model._stage3_fast_context64_verified = accepted
                manifest["validation_fast_inference_checks"].append({"panel": panel, "seed": key[2],
                    "maximum_absolute_error": float(np.max(np.abs(optimized-original))), "used": accepted})
            for context in CONTEXTS:
                scale = issued_scales(model, val_y[:1], context, val_y.shape[1]-128)
                warmups[(*key, context)] = np.abs(val_y[0, -128:])/scale[0]
    assert len(keys) == len(set(keys)) == 30
    shape = (len(keys), 2, 2, 2, 2, 2, 2, 2, config["paths"], len(METRICS))
    values = np.full(shape, np.nan, dtype=np.float64)
    checks = []
    duration, amplitude = protocol["test"]["duration"], protocol["test"]["amplitude"]
    onset, length = config["onset"], config["length"]
    phase_bounds = phases(onset, duration, length)
    for fi, family in enumerate(protocol["test"]["families"]):
        control_y, control_true = generate_base(family, config, fi)
        hashes = {"observations": array_sha256(control_y), "true_scale": array_sha256(control_true)}
        assert hashes == evaluation["test_array_sha256"][family]
        manifest["test_array_sha256"][family] = hashes
        event_y, event_true = control_y.copy(), control_true.copy()
        event_y[:, onset:onset+duration] *= amplitude
        event_true[:, onset:onset+duration] *= amplitude
        event_hashes = {"observations": array_sha256(event_y), "true_scale": array_sha256(event_true)}
        assert event_hashes == evaluation["event_array_sha256"][f"{family}_D128_A3"]
        manifest["event_array_sha256"][family] = event_hashes
        for mi, key in enumerate(keys):
            model = models[key]
            control_scales = {c: issued_scales(model, control_y, c) for c in CONTEXTS}
            event_scales = {c: pulse_scales(model, event_y, control_scales[c], c, onset, duration, 128) for c in CONTEXTS}
            quantiles = {}
            for wi, (world_y, world_scale) in enumerate(((control_y, control_scales), (event_y, event_scales))):
                for bi, b in enumerate(CONTEXTS):
                    for zi, window in enumerate(WINDOWS):
                        q = candidate_quantiles(world_y, world_scale[b], window, config["alpha"], warmups[(*key, b)])
                        np.testing.assert_array_equal(q[1], past_quantiles(world_y, world_scale[b], window, config["alpha"], warmups[(*key, b)]))
                        quantiles[(wi, b, window)] = q
            for ai, a in enumerate(CONTEXTS):
                for bi, b in enumerate(CONTEXTS):
                    for zi, window in enumerate(WINDOWS):
                        for ri, rank_name in enumerate(RANKS):
                            halves = (control_scales[a]*quantiles[(0, b, window)][ri],
                                      event_scales[a]*quantiles[(1, b, window)][ri])
                            before = float(np.max(np.abs(halves[1][:, :onset+1]-halves[0][:, :onset+1])))
                            bound = onset+duration+max(a, b+window)
                            after = float(np.max(np.abs(halves[1][:, bound:]-halves[0][:, bound:])))
                            assert before <= 2e-6 and after <= 2e-6
                            checks.append(dict(panel=key[0], architecture=key[1], seed=key[2], family=family,
                                current_context=a, archive_context=b, window=window, rank_choice=rank_name,
                                finite_memory_bound=bound, prefix_max_abs_error=before, post_bound_max_abs_error=after))
                            for wi, (yy, true, half) in enumerate(((control_y, control_true, halves[0]), (event_y, event_true, halves[1]))):
                                arrays = metric_arrays(yy, half, true, family, config)
                                for pi, phase in enumerate(PHASES):
                                    lo, hi = phase_bounds[phase]
                                    values[mi, fi, wi, ai, bi, zi, ri, pi] = np.stack([arrays[m][:, lo:hi].mean(axis=1) for m in METRICS], axis=-1)
            print(json.dumps({"family": family, "model": list(key), "elapsed_seconds": round(time.time()-started, 2)}), flush=True)
        write_manifest()
    assert np.isfinite(values).all()
    cells = candidate_frame(values, keys, protocol["test"])
    parts = component_frame(values, keys, protocol["test"])
    reproduction = reference_reproduction(parts, cells, evaluation)
    np.savez_compressed(OUT / "rank_candidate_phase_metrics.npz", values=values,
        model_panels=np.array([k[0] for k in keys]), model_architectures=np.array([k[1] for k in keys]),
        model_seeds=np.array([k[2] for k in keys]), families=np.array(protocol["test"]["families"]),
        worlds=np.array(WORLDS), current_contexts=np.array(CONTEXTS), archive_contexts=np.array(CONTEXTS),
        windows=np.array(WINDOWS), rank_choices=np.array(RANKS), phases=np.array(PHASES), metrics=np.array(METRICS),
        lower_ranks=np.array([rank_spec(w)[0] for w in WINDOWS]),
        upper_ranks=np.array([rank_spec(w)[1] for w in WINDOWS]),
        upper_probabilities=np.array([rank_spec(w)[2] for w in WINDOWS]))
    frames = {"cell_performance_by_seed.csv": cells, "paired_component_effects.csv.gz": parts,
              "negative_controls.csv": pd.DataFrame(checks)}
    for name, frame in frames.items():
        frame.to_csv(OUT / name, index=False, compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None)
    (OUT / "ceiling_reproduction.json").write_text(json.dumps(reproduction, indent=2)+"\n")
    _, _, _, final_panel_hashes = verified_inputs()
    assert final_panel_hashes == original_panel_hashes
    assert load_protocol() == protocol and sha(__file__) == manifest["source_sha256"]
    assert sha(ROOT / "scripts/analyze_rank_sensitivity.py") == manifest["analysis_source_sha256"]
    assert sha(ROOT / "research/rank_sensitivity_protocol.md") == manifest["protocol_document_sha256"]
    assert sha(REFERENCE / "manifest.json") == manifest["stage3_evaluation_manifest_sha256"]
    assert len(parts) == 184320 and len(cells) == 3840 and len(checks) == 960
    files = list(frames)+["rank_candidate_phase_metrics.npz", "ceiling_reproduction.json"]
    manifest.update(status="complete", models_evaluated=len(keys), cases=2, paths_per_family=config["paths"],
        candidate_shape=list(values.shape), elapsed_seconds=time.time()-started,
        rows={name: len(frame) for name, frame in frames.items()}, ceiling_reproduction=reproduction,
        negative_control_count=len(checks), maximum_prefix_error=max(c["prefix_max_abs_error"] for c in checks),
        maximum_post_bound_error=max(c["post_bound_max_abs_error"] for c in checks),
        artifacts={name: sha(OUT / name) for name in files})
    write_manifest()
    print(json.dumps({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"], "ceiling_reproduction": reproduction}), flush=True)


if __name__ == "__main__":
    main()
