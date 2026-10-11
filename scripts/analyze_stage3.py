"""Analyze all independently retrained Stage3 fits without selecting models.

Three training panels and five shared initialization/sampling seeds are crossed,
not 30 iid observations. Primary paths are resampled after averaging 15 fixed
paired fits. Shared seed draws across panels are retained in sensitivity only.
Validation-feature scatters are exploratory: no regression or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs/stage3_evaluation"
OUT = ROOT / "outputs/stage3_analysis"
TRAIN = ROOT / "outputs/stage3_training"
PROBES = ROOT / "outputs/stage3_response_probes"
PROTOCOL = ROOT / "configs/stage3_replication_protocol.json"
REPORT = ROOT / "research/stage3_results.md"
METRICS = ["interval_score", "coverage", "width", "coverage_deficit", "excess_width_to_oracle"]
COMPONENTS = ["total", "current_at_archive64", "archive_at_current16",
              "current_at_archive16", "archive_at_current64", "third_order"]
KEYS = ["family", "duration", "amplitude", "architecture", "phase"]
MODEL_KEYS = ["panel", "seed", "path"]
COMMON = ["panel", "family", "duration", "amplitude", "architecture", "seed"]
ARCHES = ["gru", "tcn"]
COLORS = {"gru": "#285a8a", "tcn": "#b56b2b"}
FEATURES = ["validation_nll", "older_than16_gradient_mass_fraction",
            "entry_pulse_context_contrast_mean_signed"]
FEATURE_LABELS = ["Mean validation NLL", "Older-input gradient fraction",
                  "Validation entry pulse: signed context contrast"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed(path):
    relative = str(Path(path).resolve().relative_to(ROOT))
    value = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    assert hashlib.sha256(value).hexdigest() == sha(path), f"Commit source before test access: {relative}"
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", relative],
                                   cwd=ROOT, text=True).strip()


def verified_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    training_path = TRAIN / "manifest.json"
    evaluation_path = IN / "manifest.json"
    probe_path = PROBES / "manifest.json"
    training = json.loads(training_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    probes = json.loads(probe_path.read_text())
    assert protocol["frozen"] and training["status"] == evaluation["status"] == probes["status"] == "complete"
    assert evaluation["protocol_sha256"] == training["protocol_sha256"] == probes["protocol_sha256"] == sha(PROTOCOL)
    assert training["completed_models"] == evaluation["models_evaluated"] == probes["model_rows"] == 30
    assert evaluation["cases"] == 4 and evaluation["model_exclusions"] == "none"
    assert evaluation["training_manifest_sha256"] == probes["training_manifest_sha256"] == sha(training_path)
    assert evaluation["training_driver_sha256"] == training["driver_sha256"] == sha(ROOT / "scripts/train_stage3_repeats.py")
    assert evaluation["source_sha256"] == sha(ROOT / "scripts/evaluate_stage3.py")
    assert evaluation["frozen_dependency_sha256"] == training["sources"] == protocol["sources"]
    for name, value in protocol["sources"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    for name, value in probes["sources"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    assert probes["config_sha256"] == sha(ROOT / "configs/stage3_response_probes.json")
    assert probes["replication_test_data_accessed"] is False and probes["checkpoint_selection_or_updates"] is False
    model_rows = []
    for item in protocol["panels"]:
        panel = item["panel"]
        record = next(p for p in training["panels"] if p["panel"] == panel)
        panel_path = ROOT / record["manifest"]
        assert sha(panel_path) == record["manifest_sha256"]
        fitted = json.loads(panel_path.read_text())
        assert fitted["status"] == "complete" and fitted["matching_minibatches_verified"]
        assert fitted["training_config"] == item["training_config"]
        assert len(fitted["models"]) == 10
        assert set((m["architecture"], m["model_seed"]) for m in fitted["models"]) == set(product(ARCHES, item["training_config"]["model_seeds"]))
        for model in fitted["models"]:
            assert sha(ROOT / model["checkpoint"]) == model["checkpoint_sha256"]
            model_rows.append({"panel": panel, "architecture": model["architecture"],
                               "seed": model["model_seed"], "validation_nll": model["best_validation_mean_nll"],
                               "checkpoint_sha256": model["checkpoint_sha256"]})
    for hashes in evaluation["panel_artifact_sha256"].values():
        for name, value in hashes.items():
            assert sha(ROOT / name) == value, name
    for directory, manifest in ((IN, evaluation), (PROBES, probes)):
        for name, value in manifest["artifacts"].items():
            assert sha(directory / name) == value, name
    assert evaluation["test"] == protocol["test"]
    return protocol, training, evaluation, probes, pd.DataFrame(model_rows)


def require_grid(frame, index, expected, columns, numeric=METRICS):
    """Reject extra columns, duplicate rows, missing and off-grid identities."""
    assert set(frame.columns) == set(columns), (set(frame.columns)-set(columns), set(columns)-set(frame.columns))
    assert not frame.duplicated(index).any(), f"duplicate index {index}"
    actual = pd.MultiIndex.from_frame(frame[index])
    assert len(actual) == len(expected) and set(actual) == set(expected), f"missing/off-grid rows: {index}"
    assert np.isfinite(frame[numeric].to_numpy(dtype=float)).all()


def grid_ids(protocol):
    panels = [p["panel"] for p in protocol["panels"]]
    seeds = protocol["panels"][0]["training_config"]["model_seeds"]
    for item in protocol["panels"]:
        assert item["training_config"]["model_seeds"] == seeds
        assert item["training_config"]["architectures"] == ARCHES
    test = protocol["test"]
    return panels, seeds, test["paths"], test["phases"]


def effect_tensor(frame, panels, seeds, paths):
    expected = pd.MultiIndex.from_product([panels, seeds, range(paths)], names=MODEL_KEYS)
    assert not frame.duplicated(MODEL_KEYS).any()
    indexed = frame.set_index(MODEL_KEYS)
    assert len(indexed) == len(expected) and set(indexed.index) == set(expected)
    result = indexed.reindex(expected)[METRICS].to_numpy().reshape(len(panels), len(seeds), paths, len(METRICS))
    assert np.isfinite(result).all()
    return result


def path_weights(paths, replicates, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(paths, size=(replicates, paths))
    weights = np.zeros((replicates, paths), dtype=float)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1.0/paths)
    return indices, weights


def interval(samples, level):
    tail = (1-level)/2
    return np.quantile(samples, [tail, 1-tail], axis=0)


def crossed_samples(effect, panel_indices, seed_indices, path_indices):
    # effect[P,S,N]; the SAME seed draw is broadcast to every sampled panel.
    return effect[panel_indices[:, :, None, None], seed_indices[:, None, :, None],
                  path_indices[:, None, None, :]].mean(axis=(1, 2, 3))


def tiny_self_check():
    rng = np.random.default_rng(36102)
    values = rng.normal(size=(3, 5, 7))
    indices, weights = path_weights(7, 80, 9001)
    conditional = weights @ values.mean(axis=(0, 1))
    slow_conditional = np.array([np.mean([values[p, s, n] for p in range(3) for s in range(5) for n in row]) for row in indices])
    np.testing.assert_allclose(conditional, slow_conditional, rtol=0, atol=1e-14)
    panels = rng.integers(3, size=(80, 3))
    seeds = rng.integers(5, size=(80, 5))
    crossed = crossed_samples(values, panels, seeds, indices)
    slow_crossed = np.array([np.mean([values[p, s, n] for p in pp for s in ss for n in nn])
                            for pp, ss, nn in zip(panels, seeds, indices)])
    np.testing.assert_allclose(crossed, slow_crossed, rtol=0, atol=1e-14)
    common = rng.normal(size=values.shape)
    gru, tcn = common+2, common-.5
    np.testing.assert_allclose(weights @ (gru-tcn).mean(axis=(0, 1)), 2.5, atol=1e-14)
    rows = [{"panel": p, "seed": s, "path": n,
             **dict.fromkeys(METRICS, 100*p+10*s+n)} for p in [1, 2, 3] for s in [41, 42] for n in range(7)]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=1)
    tensor = effect_tensor(frame, [1, 2, 3], [41, 42], 7)
    assert tensor[1, 1, 6, 0] == 626
    offgrid = frame.copy()
    offgrid.loc[offgrid.index[0], "path"] = 99
    rejected = 0
    for broken in (frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), offgrid):
        try:
            effect_tensor(broken, [1, 2, 3], [41, 42], 7)
        except AssertionError:
            rejected += 1
    assert rejected == 3
    return {"passed": True, "conditional_bootstrap_loop_error": float(np.max(np.abs(conditional-slow_conditional))),
            "crossed_bootstrap_loop_error": float(np.max(np.abs(crossed-slow_crossed))),
            "shared_seed_draw_broadcast_across_panels": True,
            "GRU_minus_TCN_direction_verified": True, "shuffled_index_reconstructed": True,
            "missing_duplicate_and_offgrid_rows_rejected": True}


def load_frames(protocol, evaluation):
    panels, seeds, paths, phases = grid_ids(protocol)
    cfg = protocol["test"]
    ids = COMMON+["phase", "path"]
    levels = [panels, cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES, seeds, phases, range(paths)]
    expected = pd.MultiIndex.from_product(levels, names=ids)
    interactions = pd.read_csv(IN / "paired_interactions.csv.gz")
    require_grid(interactions, ids, expected, ids+METRICS)
    components = pd.read_csv(IN / "paired_component_effects.csv.gz")
    component_ids = ids+["component"]
    expected_components = pd.MultiIndex.from_product(levels+[COMPONENTS], names=component_ids)
    require_grid(components, component_ids, expected_components, component_ids+METRICS)
    performance = pd.read_csv(IN / "performance_by_seed.csv")
    perf_ids = COMMON+["context", "window", "calibrator", "phase", "world"]
    basic = product(panels, cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES, seeds)
    methods = [(context, w, f"rolling{w}" if w else "native_gaussian") for context in cfg["contexts"] for w in [0]+cfg["rolling_windows"]]
    expected_perf = pd.MultiIndex.from_tuples([(*base, *method, phase, world) for base in basic for method in methods
                                              for phase in phases for world in ["event", "control"]], names=perf_ids)
    require_grid(performance, perf_ids, expected_perf, perf_ids+["paths", "steps"]+METRICS)
    assert (performance.paths == paths).all()
    for phase in phases:
        expected_steps = 32 if phase == "entry" else 64 if phase in ["recovery64", "late"] else 192
        assert (performance.loc[performance.phase == phase, "steps"] == expected_steps).all()
    crossed = pd.read_csv(IN / "crossed_performance_by_seed.csv")
    crossed_ids = COMMON+["current_context", "archive_context", "window", "phase", "world"]
    expected_crossed = pd.MultiIndex.from_product([panels, cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES,
        seeds, cfg["contexts"], cfg["contexts"], cfg["rolling_windows"], phases, ["event", "control"]], names=crossed_ids)
    require_grid(crossed, crossed_ids, expected_crossed, crossed_ids+["paths", "steps"]+METRICS)
    negative = pd.read_csv(IN / "negative_controls.csv")
    negative_ids = COMMON+["current_context", "archive_context", "window"]
    expected_negative = pd.MultiIndex.from_product([panels, cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES,
        seeds, cfg["contexts"], cfg["contexts"], cfg["rolling_windows"]], names=negative_ids)
    negative_numeric = ["forgetting_bound", "half_width_max_abs_error_after_bound"]
    require_grid(negative, negative_ids, expected_negative, negative_ids+negative_numeric, negative_numeric)
    bounds = cfg["onset"]+negative.duration+np.maximum(negative.current_context, negative.archive_context+negative.window)
    np.testing.assert_array_equal(negative.forgetting_bound, bounds)
    assert negative.half_width_max_abs_error_after_bound.between(0, 2e-6).all()
    diagonal = pd.read_csv(IN / "diagonal_reproduction.csv")
    diagonal_ids = COMMON+["context", "window"]
    expected_diagonal = pd.MultiIndex.from_product([panels, cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES,
        seeds, cfg["contexts"], cfg["rolling_windows"]], names=diagonal_ids)
    diagonal_numeric = ["event_half_width_error", "control_half_width_error", "all_five_metric_max_abs_error"]
    require_grid(diagonal, diagonal_ids, expected_diagonal, diagonal_ids+diagonal_numeric, diagonal_numeric)
    assert ((diagonal[diagonal_numeric] >= 0) & (diagonal[diagonal_numeric] <= 1e-10)).all().all()
    assert len(negative) == evaluation["rows"]["negative_controls.csv"] == 960
    assert len(diagonal) == evaluation["rows"]["diagonal_reproduction.csv"] == 480
    for name, frame in (("paired_interactions.csv.gz", interactions), ("paired_component_effects.csv.gz", components),
                        ("performance_by_seed.csv", performance), ("crossed_performance_by_seed.csv", crossed)):
        assert len(frame) == evaluation["rows"][name]
    performance_index = COMMON+["phase", "context", "window", "world"]
    deployed = performance[performance.window != 0].set_index(performance_index)[METRICS].sort_index()
    means = interactions.groupby(COMMON+["phase"])[METRICS].mean().sort_index()
    diagonal_mean_error = 0.
    for identity, row in means.iterrows():
        direct = np.zeros(len(METRICS))
        for context, window, sign in ((64, 128, 1), (16, 128, -1), (64, 32, -1), (16, 32, 1)):
            direct += sign*(deployed.loc[(*identity, context, window, "event")].to_numpy()
                            -deployed.loc[(*identity, context, window, "control")].to_numpy())
        diagonal_mean_error = max(diagonal_mean_error, float(np.max(np.abs(direct-row.to_numpy()))))
        np.testing.assert_allclose(direct, row.to_numpy(), rtol=0, atol=1e-10)
    return interactions, components, performance, crossed, {"independent_performance_diagonal_mean_error": diagonal_mean_error}


def summarize(tensor, weights, panels, seeds, level=.95):
    estimate = tensor.mean(axis=(0, 1, 2))
    bounds = interval(weights @ tensor.mean(axis=(0, 1)), level)
    fit_means = tensor.mean(axis=2)
    rows, panel_rows, individual = [], [], []
    panel_bounds_all = [interval(weights @ tensor[p].mean(axis=0), level) for p in range(len(panels))]
    for j, metric in enumerate(METRICS):
        rows.append({"metric": metric, "estimate": estimate[j], "ci_low": bounds[0, j], "ci_high": bounds[1, j],
                     "interval_level": level, "fit_min": fit_means[:, :, j].min(), "fit_max": fit_means[:, :, j].max(),
                     "fixed_paired_fits": len(panels)*len(seeds), "interval_scope": "unadjusted descriptive; conditional on fixed fits"})
        for p, panel in enumerate(panels):
            panel_bounds = panel_bounds_all[p]
            per_seed = fit_means[p, :, j]
            panel_rows.append({"panel": panel, "metric": metric, "estimate": per_seed.mean(),
                               "seed_std": per_seed.std(ddof=1), "seed_min": per_seed.min(), "seed_max": per_seed.max(),
                               "ci_low": panel_bounds[0, j], "ci_high": panel_bounds[1, j],
                               "interval_level": level, "fixed_paired_fits": len(seeds)})
            for s, seed in enumerate(seeds):
                individual.append({"panel": panel, "seed": seed, "metric": metric, "estimate": per_seed[s]})
    return rows, panel_rows, individual


def analyze(interactions, components, protocol):
    panels, seeds, n, phases = grid_ids(protocol)
    cfg, spec = protocol["test"], protocol["analysis"]
    ref = protocol["evaluation"]["primary_reference"]
    b, seed = spec["bootstrap_replicates"], spec["bootstrap_seed"]
    path_indices, primary_weights = path_weights(n, b, seed)
    _, descriptive_weights = path_weights(n, b, seed+2)
    rng = np.random.default_rng(seed+1)
    panel_indices = rng.integers(len(panels), size=(b, len(panels)))
    seed_indices = rng.integers(len(seeds), size=(b, len(seeds)))
    tensors, parts = {}, {}
    tables = {key: [] for key in ["secondary_interactions", "individual_seed_effects", "panel_interactions",
              "component_summary", "component_panel_summary", "individual_seed_components",
              "secondary_architecture_contrasts", "component_architecture_contrasts", "component_variability",
              "primary_contrasts", "primary_panel_contrasts", "primary_individual_seeds", "primary_leave_one_panel_out"]}
    for key, group in interactions.groupby(KEYS, sort=True):
        tensor = effect_tensor(group, panels, seeds, n)
        tensors[key] = tensor
        rows, panel_rows, individual = summarize(tensor, descriptive_weights, panels, seeds)
        for target, source in (("secondary_interactions", rows), ("panel_interactions", panel_rows), ("individual_seed_effects", individual)):
            tables[target].extend({**dict(zip(KEYS, key)), **row} for row in source)
    assert set(tensors) == set(product(cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES, phases))
    for key, group in components.groupby(KEYS+["component"], sort=True):
        tensor = effect_tensor(group, panels, seeds, n)
        parts[key] = tensor
        rows, panel_rows, individual = summarize(tensor, descriptive_weights, panels, seeds)
        for target, source in (("component_summary", rows), ("component_panel_summary", panel_rows), ("individual_seed_components", individual)):
            tables[target].extend({**dict(zip(KEYS+["component"], key)), **row} for row in source)
    assert set(parts) == set(product(cfg["families"], cfg["durations"], cfg["amplitudes"], ARCHES, phases, COMPONENTS))
    diagonal_error = decomposition_error = entry_duplicate_error = 0.
    for key, tensor in tensors.items():
        diagonal_error = max(diagonal_error, float(np.max(np.abs(tensor-parts[(*key, "total")]))))
        np.testing.assert_allclose(tensor, parts[(*key, "total")], rtol=0, atol=1e-10)
        for current, archive in [(COMPONENTS[1], COMPONENTS[2]), (COMPONENTS[3], COMPONENTS[4])]:
            error = tensor-parts[(*key, current)]-parts[(*key, archive)]
            decomposition_error = max(decomposition_error, float(np.max(np.abs(error))))
            np.testing.assert_allclose(error, 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(parts[(*key, COMPONENTS[5])], parts[(*key, COMPONENTS[1])]-parts[(*key, COMPONENTS[3])], rtol=0, atol=1e-10)
        np.testing.assert_allclose(parts[(*key, COMPONENTS[5])], parts[(*key, COMPONENTS[4])]-parts[(*key, COMPONENTS[2])], rtol=0, atol=1e-10)
    # Duration32 and128 have identical first32 observations and forecasts.
    for family, amplitude, architecture in product(cfg["families"], cfg["amplitudes"], ARCHES):
        first = tensors[(family, 32, amplitude, architecture, "entry")]
        second = tensors[(family, 128, amplitude, architecture, "entry")]
        entry_duplicate_error = max(entry_duplicate_error, float(np.max(np.abs(first-second))))
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-10)
        for component in COMPONENTS:
            np.testing.assert_allclose(parts[(family, 32, amplitude, architecture, "entry", component)],
                                       parts[(family, 128, amplitude, architecture, "entry", component)], rtol=0, atol=1e-10)
    for family, duration, amplitude, phase in product(cfg["families"], cfg["durations"], cfg["amplitudes"], phases):
        gru = tensors[(family, duration, amplitude, "gru", phase)]
        tcn = tensors[(family, duration, amplitude, "tcn", phase)]
        difference = gru-tcn
        base = {"family": family, "duration": duration, "amplitude": amplitude, "phase": phase}
        rows, _, _ = summarize(difference, descriptive_weights, panels, seeds)
        tables["secondary_architecture_contrasts"].extend({**base, **r} for r in rows)
        for component in COMPONENTS:
            gp = parts[(family, duration, amplitude, "gru", phase, component)]
            tp = parts[(family, duration, amplitude, "tcn", phase, component)]
            rows, _, _ = summarize(gp-tp, descriptive_weights, panels, seeds)
            tables["component_architecture_contrasts"].extend({**base, "component": component, **r} for r in rows)
        for p, panel in enumerate(panels):
            for route, current, archive in [("archive64_first", COMPONENTS[1], COMPONENTS[2]),
                                             ("current64_first", COMPONENTS[3], COMPONENTS[4])]:
                g_current = parts[(family, duration, amplitude, "gru", phase, current)][p].mean(axis=1)
                g_archive = parts[(family, duration, amplitude, "gru", phase, archive)][p].mean(axis=1)
                t_current = parts[(family, duration, amplitude, "tcn", phase, current)][p].mean(axis=1)
                t_archive = parts[(family, duration, amplitude, "tcn", phase, archive)][p].mean(axis=1)
                for j, metric in enumerate(METRICS):
                    gc, ga, tc, ta = [v[:, j].std(ddof=1) for v in (g_current, g_archive, t_current, t_archive)]
                    tables["component_variability"].append({**base, "panel": panel, "route": route, "metric": metric,
                        "gru_current_seed_std": gc, "gru_archive_seed_std": ga,
                        "tcn_current_seed_std": tc, "tcn_archive_seed_std": ta,
                        "tcn_current_minus_archive_std": tc-ta, "tcn_minus_gru_current_std": tc-gc,
                        "tcn_current_std_greater_than_archive": bool(tc > ta),
                        "tcn_current_std_greater_than_gru": bool(tc > gc), "seeds": len(seeds)})
        if (family, duration, amplitude) != (ref["family"], ref["duration"], ref["amplitude"]) or phase not in ref["phases"]:
            continue
        j = METRICS.index(ref["metric"])
        effect = difference[:, :, :, j]
        per_fit = effect.mean(axis=2)
        lo, hi = interval(primary_weights @ effect.mean(axis=(0, 1)), spec["reference_interval_level"])
        cross_lo, cross_hi = interval(crossed_samples(effect, panel_indices, seed_indices, path_indices), spec["reference_interval_level"])
        tables["primary_contrasts"].append({**base, "metric": ref["metric"], "estimate": effect.mean(), "ci_low": lo, "ci_high": hi,
            "interval_level": spec["reference_interval_level"], "crossed_sensitivity_low": cross_lo, "crossed_sensitivity_high": cross_hi,
            "gru_interaction": gru[:, :, :, j].mean(), "tcn_interaction": tcn[:, :, :, j].mean(),
            "fit_min": per_fit.min(), "fit_max": per_fit.max(), "paths": n, "fixed_paired_fits": len(panels)*len(seeds)})
        for p, panel in enumerate(panels):
            pl, ph = interval(primary_weights @ effect[p].mean(axis=0), spec["reference_interval_level"])
            tables["primary_panel_contrasts"].append({**base, "panel": panel, "estimate": per_fit[p].mean(), "ci_low": pl, "ci_high": ph,
                "seed_std": per_fit[p].std(ddof=1), "seed_min": per_fit[p].min(), "seed_max": per_fit[p].max(),
                "gru_interaction": gru[p, :, :, j].mean(), "tcn_interaction": tcn[p, :, :, j].mean(), "interval_level": spec["reference_interval_level"]})
            tables["primary_leave_one_panel_out"].append({**base, "omitted_panel": panel, "estimate": np.delete(effect, p, axis=0).mean()})
            for s, model_seed in enumerate(seeds):
                tables["primary_individual_seeds"].append({**base, "panel": panel, "seed": model_seed, "estimate": per_fit[p, s],
                    "gru_interaction": gru[p, s, :, j].mean(), "tcn_interaction": tcn[p, s, :, j].mean()})
    checks = {"diagonal_component_max_error": diagonal_error, "decomposition_max_error": decomposition_error,
              "identical_entry_across_durations_max_error": entry_duplicate_error}
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}, checks


def performance_summaries(performance, crossed):
    results = {}
    for name, frame, context_keys in (("performance_summary", performance, ["context", "window", "calibrator"]),
                                     ("crossed_performance_summary", crossed, ["current_context", "archive_context", "window"])):
        keys = ["family", "duration", "amplitude", "architecture"]+context_keys+["phase", "world"]
        means = frame.groupby(keys, sort=True)[METRICS].mean().reset_index()
        means["fixed_fits"] = 15
        means["paths"] = 128
        results[name] = means
        panel_keys = ["panel"]+keys
        results[name.replace("summary", "panel_summary")] = frame.groupby(panel_keys, sort=True)[METRICS].mean().reset_index()
    return results


def validated_features(models, tables):
    features = pd.read_csv(PROBES / "model_features.csv")
    assert len(features) == 30 and not features.duplicated(["panel", "architecture", "seed"]).any()
    assert set(map(tuple, features[["panel", "architecture", "seed"]].to_numpy())) == set(map(tuple, models[["panel", "architecture", "seed"]].to_numpy()))
    merged = features.merge(models, on=["panel", "architecture", "seed"], suffixes=("", "_training"), validate="one_to_one")
    np.testing.assert_allclose(merged.validation_nll, merged.validation_nll_training, rtol=0, atol=1e-12)
    assert (merged.checkpoint_sha256 == merged.checkpoint_sha256_training).all()
    assert np.isfinite(features[FEATURES].to_numpy()).all(), "Required scatter features are undefined"
    individual = tables["individual_seed_effects"]
    individual = individual[(individual.family == "gaussian") & (individual.duration == 128)
        & (individual.amplitude == 3) & (individual.metric == "interval_score") & individual.phase.isin(["entry", "recovery64"])]
    current = tables["individual_seed_components"]
    current = current[(current.family == "gaussian") & (current.duration == 128)
        & (current.amplitude == 3) & (current.metric == "interval_score") & (current.component == "current_at_archive64")
        & current.phase.isin(["entry", "recovery64"])]
    keys = ["panel", "architecture", "seed", "phase"]
    joined = individual[keys+["estimate"]].rename(columns={"estimate": "diagonal_interaction"}).merge(
        current[keys+["estimate"]].rename(columns={"estimate": "current_at_archive64"}), on=keys, validate="one_to_one")
    joined = joined.merge(features, on=["panel", "architecture", "seed"], validate="many_to_one")
    assert len(joined) == 60
    return joined


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})


def plots(tables, protocol):
    style()
    ref = protocol["evaluation"]["primary_reference"]
    panels, seeds, _, _ = grid_ids(protocol)
    primary = tables["primary_contrasts"].set_index("phase")
    panel = tables["primary_panel_contrasts"]
    individual = tables["primary_individual_seeds"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, phase in zip(axes[:, 0], ref["phases"]):
        row = primary.loc[phase]
        ax.axvline(0, color="#999999", lw=.8)
        ax.hlines(0, row.ci_low, row.ci_high, color="#285a8a", lw=3)
        ax.scatter(row.estimate, 0, s=55, color="#285a8a")
        ax.hlines(-.22, row.crossed_sensitivity_low, row.crossed_sensitivity_high, color="#999999", lw=2)
        for p, panel_id in enumerate(panels, start=1):
            selected = panel[(panel.phase == phase) & (panel.panel == panel_id)].iloc[0]
            ax.hlines(p, selected.ci_low, selected.ci_high, color="#537681", lw=2)
            ax.scatter(selected.estimate, p, color="#537681", s=38)
            fit = individual[(individual.phase == phase) & (individual.panel == panel_id)]
            ax.scatter(fit.estimate, p+np.linspace(-.14, .14, len(fit)), color="#777777", marker="x", s=22)
        ax.set_yticks(range(4), ["All 15 fixed pairs", "Training panel 1", "Training panel 2", "Training panel 3"])
        ax.set_ylim(3.5, -.6)
        ax.set_title(f"{phase}: GRU J minus TCN J")
        ax.set_xlabel("Interval-score interaction contrast")
    values = tables["individual_seed_components"]
    for ax, phase in zip(axes[:, 1], ref["phases"]):
        selected = values[(values.family == ref["family"]) & (values.duration == ref["duration"])
            & (values.amplitude == ref["amplitude"]) & (values.metric == ref["metric"]) & (values.phase == phase)]
        for p, panel_id in enumerate(panels):
            for architecture, offset in [("gru", -.12), ("tcn", .12)]:
                for component, further, marker in [("current_at_archive64", -.04, "o"), ("archive_at_current16", .04, "x")]:
                    fit = selected[(selected.panel == panel_id) & (selected.architecture == architecture) & (selected.component == component)].set_index("seed").reindex(seeds)
                    ax.scatter(p+offset+further+np.linspace(-.018, .018, len(seeds)), fit.estimate,
                               color=COLORS[architecture], marker=marker, s=30, alpha=.8)
        ax.axhline(0, color="#999999", lw=.8)
        ax.set_xticks(range(len(panels)), [f"Panel {p}" for p in panels])
        ax.set_ylabel("Component: IS interaction units")
        ax.set_title(f"{phase}: all five seeds per architecture/panel")
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[a], label=a.upper()) for a in ARCHES]
    handles += [Line2D([], [], marker="o", ls="", color="#555555", label="Current at archive64"),
                Line2D([], [], marker="x", ls="", color="#555555", label="Archive at current16")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .025), frameon=False)
    fig.text(.02, .005, "Left: 97.5% fixed-fit paired-path intervals; grey sensitivity line also resamples panels and shared seeds.\n"
             "Right: each point is one seed mean; components are correlated and pathway-dependent, not mediated percentages.", fontsize=8)
    fig.suptitle("Independent training-data replication | Gaussian A3 D128")
    fig.tight_layout(rect=(0, .13, 1, .95))
    fig.savefig(OUT / "primary_replication_and_components.png")
    plt.close(fig)
    # Each architecture/panel retains all seeds. Shared duration entry is shown only once.
    grid = tables["panel_interactions"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for row, family in enumerate(protocol["test"]["families"]):
        for col, phase in enumerate(["entry", "recovery64"]):
            ax = axes[row, col]
            selected = grid[(grid.family == family) & (grid.phase == phase) & (grid.metric == "interval_score")]
            for p, panel_id in enumerate(panels):
                for architecture, shift in [("gru", -.06), ("tcn", .06)]:
                    fit = selected[(selected.panel == panel_id) & (selected.architecture == architecture)].sort_values("duration")
                    ax.plot(np.arange(len(fit))+shift+.018*(p-1), fit.estimate,
                            color=COLORS[architecture], marker=["o", "s", "^"][p], alpha=.8, lw=1)
            ax.axhline(0, color="#999999", lw=.8)
            ax.set_xticks([0, 1], ["32", "128"])
            ax.set_title(f"{family.replace('student_t5', 'Student-t5')} | {phase}")
            ax.set_xlabel("Pulse duration")
            ax.set_ylabel("Panel-mean diagonal J")
    handles = [Line2D([], [], color=COLORS[a], label=a.upper()) for a in ARCHES]
    handles += [Line2D([], [], color="#555555", marker=m, ls="", label=f"Panel {p}") for p, m in zip(panels, ["o", "s", "^"])]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, .025), frameon=False)
    fig.text(.04, .005, "Five fixed seed fits per panel; shared noise across durations. Entry32 is identical for D32/D128 by causality.\n"
             "Secondary means are descriptive; complete five metrics, all individual fits and intervals are stored in CSV.", fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(OUT / "complete_replication_grid.png")
    plt.close(fig)
    joined = tables["validation_feature_outcomes"]
    for phase in ref["phases"]:
        for target, title in [("diagonal_interaction", "Diagonal J"), ("current_at_archive64", "Current at archive64")]:
            fig, axes = plt.subplots(3, 3, figsize=(12.2, 10))
            for row, (feature, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
                for col, panel_id in enumerate(panels):
                    ax = axes[row, col]
                    selected = joined[(joined.phase == phase) & (joined.panel == panel_id)]
                    for architecture in ARCHES:
                        fit = selected[selected.architecture == architecture]
                        ax.scatter(fit[feature], fit[target], color=COLORS[architecture], s=33, alpha=.85)
                        for _, point in fit.iterrows():
                            ax.annotate(str(int(point.seed)-4100), (point[feature], point[target]),
                                        xytext=(3, 3), textcoords="offset points", fontsize=7, color=COLORS[architecture])
                    ax.axhline(0, color="#bbbbbb", lw=.7)
                    ax.set_xlabel(label, fontsize=8)
                    if col == 0:
                        ax.set_ylabel(f"{title} (IS units)")
                    if row == 0:
                        ax.set_title(f"Training panel {panel_id}")
            fig.legend(handles=[Line2D([], [], marker="o", ls="", color=COLORS[a], label=a.upper()) for a in ARCHES],
                       loc="lower center", ncol=2, bbox_to_anchor=(.5, .04), frameon=False)
            fig.suptitle(f"Exploratory validation-response scatter | {phase} | {title}")
            fig.text(.035, .01, "Every fitted model retained. Labels 1-5 identify seeds4101-4105. No regression, significance test, equivalence or prediction claim.\n"
                     "Different validation data in each panel; raw NLL comparisons are within-panel. Pulse feature always uses validation ENTRY.", fontsize=8)
            fig.tight_layout(rect=(0, .11, 1, .95))
            fig.savefig(OUT / f"validation_scatter_{phase}_{target}.png")
            plt.close(fig)


def md_table(frame, columns):
    lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for _, row in frame.iterrows():
        values = [f"{row[c]:+.5f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def write_report(tables, protocol, evaluation, checks):
    primary, panel = tables["primary_contrasts"], tables["primary_panel_contrasts"]
    ref = protocol["evaluation"]["primary_reference"]
    var = tables["component_variability"]
    var = var[(var.family == ref["family"]) & (var.duration == ref["duration"]) & (var.amplitude == ref["amplitude"])
              & (var.metric == ref["metric"]) & var.phase.isin(ref["phases"])]
    entry_var = var[var.phase == "entry"]
    joint_count = int((entry_var.tcn_current_std_greater_than_archive & entry_var.tcn_current_std_greater_than_gru).sum())
    statements = []
    for _, row in primary.iterrows():
        each = panel[panel.phase == row.phase]
        loo = tables["primary_leave_one_panel_out"]
        loo = loo[loo.phase == row.phase]
        signs = "、".join(f"panel{int(r.panel)}={r.estimate:+.5f}" for _, r in each.iterrows())
        statements.append(f"- `{row.phase}`：GRU−TCN = {row.estimate:+.5f}，固定15对拟合的97.5%路径区间 [{row.ci_low:+.5f}, {row.ci_high:+.5f}]；crossed敏感性区间 [{row.crossed_sensitivity_low:+.5f}, {row.crossed_sensitivity_high:+.5f}]。逐panel均值：{signs}；删一panel范围 [{loo.estimate.min():+.5f}, {loo.estimate.max():+.5f}]。")
    components = tables["component_summary"]
    components = components[(components.family == ref["family"]) & (components.duration == ref["duration"])
        & (components.amplitude == ref["amplitude"]) & (components.metric == ref["metric"]) & components.phase.isin(ref["phases"])]
    comp_pivot = components.pivot(index=["architecture", "phase"], columns="component", values="estimate").reset_index()
    native = tables["performance_summary"]
    native = native[(native.family == ref["family"]) & (native.duration == ref["duration"]) & (native.amplitude == ref["amplitude"])
        & (native.world == "event") & (native.calibrator == "native_gaussian") & native.phase.isin(ref["phases"])]
    text = f"""# Stage 3：独立训练数据重复与验证响应诊断

三个新训练/验证面板、每面板五个配对初始化/采样seed、两个架构的全部30个拟合，以及新的4个测试case都已评估。当前读数：Gaussian entry 中，TCN current分量的seed标准差同时大于archive及GRU current的panel×分解路线组合为 **{joint_count}/6**。这是描述性重复读数，不能把六个相关组合当六次独立检验。架构contrast与逐panel差异见下表；本报告不预设其跨panel方向一致。

日期：2026-10-11。Stage3是受Stage2启发、在新训练及新测试访问前冻结的重复研究；没有新算法选择、checkpoint修改或模型排除。原训练预处理 `(raw/3, log1p(abs(raw)))`、零均值Gaussian尺度head、上下文16/64混合训练、优化与early-stopping规则均保持不变。唯一训练协议变化是独立train/validation数据seeds及输出目录。这里比较两个具体训练程序，参数量和优化历程仍不完全相同。

## 主要读数与统计对象

{chr(10).join(statements)}

{md_table(primary, ['phase', 'gru_interaction', 'tcn_interaction', 'estimate', 'ci_low', 'ci_high', 'crossed_sensitivity_low', 'crossed_sensitivity_high'])}

J为已做event−matched-control的diagonal context×window interaction；主要contrast取J_GRU−J_TCN。正负不能直接当绝对区间分数排名或长记忆损害。主要case固定Gaussian D128 A3、interval score、entry32/recovery64两个phase。

4000次bootstrap先对 **3×5=15对固定拟合** 求平均，再共同重抽128条独立paired paths；两个phase使用97.5% marginal intervals作Bonferroni近似调整。这些区间条件于保存的模型，不是初始化或训练数据总体推断。逐panel的97.5%区间仅描述，不是对六个panel×phase端点的联合校正。Crossed敏感性另重抽3个训练panel、5个seed索引和path；同一seed draw用于所有抽到的panel，保留共享初始化/采样设计。三个panel和五seed不足以给稳定总体结论。两个架构共有path，duration32/128共有family基础噪声，不能把30fits×128paths当3840个独立样本。

![主要重复与通道分量](../outputs/stage3_analysis/primary_replication_and_components.png)

## 三个训练panel与全部初始化

{md_table(panel, ['phase', 'panel', 'gru_interaction', 'tcn_interaction', 'estimate', 'seed_std', 'seed_min', 'seed_max'])}

[全部15对individual contrasts](../outputs/stage3_analysis/primary_individual_seeds.csv)、[全部case/phase/metric individual effects](../outputs/stage3_analysis/individual_seed_effects.csv)与[删一训练panel](../outputs/stage3_analysis/primary_leave_one_panel_out.csv)保留全部seed，未以验证或测试排名删除模型。Seed标准差是每panel五个固定seed均值的样本标准差，不能转换为方差贡献百分比。

四个case为Gaussian/t5×D32/D128×A3。Entry只看共同最初32步；两duration的entry J和全部components逐path一致，最大误差为 `{checks['identical_entry_across_durations_max_error']:.3g}`。这来自因果前缀与共同噪声，不能算两次duration鲁棒性重复。Recovery评分时刻随duration改变。全四phase、五指标均保留：[secondary interactions](../outputs/stage3_analysis/secondary_interactions.csv)、[architecture contrasts](../outputs/stage3_analysis/secondary_architecture_contrasts.csv)。95% secondary区间仅描述、未作多重校正。

![完整重复网格](../outputs/stage3_analysis/complete_replication_grid.png)

## Current与archive的两条路线

`total = current_at_archive64 + archive_at_current16 = current_at_archive16 + archive_at_current64`。分量是crossed factorial的精确计算差分，不是唯一因果中介比例；两路线可以不同，相关分量CI端点不可相加。

{md_table(comp_pivot, ['architecture', 'phase', 'total', 'current_at_archive64', 'archive_at_current16', 'current_at_archive16', 'archive_at_current64', 'third_order'])}

{md_table(var, ['phase', 'panel', 'route', 'gru_current_seed_std', 'gru_archive_seed_std', 'tcn_current_seed_std', 'tcn_archive_seed_std'])}

[完整逐panel variability](../outputs/stage3_analysis/component_variability.csv)、[全部components汇总](../outputs/stage3_analysis/component_summary.csv)、[全部individual components](../outputs/stage3_analysis/individual_seed_components.csv)包括t5、D32与另外三指标/phase。不预设TCN必更不稳定，current与archive也不必同向。固定archive时current×window的log-width交互为零是代数性质；非线性interval score的交互可以非零，不能把恒等式或有限遗忘界包装为新定理。

## 绝对表现与覆盖代价

主参考case的原生Gaussian区间、event world描述均值如下；coverage为比例，IS越小越好。它与J的差分问题不同。

{md_table(native, ['architecture', 'context', 'phase', 'interval_score', 'coverage', 'width'])}

[所有可部署配置的event/control绝对表现](../outputs/stage3_analysis/performance_summary.csv)完整保留native/rolling32/128；[逐panel绝对表现](../outputs/stage3_analysis/performance_panel_summary.csv)和[crossed诊断配置](../outputs/stage3_analysis/crossed_performance_summary.csv)分别保存。没有对绝对胜负作新显著性检验；不能只用coverage提高或某个phase最优宣称新方法改善。

## 仅验证集响应散点

独立冻结的验证诊断使用各panel前256个已有validation target pairs的梯度/older-block置换，并用64条额外Gaussian pulse验证路径构造尺度响应。模型输入不含事件标签，所有checkpoint保持固定。图仅画mean validation NLL、older-gradient fraction、**entry** pulse signed context64−16 log-scale响应与测试J/current@archive64的关系；后者两个测试phase都展示同一entry feature，不按结果选更相关feature。

[完整feature与outcome对照](../outputs/stage3_analysis/validation_feature_outcomes.csv)保留全部30拟合。散点按训练panel分列、架构着色，编号1–5对应seed4101–4105。没有回归、相关性显著性、meta-model拟合、equivalence或预测能力声明；3个panel与共享5seed不足以从散点确立一般机制。每panel的NLL和梯度来自不同validation观测，不能把panel均值差直接解释为模型训练质量；图按panel分面，没有pooled correlation。梯度局部，置换可能分布外，pulse属于人为模型响应probe。其与outcome关联仍exploratory。

![Entry current散点](../outputs/stage3_analysis/validation_scatter_entry_current_at_archive64.png)

![Recovery current散点](../outputs/stage3_analysis/validation_scatter_recovery64_current_at_archive64.png)

J的[entry散点](../outputs/stage3_analysis/validation_scatter_entry_diagonal_interaction.png)与[recovery散点](../outputs/stage3_analysis/validation_scatter_recovery64_diagonal_interaction.png)也完整保存。“相近NLL”未证明实质等效；若要claim等效须先规定margin与独立路径等效性评估。若要claim超出NLL的预测信息，应另冻低维预测模型与新training-panel holdout，再比较NLL-only基线，当前阶段未做此检验。

## 先例与贡献边界

[D’Amour et al., JMLR2022](https://jmlr.org/papers/v23/20-1335.html)已研究域内表现相近的随机初始化模型在stress behavior/校准上不同；[HARNet §5.1](https://arxiv.org/html/2205.07719)已有神经波动模型初始化稳定性分析。因此本研究不声称首次发现seed不稳定或validation不足。NCC神经控制、AdaptNC反事实replay、RetroAdj残差重算、SAOCP多尺度专家、PID scorecasting和HopCPT状态检索均是直接先例。完整区别与全文访问限制见[贡献审查](contribution_review.md)。

目前可争取的贡献是具体的fixed-neural-scale current/archive通道机制、独立训练数据重复与验证诊断的边界；是否形成A+级结论取决于证据和论证，不能用代码规模、次数或单个条件区间代替。当前没有提出或验证有优势的新金融部署算法。已看过的ECB测试period仍为探索数据，不追溯改称未见真实holdout。

## 实现检查和来源

{evaluation['negative_control_check_count']}个crossed遗忘检查与{evaluation['diagonal_check_count']}个diagonal复现通过；遗忘界之后半宽最大差 `{evaluation['maximum_crossed_half_width_error_after_bound']:.3g}`。两path分解与主J逐path复现最大误差分别 `{checks['decomposition_max_error']:.3g}` / `{checks['diagonal_component_max_error']:.3g}`。这些是数学/实现负控制，不是覆盖保证或真实市场192天恢复保证。固定观测上下文和rolling归档的界只在event/control退出后观测完全相同时成立。

分析脚本在读取本阶段test输出前提交；验证冻结protocol、训练面板/checkpoint、eval/probe sources及全部artifact哈希，并拒绝已有manifest的覆盖。Exact grid、missing/duplicate/off-grid rows、paired index与两bootstrap独立toy循环检查都通过。全部输出/report/source哈希写入[manifest](../outputs/stage3_analysis/manifest.json)。

```sh
.venv/bin/python scripts/analyze_stage3.py --self-check
.venv/bin/python scripts/analyze_stage3.py
```
"""
    REPORT.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check()))
        return
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Stage3 analysis manifest exists; refusing overwrite")
    source_commit = committed(__file__)
    committed(PROTOCOL)
    source_sha = sha(__file__)
    protocol, training, evaluation, probes, model_rows = verified_inputs()
    started = time.time()
    interactions, components, performance, crossed, frame_checks = load_frames(protocol, evaluation)
    tables, checks = analyze(interactions, components, protocol)
    checks.update(frame_checks)
    tables.update(performance_summaries(performance, crossed))
    tables["validation_feature_outcomes"] = validated_features(model_rows, tables)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    plots(tables, protocol)
    write_report(tables, protocol, evaluation, checks)
    assert sha(__file__) == source_sha
    # Verify every dependency once more; no source, checkpoint or input may drift.
    verified_inputs()
    manifest = {"status": "complete", "analysis_source_sha256": source_sha,
        "analysis_source_commit_before_test_access": source_commit,
        "protocol_sha256": sha(PROTOCOL), "training_manifest_sha256": sha(TRAIN / "manifest.json"),
        "evaluation_manifest_sha256": sha(IN / "manifest.json"), "probe_manifest_sha256": sha(PROBES / "manifest.json"),
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [IN / name for name in evaluation["artifacts"]]
                         + [PROBES / name for name in probes["artifacts"]]},
        "dependency_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "scripts/evaluate_stage3.py",
            ROOT / "scripts/train_stage3_repeats.py", ROOT / "scripts/stage3_response_probes.py",
            ROOT / "configs/stage3_response_probes.json"]},
        "primary_bootstrap": {"seed": protocol["analysis"]["bootstrap_seed"], "replicates": 4000, "level": .975,
            "unit": "common paired path, averaging all15 fixed model pairs before resampling",
            "adjustment": "two primary reference phases Bonferroni marginal intervals; percentile bootstrap approximation"},
        "crossed_sensitivity": {"seed": protocol["analysis"]["bootstrap_seed"]+1,
            "unit": "training-panel, shared seed index across every sampled panel, and common path; sensitivity only"},
        "descriptive_bootstrap": {"seed": protocol["analysis"]["bootstrap_seed"]+2, "replicates": 4000, "level": .95,
            "scope": "unadjusted descriptive intervals conditional on fitted models"},
        "validation_scatter": {"features": FEATURES, "fitted_meta_model": False, "association_scope": "exploratory",
            "rows": 60, "all30_models_retained": True},
        "checks": {**tiny_self_check(), **checks}, "row_counts": {name: len(frame) for name, frame in tables.items()},
        "artifact_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        "report_sha256": sha(REPORT), "wall_seconds": time.time()-started}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(tables["primary_contrasts"].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
