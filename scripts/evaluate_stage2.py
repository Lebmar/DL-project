"""Evaluate the frozen matched panel on independent paired volatility pulses.

No fitting, checkpoint selection or parameter search occurs here. All 30 cases
are retained. Event indices are used only to generate/evaluate paths and avoid
recomputing exactly identical contexts; they never enter a predictor.
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
from scipy.stats import norm, t as student

from stage2_baselines import ScaleModel
from stage2_models import load_checkpoint
from train_stage2 import generate_paths, array_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage2_evaluation"
METRICS = ["interval_score", "coverage", "width", "coverage_deficit",
           "excess_width_to_oracle"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs():
    protocol_path = ROOT / "configs/stage2_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((ROOT / "outputs/stage2_training/manifest.json").read_text())
    controls = json.loads((ROOT / "outputs/stage2_controls/models.json").read_text())
    assert protocol["frozen"] and manifest["status"] == "complete"
    assert manifest["matching_minibatches_verified"]
    assert sha(protocol_path) == manifest["protocol_sha256"] == controls["provenance"]["protocol_sha256"]
    for name, value in manifest["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    assert sha(ROOT / "scripts/stage2_baselines.py") == controls["provenance"]["baseline_source_sha256"]
    for entry in manifest["models"]:
        assert sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
    assert len(manifest["models"]) == 10
    assert {(m["architecture"], m["model_seed"]) for m in manifest["models"]} == {
        (a, s) for a in protocol["training_config"]["architectures"]
        for s in protocol["training_config"]["model_seeds"]}
    return protocol, manifest, controls


def predict_scales(model, y, context, start=0, stop=None, batch_size=2048):
    """Reset model state for every target; input excludes current y[t].

    Initial missing context is padded with zero observations; it is long gone
    by the scored onset at t384. Padding is not a future-dependent transform.
    """
    y = np.asarray(y, dtype=np.float64)
    stop = y.shape[1] if stop is None else stop
    n, width = len(y), stop-start
    padded = np.pad(y, ((0, 0), (context, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, context, axis=1)[:, start:stop]
    result = np.empty(n*width, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, n*width, batch_size):
            flat = np.arange(offset, min(offset+batch_size, n*width))
            x = torch.tensor(windows[flat//width, flat % width], dtype=torch.float32)
            result[flat] = model(x).exp().numpy()
    return result.reshape(n, width)


def rolling_half_width(y, sigma, window, alpha, warmup):
    """Finite-rank quantile of previously issued scores only, vectorized."""
    scores = np.abs(y)/sigma
    warm = np.broadcast_to(np.asarray(warmup)[None, :], (len(y), len(warmup)))
    assert warm.shape[1] >= window
    all_scores = np.concatenate((warm, scores), axis=1)
    # Window indexed t starts at t-W and ends at t-1, never including score[t].
    views = np.lib.stride_tricks.sliding_window_view(all_scores, window, axis=1)
    views = views[:, warm.shape[1]-window:warm.shape[1]-window+y.shape[1]]
    rank = min(window, int(np.ceil((window+1)*(1-alpha))))-1
    q = np.partition(views, rank, axis=-1)[:, :, rank]
    return sigma*q


def generate_base(family, config, family_index):
    """One independent family stream; same noise shared across its 10 pulses."""
    rng = np.random.default_rng(config["path_seed"]+1000*family_index)
    shape = (config["paths"], config["length"])
    sigma = np.ones(shape)
    if family == "student_t5":
        df = config["student_t_df"]
        z = rng.standard_t(df, shape)*np.sqrt((df-2)/df)
    else:
        if family == "sv_gaussian":
            phi, eta = config["sv_phi"], config["sv_eta"]
            variance = eta**2/(1-phi**2)
            mean = -.5*variance
            v = np.empty(shape)
            v[:, 0] = rng.normal(mean, np.sqrt(variance), shape[0])
            innovations = rng.normal(0, eta, shape)
            for j in range(1, shape[1]):
                v[:, j] = mean+phi*(v[:, j-1]-mean)+innovations[:, j]
            sigma = np.exp(v/2)
        z = rng.normal(size=shape)
    return z*sigma, sigma


def phases(onset, duration, length):
    exit_time = onset+duration
    return {"entry": (onset, onset+min(duration, 32)),
            "recovery64": (exit_time, min(exit_time+64, length)),
            "recovery192": (exit_time, min(exit_time+192, length)),
            "late": (exit_time+192, min(exit_time+256, length))}


def metric_arrays(y, half_width, true_sigma, family, config):
    alpha = config["alpha"]
    if family == "student_t5":
        df = config["student_t_df"]
        unit = np.sqrt((df-2)/df)
        probability = 2*student.cdf(half_width/(true_sigma*unit), df)-1
        oracle_q = unit*student.ppf(1-alpha/2, df)
    else:
        probability = 2*norm.cdf(half_width/true_sigma)-1
        oracle_q = norm.ppf(1-alpha/2)
    width = 2*half_width
    return {"interval_score": width+(2/alpha)*np.maximum(np.abs(y)-half_width, 0),
            "coverage": (np.abs(y) <= half_width).astype(float),
            "width": width,
            "coverage_deficit": np.maximum(1-alpha-probability, 0),
            "excess_width_to_oracle": np.maximum(width-2*oracle_q*true_sigma, 0)}


def tiny_self_check():
    """Index/quantile checks independent of project paths or fitted weights."""
    from stage2_models import make_model
    torch.manual_seed(78126)
    rng = np.random.default_rng(78126)
    y = rng.normal(size=(3, 260))
    changed = y.copy()
    changed[:, 130:] += 25
    warm = rng.uniform(.1, 3, 128)
    sigma = np.ones_like(y)
    for window in (32, 128):
        actual = rolling_half_width(y, sigma, window, .1, warm)
        rank = min(window, int(np.ceil((window+1)*.9)))-1
        expected = []
        history = np.concatenate((np.broadcast_to(warm, (3, 128)), np.abs(y)), axis=1)
        for j in range(y.shape[1]):
            expected.append(np.sort(history[:, 128+j-window:128+j], axis=1)[:, rank])
        np.testing.assert_array_equal(actual, np.asarray(expected).T)
        other = rolling_half_width(changed, sigma, window, .1, warm)
        np.testing.assert_array_equal(actual[:, :131], other[:, :131])
    for arch in ("gru", "tcn"):
        model = make_model(arch)
        for context in (16, 64):
            issued = predict_scales(model, y, context)
            altered = predict_scales(model, changed, context)
            np.testing.assert_allclose(issued[:, :131], altered[:, :131], rtol=0, atol=1e-6)
            suffix = predict_scales(model, y, context, 100, 180)
            np.testing.assert_allclose(issued[:, 100:180], suffix, rtol=0, atol=1e-6)
    return {"passed": True, "rolling_matches_scalar_previous_score_quantile": True,
            "future_and_current_outcome_excluded_from_scale_and_q": True,
            "chunked_contexts_match_full_prediction": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.self_check:
        print(json.dumps(tiny_self_check()))
        return
    protocol, training, controls = verified_inputs()
    relative_source = str(Path(__file__).resolve().relative_to(ROOT))
    committed = subprocess.run(["git", "show", f"HEAD:{relative_source}"], cwd=ROOT,
                               check=True, capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest() == sha(__file__), "commit evaluator before test access"
    code_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                 capture_output=True, text=True).stdout.strip()
    if (OUT/"manifest.json").exists():
        raise FileExistsError("An evaluation manifest already exists; refusing silent test rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sources = {name: sha(ROOT/"scripts"/name) for name in
               ["evaluate_stage2.py", "stage2_baselines.py", "stage2_models.py", "train_stage2.py"]}
    manifest = {"status": "evaluating", "protocol_sha256": training["protocol_sha256"],
                "code_commit_before_test_access": code_commit,
                "sources": sources, "training_manifest_sha256": sha(ROOT/"outputs/stage2_training/manifest.json"),
                "controls_sha256": sha(ROOT/"outputs/stage2_controls/models.json"),
                "test": protocol["test"], "family_seed_rule": "path_seed + 1000 * zero-based family index",
                "test_selection": "none; all30 frozen cases retained", "self_check": tiny_self_check(),
                "cache": "only identical model contexts reused; affected targets [onset+1,exit+L)",
                "SV_metric_conditioning": "oracle latent state conditional, not observed-history conditional"}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    tc = protocol["training_config"]
    val_y = generate_paths(tc["validation_data"], tc["generator"])
    assert array_sha256(val_y) == training["data_sha256"]["validation"]
    models, warmups = {}, {}
    for entry in training["models"]:
        arch, seed = entry["architecture"], entry["model_seed"]
        model, _ = load_checkpoint(ROOT/entry["checkpoint"])
        for context in protocol["test"]["contexts"]:
            key = (arch, seed, context)
            models[key] = model
            val_sigma = predict_scales(model, val_y[:1], context, val_y.shape[1]-128)
            warmups[key] = np.abs(val_y[0, -128:])/val_sigma[0]
    baseline_models = {name: ScaleModel(**value) for name, value in controls["models"].items()}
    summaries, interaction_rows, negative_controls, curve_sums = [], [], [], {}
    test_hashes = {}
    config = protocol["test"]
    onset, length = config["onset"], config["length"]
    alpha, native_q = config["alpha"], norm.ppf(1-config["alpha"]/2)
    for family_index, family in enumerate(config["families"]):
        control_y, control_true = generate_base(family, config, family_index)
        test_hashes[family] = {"observations": array_sha256(control_y), "true_scale": array_sha256(control_true)}
        control_scales = {key: predict_scales(model, control_y, key[2]) for key, model in models.items()}
        for name, model in baseline_models.items():
            key = ("ewma", 0, int(name[4:])) if name.startswith("ewma") else (name, 0, 0)
            control_scales[key] = model.scale_paths(control_y)
            warmups[key] = controls["calibration_specs"][name]["warmup_scores"]
        control_metrics, control_widths = {}, {}
        for key, sigma in control_scales.items():
            for window in (0, *config["rolling_windows"]):
                half = sigma*native_q if window == 0 else rolling_half_width(control_y, sigma, window, alpha, warmups[key])
                tag = (*key, window)
                control_widths[tag] = half
                control_metrics[tag] = metric_arrays(control_y, half, control_true, family, config)
        for duration in config["durations"]:
            exit_time = onset+duration
            for amplitude in config["amplitudes"]:
                y, true = control_y.copy(), control_true.copy()
                y[:, onset:exit_time] *= amplitude
                true[:, onset:exit_time] *= amplitude
                slices = phases(onset, duration, length)
                case_means = {}
                reference = family in ("gaussian", "student_t5") and duration == 128 and amplitude == 3
                reference_arrays = {"y_event": y, "y_control": control_y, "true_sigma_event": true,
                                    "true_sigma_control": control_true, "onset": onset, "exit": exit_time}
                for key, original_sigma in control_scales.items():
                    arch, seed, context = key
                    if arch in ("gru", "tcn"):
                        sigma = original_sigma.copy()
                        sigma[:, onset+1:exit_time+context] = predict_scales(models[key], y, context, onset+1, exit_time+context)
                        # Independently recompute a prefix and the boundary: cache does not hide an indexing error.
                        check_targets = [onset, exit_time+context]
                        errors = [float(np.max(np.abs(predict_scales(models[key], y[:2], context, j, j+1)
                                                         -sigma[:2, j:j+1]))) for j in check_targets]
                        assert max(errors) <= 2e-6
                    else:
                        name = f"ewma{context}" if arch == "ewma" else arch
                        sigma = baseline_models[name].scale_paths(y)
                    if reference and seed == 4101:
                        reference_arrays[f"{arch}_c{context}_event"] = sigma
                        reference_arrays[f"{arch}_c{context}_control"] = original_sigma
                    for window in (0, *config["rolling_windows"]):
                        tag = (*key, window)
                        half = sigma*native_q if window == 0 else rolling_half_width(y, sigma, window, alpha, warmups[key])
                        original_half = control_widths[tag]
                        np.testing.assert_allclose(half[:, :onset+1], original_half[:, :onset+1], rtol=0, atol=2e-6)
                        if arch in ("gru", "tcn"):
                            late_error = float(np.max(np.abs(half[:, exit_time+192:]-original_half[:, exit_time+192:])))
                            assert late_error <= 2e-6
                            negative_controls.append({"family": family, "duration": duration, "amplitude": amplitude,
                                                      "architecture": arch, "seed": seed, "context": context,
                                                      "window": window, "late_max_abs_width_error": 2*late_error})
                        event_metrics = metric_arrays(y, half, true, family, config)
                        for phase, (lo, hi) in slices.items():
                            phase_means = {}
                            for world, arrays in (("event", event_metrics), ("control", control_metrics[tag])):
                                per_path = {m: arrays[m][:, lo:hi].mean(axis=1) for m in METRICS}
                                phase_means[world] = per_path
                                summaries.append({"family": family, "duration": duration, "amplitude": amplitude,
                                                  "architecture": arch, "seed": seed, "context": context, "window": window,
                                                  "phase": phase, "world": world, "paths": len(y), "steps": hi-lo,
                                                  **{m: float(v.mean()) for m, v in per_path.items()}})
                            case_means[(arch, seed, context, window, phase)] = {
                                m: phase_means["event"][m]-phase_means["control"][m] for m in METRICS}
                        if reference and arch in ("gru", "tcn"):
                            for world, arrays in (("event", event_metrics), ("control", control_metrics[tag])):
                                curve_key = (family, arch, context, window, world)
                                curves = np.stack([arrays[m].mean(axis=0) for m in METRICS])
                                curve_sums[curve_key] = curve_sums.get(curve_key, np.zeros_like(curves))+curves/5
                for arch in ("gru", "tcn", "ewma"):
                    seeds = tc["model_seeds"] if arch != "ewma" else [0]
                    for seed in seeds:
                        for phase in slices:
                            keys = [(arch, seed, 64, 128, phase), (arch, seed, 16, 128, phase),
                                    (arch, seed, 64, 32, phase), (arch, seed, 16, 32, phase)]
                            contrasts = {m: case_means[keys[0]][m]-case_means[keys[1]][m]
                                          -case_means[keys[2]][m]+case_means[keys[3]][m] for m in METRICS}
                            for p in range(len(y)):
                                interaction_rows.append({"family": family, "duration": duration, "amplitude": amplitude,
                                                         "architecture": arch, "seed": seed, "phase": phase, "path": p,
                                                         **{m: float(v[p]) for m, v in contrasts.items()}})
                if reference:
                    np.savez_compressed(OUT/f"reference_{family}.npz", **reference_arrays)
                print(json.dumps({"family": family, "duration": duration, "amplitude": amplitude,
                                  "elapsed_seconds": round(time.time()-started, 2)}), flush=True)
    pd.DataFrame(summaries).to_csv(OUT/"performance_by_seed.csv", index=False)
    pd.DataFrame(interaction_rows).to_csv(OUT/"paired_interactions.csv.gz", index=False,
                                          compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(negative_controls).to_csv(OUT/"negative_controls.csv", index=False)
    curve_rows = []
    for (family, arch, context, window, world), curves in curve_sums.items():
        for j in range(onset-128, onset+128+256):
            curve_rows.append({"family": family, "architecture": arch, "context": context,
                               "window": window, "world": world, "time": j,
                               **{m: float(curves[i, j]) for i, m in enumerate(METRICS)}})
    pd.DataFrame(curve_rows).to_csv(OUT/"reference_curves.csv.gz", index=False,
                                   compression={"method": "gzip", "mtime": 0})
    assert sources == {name: sha(ROOT/"scripts"/name) for name in sources}, "source changed during evaluation"
    manifest.update(status="complete", cases=30, test_arrays_sha256=test_hashes,
                    performance_rows=len(summaries), interaction_rows=len(interaction_rows),
                    negative_control_checks=len(negative_controls),
                    maximum_negative_control_width_error=float(pd.DataFrame(negative_controls).late_max_abs_width_error.max()),
                    elapsed_seconds=time.time()-started,
                    artifacts={p.name: sha(p) for p in OUT.iterdir() if p.name != "manifest.json"})
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
