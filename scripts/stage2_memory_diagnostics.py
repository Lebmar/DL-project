"""Frozen validation-only diagnostics of fitted Stage 2 neural scale memory.

Uses the first 256 of the saved 8192 validation target pairs for all ten models.
Raw-input gradients are local sensitivity, and older-history permutation is a
model-dependence diagnostic that can create contexts outside the training
distribution. Neither establishes real-world causal effects or a deployable
method. No test data or test results are read, and no model is updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stage2_models import load_checkpoint
from train_stage2 import array_sha256, context_tensor, generate_paths


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "outputs" / "stage2_training"
OUT = ROOT / "outputs" / "stage2_memory"
DIAGNOSTIC_CONFIG = {
    "source": "saved Stage 2 validation_target_pairs.npz, not a new sample",
    "expected_validation_pairs": 8192,
    "target_pair_start": 0,
    "target_pair_count": 256,
    "contexts": [16, 64],
    "gradient_context": 64,
    "recent_observations_unchanged": 16,
    "older_block_permutation_seed": 920110,
    "permutation_unit": "entire older48 block between the 256 target samples",
    "absolute_delta_quantile": 0.95,
    "quantile_method": "linear",
    "device": "cpu",
    "torch_threads": 4,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"{description} mismatch: {actual!r} != {expected!r}")


def git_code_commit():
    """Require this diagnostic code to have been committed before execution."""
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative],
                            cwd=ROOT, check=False)
    if result.returncode != 0:
        raise ValueError("commit this diagnostic script before running it")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT, text=True).strip()
    if not commit:
        raise ValueError("diagnostic script has no committed sampling configuration")
    return commit


def check_training_inputs():
    manifest_path = TRAINING / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    require_equal(manifest.get("status"), "complete", "training status")
    require_equal(manifest.get("matching_minibatches_verified"), True,
                  "matched-minibatch verification")
    config = manifest["training_config"]
    require_equal(config["architectures"], ["gru", "tcn"], "architectures")
    require_equal(config["contexts"], DIAGNOSTIC_CONFIG["contexts"], "contexts")
    seeds = config["model_seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("the frozen panel must contain five distinct model seeds")
    expected_panel = {(architecture, seed)
                      for architecture in config["architectures"] for seed in seeds}
    observed_panel = [(item["architecture"], item["model_seed"])
                      for item in manifest["models"]]
    if len(observed_panel) != 10 or set(observed_panel) != expected_panel:
        raise ValueError("training manifest does not contain exactly ten matched models")
    require_equal(manifest.get("completed_models"), 10, "completed model count")
    for name, digest in manifest["source_sha256"].items():
        require_equal(file_sha256(ROOT / "scripts" / name), digest,
                      f"training source {name}")
    require_equal(file_sha256(TRAINING / "frozen_protocol.json"),
                  manifest["protocol_sha256"], "copied frozen protocol")
    require_equal(file_sha256(Path(manifest["protocol_source"])),
                  manifest["protocol_sha256"], "original frozen protocol")
    return manifest, hashlib.sha256(manifest_raw).hexdigest()


def gradient_profile(model, observed_context):
    """Average |d log(sigma_next) / d raw y_lag|, with lag1 most recent."""
    inputs = observed_context.detach().clone().requires_grad_(True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    log_scale = model(inputs)
    gradients, = torch.autograd.grad(log_scale.sum(), inputs)
    if not torch.isfinite(gradients).all() or not torch.isfinite(log_scale).all():
        raise FloatingPointError("nonfinite local gradient or predicted log scale")
    # Batch elements do not interact: neither architecture has batch statistics.
    lag_profile = gradients.abs().mean(dim=0).flip(0).detach().cpu().numpy()
    total_mass = float(lag_profile.sum())
    older_mass = float(lag_profile[DIAGNOSTIC_CONFIG["recent_observations_unchanged"]:].sum())
    older_fraction = older_mass / total_mass if total_mass > 0 else float("nan")
    return log_scale.detach().cpu().numpy(), lag_profile, {
        "mean_total_absolute_gradient_mass": total_mass,
        "mean_older_than16_absolute_gradient_mass": older_mass,
        "older_than16_gradient_mass_fraction": older_fraction,
        "all_gradients_zero": total_mass == 0,
        "targets_with_all_gradients_zero_fraction":
            float((gradients.abs().sum(dim=1) == 0).float().mean()),
    }


def q95_absolute(values):
    return float(np.quantile(np.abs(values),
                             DIAGNOSTIC_CONFIG["absolute_delta_quantile"],
                             method=DIAGNOSTIC_CONFIG["quantile_method"]))


def diagnose_model(model, context16, context64, permutation):
    log64, profile, summary = gradient_profile(model, context64)
    older_width = DIAGNOSTIC_CONFIG["gradient_context"] - DIAGNOSTIC_CONFIG["recent_observations_unchanged"]
    permuted = context64.detach().clone()
    permuted[:, :older_width] = context64[permutation, :older_width]
    if not torch.equal(permuted[:, older_width:], context64[:, older_width:]):
        raise AssertionError("older-block permutation changed the recent16 inputs")
    with torch.no_grad():
        log_permuted = model(permuted).cpu().numpy()
        log16 = model(context16).cpu().numpy()
    if not np.isfinite(log_permuted).all() or not np.isfinite(log16).all():
        raise FloatingPointError("nonfinite context diagnostic output")
    delta_permutation = log_permuted - log64
    delta_context_log = log64 - log16
    scale64, scale16 = np.exp(log64), np.exp(log16)
    delta_context_scale = scale64 - scale16
    lower, upper = model.log_scale_bounds
    summary.update({
        "older_permutation_mean_abs_delta_log_scale": float(np.abs(delta_permutation).mean()),
        "older_permutation_q95_abs_delta_log_scale": q95_absolute(delta_permutation),
        "older_permutation_mean_signed_delta_log_scale": float(delta_permutation.mean()),
        "context64_minus16_mean_signed_delta_log_scale": float(delta_context_log.mean()),
        "context64_minus16_mean_abs_delta_log_scale": float(np.abs(delta_context_log).mean()),
        "context64_minus16_q95_abs_delta_log_scale": q95_absolute(delta_context_log),
        "context64_minus16_mean_signed_delta_scale": float(delta_context_scale.mean()),
        "context64_minus16_mean_abs_delta_scale": float(np.abs(delta_context_scale).mean()),
        "context64_minus16_q95_abs_delta_scale": q95_absolute(delta_context_scale),
        "context64_log_scale_at_clamp_fraction":
            float(((log64 <= lower) | (log64 >= upper)).mean()),
    })
    targets = pd.DataFrame({
        "log_scale_context16": log16,
        "log_scale_context64": log64,
        "scale_context16": scale16,
        "scale_context64": scale64,
        "log_scale_older48_permuted": log_permuted,
        "older_permutation_delta_log_scale": delta_permutation,
        "context64_minus16_delta_log_scale": delta_context_log,
        "context64_minus16_delta_scale": delta_context_scale,
    })
    return profile, summary, targets


def make_plot(curves, summary, model_seeds):
    colors = {"gru": "#2669a5", "tcn": "#b85b36"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for architecture in ["gru", "tcn"]:
        part = curves[curves.architecture.eq(architecture)]
        for seed in model_seeds:
            line = part[part.model_seed.eq(seed)].sort_values("lag")
            axes[0, 0].plot(line.lag, line.mean_abs_raw_input_gradient,
                            color=colors[architecture], alpha=.22, linewidth=.8)
        mean = part.groupby("lag").mean_abs_raw_input_gradient.mean()
        axes[0, 0].plot(mean.index, mean.to_numpy(), color=colors[architecture],
                        linewidth=2, label=architecture.upper())
    axes[0, 0].axvline(16.5, linestyle="--", color="gray", linewidth=1)
    axes[0, 0].set_xlabel("Input lag (1 = most recent observed value)")
    axes[0, 0].set_ylabel("Mean absolute raw-input gradient")
    axes[0, 0].set_title("Local sensitivity of predicted log scale")
    axes[0, 0].legend(frameon=False)
    panels = [
        (axes[0, 1], "older_than16_gradient_mass_fraction", "Gradient mass at lags17-64", "Fraction of total mass"),
        (axes[1, 0], "older_permutation_mean_abs_delta_log_scale", "Older48 block permutation; recent16 fixed", "Mean absolute change in log scale"),
        (axes[1, 1], "context64_minus16_mean_abs_delta_scale", "Context64 vs context16", "Mean absolute change in predicted scale"),
    ]
    for axis, metric, title, label in panels:
        for x, architecture in enumerate(["gru", "tcn"]):
            values = summary[summary.architecture.eq(architecture)].set_index("model_seed").loc[model_seeds, metric].to_numpy()
            positions = x + np.linspace(-.12, .12, len(values))
            axis.scatter(positions, values, color=colors[architecture], s=32, alpha=.85)
            if np.isfinite(values).any():
                axis.plot([x-.18, x+.18], [np.nanmean(values)]*2,
                           color=colors[architecture], linewidth=2)
        axis.set_xticks([0, 1], ["GRU", "TCN"])
        axis.set_title(title)
        axis.set_ylabel(label)
    for axis in axes.flat:
        axis.grid(alpha=.15)
    fig.suptitle("Validation-only model dependence: first256 saved targets, five seeds per architecture\n"
                 "Descriptive diagnostics; gradients and permutations do not establish causal effects", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / "validation_memory_diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps(DIAGNOSTIC_CONFIG, indent=2))
        return
    started = time.time()
    diagnostic_code_commit = git_code_commit()
    source_hash = file_sha256(__file__)
    training, training_manifest_hash = check_training_inputs()
    config = training["training_config"]
    validation_config = config["validation_data"]
    require_equal(validation_config["target_pairs"],
                  DIAGNOSTIC_CONFIG["expected_validation_pairs"], "validation pair count")
    validation_y = generate_paths(validation_config, config["generator"])
    require_equal(array_sha256(validation_y), training["data_sha256"]["validation"],
                  "regenerated validation observations")
    pair_path = TRAINING / "validation_target_pairs.npz"
    with np.load(pair_path) as saved:
        pair_paths, pair_targets = saved["paths"].copy(), saved["targets"].copy()
    require_equal(array_sha256(pair_paths), training["data_sha256"]["validation_pair_paths"],
                  "saved validation pair paths")
    require_equal(array_sha256(pair_targets), training["data_sha256"]["validation_pair_targets"],
                  "saved validation pair targets")
    require_equal(len(pair_paths), DIAGNOSTIC_CONFIG["expected_validation_pairs"], "saved pair count")
    start = DIAGNOSTIC_CONFIG["target_pair_start"]
    stop = start + DIAGNOSTIC_CONFIG["target_pair_count"]
    selected_paths, selected_targets = pair_paths[start:stop], pair_targets[start:stop]
    if (selected_paths.min() < 0 or selected_paths.max() >= len(validation_y)
            or selected_targets.min() < 64 or selected_targets.max() >= validation_y.shape[1]):
        raise ValueError("invalid selected validation indices")
    torch.set_num_threads(DIAGNOSTIC_CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)
    contexts = {c: context_tensor(validation_y, selected_paths, selected_targets, c, "cpu")
                for c in DIAGNOSTIC_CONFIG["contexts"]}
    permutation = np.random.default_rng(DIAGNOSTIC_CONFIG["older_block_permutation_seed"]).permutation(stop-start)
    if not np.array_equal(np.sort(permutation), np.arange(stop-start)):
        raise AssertionError("donor indices must form a permutation")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("stage2_memory already contains results; inspect before rerunning")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "date": "2026-10-11", "diagnostic_config": DIAGNOSTIC_CONFIG,
        "diagnostic_code_commit": diagnostic_code_commit,
        "source_sha256": source_hash, "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__,
        "training_manifest_sha256": training_manifest_hash,
        "training_source_sha256": training["source_sha256"],
        "protocol_sha256": training["protocol_sha256"],
        "validation_sha256": array_sha256(validation_y),
        "saved_pair_file_sha256": file_sha256(pair_path),
        "selected_pair_paths_sha256": array_sha256(selected_paths),
        "selected_pair_targets_sha256": array_sha256(selected_targets),
        "permutation_sha256": array_sha256(permutation),
        "permutation_fixed_point_count": int((permutation == np.arange(len(permutation))).sum()),
        "models": [],
        "interpretation": [
            "no test data or test results read; no training, tuning or checkpoint selection",
            "256 targets share validation paths and may overlap; no iid confidence intervals claimed",
            "raw-input gradients measure local sensitivity and can vanish under saturation/clamping",
            "older48 permutation breaks within-context dependence and may create out-of-distribution inputs",
            "diagnostics do not identify real-world causal effects or validate a deployable method",
            "older-gradient fraction is NaN when all gradient mass is zero, not evidence of recent-only memory",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pair_table = pd.DataFrame({"sample": np.arange(stop-start),
                               "saved_pair_index": np.arange(start, stop),
                               "validation_path": selected_paths,
                               "validation_target": selected_targets,
                               "older48_donor_sample": permutation})
    pair_table.to_csv(OUT / "selected_validation_pairs.csv", index=False)
    curve_rows, summary_rows, target_tables = [], [], []
    panel = {(m["architecture"], m["model_seed"]): m for m in training["models"]}
    for seed in config["model_seeds"]:
        for architecture in config["architectures"]:
            record = panel[(architecture, seed)]
            checkpoint = ROOT / record["checkpoint"]
            require_equal(file_sha256(checkpoint), record["checkpoint_sha256"], "checkpoint file")
            model, payload = load_checkpoint(checkpoint, device="cpu")
            require_equal(payload["architecture"], architecture, "checkpoint architecture")
            require_equal(payload["model_seed"], seed, "checkpoint seed")
            require_equal(payload["protocol_sha256"], training["protocol_sha256"], "checkpoint protocol")
            require_equal(payload["model_config"], config["model"], "checkpoint model config")
            require_equal(payload["contexts"], config["contexts"], "checkpoint contexts")
            require_equal(payload["selected_epoch"], record["selected_epoch"], "checkpoint selected epoch")
            profile, summary, targets = diagnose_model(model, contexts[16], contexts[64], permutation)
            for lag, value in enumerate(profile, start=1):
                curve_rows.append({"architecture": architecture, "model_seed": seed,
                                   "lag": lag, "mean_abs_raw_input_gradient": float(value),
                                   "older_than16": lag > 16})
            summary_rows.append({"architecture": architecture, "model_seed": seed,
                                 "targets": stop-start, **summary})
            targets.insert(0, "sample", np.arange(stop-start))
            targets.insert(0, "model_seed", seed)
            targets.insert(0, "architecture", architecture)
            target_tables.append(targets)
            manifest["models"].append({"architecture": architecture, "model_seed": seed,
                                       "checkpoint_sha256": record["checkpoint_sha256"],
                                       "selected_epoch": payload["selected_epoch"]})
    curves, summary = pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)
    curves.to_csv(OUT / "gradient_by_lag.csv", index=False)
    summary.to_csv(OUT / "model_memory_summary.csv", index=False)
    pd.concat(target_tables, ignore_index=True).to_csv(OUT / "target_diagnostics.csv", index=False)
    make_plot(curves, summary, config["model_seeds"])
    # Detect accidental mutation of the prerequisite files during this run.
    _, final_training_hash = check_training_inputs()
    require_equal(final_training_hash, training_manifest_hash, "training manifest after diagnostics")
    require_equal(file_sha256(__file__), source_hash, "diagnostic source after execution")
    for record in training["models"]:
        require_equal(file_sha256(ROOT / record["checkpoint"]), record["checkpoint_sha256"],
                      "checkpoint after diagnostics")
    manifest["status"] = "complete"
    manifest["completed_models"] = len(manifest["models"])
    manifest["wall_seconds"] = time.time()-started
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(OUT.iterdir())
                                  if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "models": len(manifest["models"]),
                      "validation_targets_per_model": stop-start,
                      "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
