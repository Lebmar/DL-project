"""Exploratory causal GRU-context x calibration-memory simulation.

The same fitted GRU weights are used with 16 and 64 past observations. Training
mixes both contexts; all hyperparameters are fixed before evaluation. Current
known Gaussian sigma is evaluation-only. No event identity, duration, or oracle
reset enters the GRU or the rolling calibrator. Not a novelty or market claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import norm
from torch import nn

from run_pilot import training_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "gru_simulation"
ALPHA = 0.1
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
TRAIN_SEEDS = (1701, 1702, 1703)
Q90 = float(norm.ppf(.95))


class ScaleGRU(nn.Module):
    def __init__(self, hidden=24):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, observed_context):
        # Pointwise transforms only; temporal aggregation is learned by GRU.
        x = torch.stack((observed_context / 3.0,
                         torch.log1p(observed_context.abs())), dim=-1)
        _, hidden = self.gru(x)
        return self.head(hidden[-1]).squeeze(-1).clamp(-3, 3)


def gaussian_loss(log_scale, target):
    return (log_scale + .5 * target.square() * torch.exp(-2 * log_scale)).mean()


def draw_context(y, paths, times, context):
    indices = times[:, None] + np.arange(-context, 0)[None, :]
    return torch.tensor(y[paths[:, None], indices], dtype=torch.float32)


def fit(seed, epochs, batches=24, batch_size=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_y = training_paths(seed + 10000, n=128, length=512)
    valid_y = training_paths(seed + 20000, n=40, length=512)
    model = ScaleGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    # Fixed common validation targets, separate from both training and test.
    v_paths = rng.integers(0, len(valid_y), 2048)
    v_times = rng.integers(64, valid_y.shape[1], 2048)
    valid_x = {c: draw_context(valid_y, v_paths, v_times, c) for c in CONTEXTS}
    valid_target = torch.tensor(valid_y[v_paths, v_times], dtype=torch.float32)
    history, best, best_loss = [], None, float("inf")
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        # Equal number of each context every epoch; randomized batch order.
        contexts = rng.permutation(np.resize(np.array(CONTEXTS), batches))
        for context in contexts:
            paths = rng.integers(0, len(train_y), batch_size)
            times = rng.integers(64, train_y.shape[1], batch_size)
            x = draw_context(train_y, paths, times, int(context))
            target = torch.tensor(train_y[paths, times], dtype=torch.float32)
            loss = gaussian_loss(model(x), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = {c: float(gaussian_loss(model(valid_x[c]), valid_target)) for c in CONTEXTS}
        mean_val = float(np.mean(list(val.values())))
        history.append({"model_seed": seed, "epoch": epoch + 1,
                        "train_nll": float(np.mean(batch_losses)),
                        "validation_nll_context16": val[16],
                        "validation_nll_context64": val[64],
                        "mean_validation_nll": mean_val})
        if mean_val < best_loss:
            best_loss = mean_val
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    model.eval()
    return model, history, {"seed": seed, "epochs": epochs,
                           "batches_per_epoch": batches, "batch_size": batch_size,
                           "independent_training_paths": len(train_y),
                           "independent_validation_paths": len(valid_y),
                           "best_mean_validation_nll": best_loss,
                           "unit_scale_validation_nll": float(.5 * valid_target.square().mean()),
                           "parameters": sum(p.numel() for p in model.parameters())}


def infer_targets(model, y, context, start=0, stop=None, batch_size=1024):
    """Every target t receives only y[max(0,t-context):t], left padded at start."""
    stop = y.shape[1] if stop is None else stop
    n = len(y)
    times = np.tile(np.arange(start, stop), n)
    paths = np.repeat(np.arange(n), stop - start)
    output = np.empty(len(times))
    with torch.no_grad():
        for i in range(0, len(times), batch_size):
            tt, pp = times[i:i+batch_size], paths[i:i+batch_size]
            indices = tt[:, None] + np.arange(-context, 0)[None, :]
            x = y[pp[:, None], np.maximum(indices, 0)]
            x = np.where(indices < 0, 0.0, x)
            output[i:i+batch_size] = np.exp(model(torch.tensor(x, dtype=torch.float32)).numpy())
    return output.reshape(n, stop-start)


def intervals(y, scale, window=None):
    n, length = y.shape
    if window is None:
        return Q90 * scale
    half = np.full_like(y, np.nan)
    scores = np.abs(y) / scale
    rank = int(np.ceil((window + 1) * (1 - ALPHA))) - 1
    for t in range(window, length):
        # This slice excludes y_t. The t-score can only affect t+1 onward.
        q = np.partition(scores[:, t-window:t], rank, axis=1)[:, rank]
        half[:, t] = q * scale[:, t]
    return half


def path_rows(seed, scenario, context, calibrator, window,
              y, sigma, half, onset, end):
    prob = 2 * ndtr(half / sigma) - 1
    score = 2 * half + 2 / ALPHA * np.maximum(np.abs(y) - half, 0)
    oracle_width = 2 * Q90 * sigma
    phases = {"pre": (onset-128, onset),
              "onset32": (onset, onset+32), "event": (onset, end)}
    if end < y.shape[1]:
        phases["recovery64"] = (end, min(end+64, y.shape[1]))
    rows = []
    for phase, (lo, hi) in phases.items():
        if hi <= lo:
            continue
        for path in range(len(y)):
            sl = slice(lo, hi)
            rows.append({"model_seed": seed, "scenario": scenario,
                         "context": context, "calibrator": calibrator,
                         "window": 0 if window is None else window,
                         "phase": phase, "trajectory": path, "n_times": hi-lo,
                         "expected_coverage": prob[path, sl].mean(),
                         "empirical_coverage": (np.abs(y[path, sl]) <= half[path, sl]).mean(),
                         "width": (2*half[path, sl]).mean(),
                         "interval_score": score[path, sl].mean(),
                         "mean_coverage_deficit": np.maximum(.9-prob[path, sl], 0).mean(),
                         "mean_excess_width_to_oracle": np.maximum(2*half[path, sl]-oracle_width[path, sl], 0).mean()})
    return rows, prob


def interaction_tables(frame, n, rng_seed=30991, bootstrap_reps=2000):
    """Model and test paths form a crossed design, not 3*n iid experiments."""
    measures = ["interval_score", "mean_coverage_deficit", "mean_excess_width_to_oracle"]
    rolling = frame[frame.calibrator.eq("rolling")]
    records, path_records = [], []
    rng = np.random.default_rng(rng_seed)
    for metric in measures:
        for (seed, scenario, phase), part in rolling.groupby(["model_seed", "scenario", "phase"]):
            pivot = part.pivot(index="trajectory", columns=["context", "window"], values=metric)
            delta = ((pivot[(64, 128)]-pivot[(64, 32)])
                     -(pivot[(16, 128)]-pivot[(16, 32)]))
            for path, value in delta.items():
                path_records.append({"model_seed": seed, "scenario": scenario,
                                     "phase": phase, "metric": metric,
                                     "trajectory": path, "raw_interaction": value})
    paths = pd.DataFrame(path_records)
    # Fixed onset32 exists in no_change and every duration; recovery phases
    # start at different dates and are not naively compared to no_change.
    selected = paths[paths.phase.eq("onset32")].copy()
    control = selected[selected.scenario.eq("no_change")][
        ["model_seed", "metric", "trajectory", "raw_interaction"]].rename(
            columns={"raw_interaction": "control_interaction"})
    adjusted = selected.merge(control, on=["model_seed", "metric", "trajectory"], validate="many_to_one")
    adjusted["adjusted_interaction"] = adjusted.raw_interaction-adjusted.control_interaction
    for (seed, scenario, metric), part in adjusted.groupby(["model_seed", "scenario", "metric"]):
        values = part.adjusted_interaction.to_numpy()
        resamples = rng.integers(0, n, size=(bootstrap_reps, n))
        boot = values[resamples].mean(axis=1)
        records.append({"model_seed": seed, "scenario": scenario, "phase": "onset32",
                        "metric": metric, "raw_interaction": part.raw_interaction.mean(),
                        "control_interaction": part.control_interaction.mean(),
                        "adjusted_interaction": values.mean(),
                        "test_path_se_conditional_on_model": values.std(ddof=1)/np.sqrt(n),
                        "ci95_low": np.quantile(boot, .025),
                        "ci95_high": np.quantile(boot, .975)})
    by_seed = pd.DataFrame(records)
    pooled = []
    for (scenario, metric), part in adjusted.groupby(["scenario", "metric"]):
        grid = part.pivot(index="model_seed", columns="trajectory", values="adjusted_interaction").to_numpy()
        # Crossed resampling: model seeds and path ids independently; all
        # model outputs on a selected path stay paired with no_change.
        bvals = np.empty(bootstrap_reps)
        for b in range(bootstrap_reps):
            si = rng.integers(0, grid.shape[0], grid.shape[0])
            ti = rng.integers(0, grid.shape[1], grid.shape[1])
            bvals[b] = grid[np.ix_(si, ti)].mean()
        seed_means = grid.mean(axis=1)
        pooled.append({"scenario": scenario, "phase": "onset32", "metric": metric,
                       "adjusted_interaction_mean": grid.mean(),
                       "min_seed_effect": seed_means.min(), "max_seed_effect": seed_means.max(),
                       "seed_effect_sd": seed_means.std(ddof=1),
                       "all_seeds_positive": bool((seed_means > 0).all()),
                       "all_seeds_negative": bool((seed_means < 0).all()),
                       "crossed_bootstrap_ci95_low": np.quantile(bvals, .025),
                       "crossed_bootstrap_ci95_high": np.quantile(bvals, .975)})
    return paths, adjusted, by_seed, pd.DataFrame(pooled)


def make_plot(curves, n):
    # Curves average models and paths only descriptively; inference is in CSV.
    grouped = curves.groupby(["scenario", "context", "window", "relative_time"])[
        ["expected_coverage", "width", "oracle_width"]].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="row")
    colors = ("#2669a5", "#bd6b36", "#778047", "#956b9f")
    combos = [(16, 32), (16, 128), (64, 32), (64, 128)]
    for col, scenario in enumerate(["shock_5", "shift_128"]):
        d = grouped[grouped.scenario.eq(scenario)]
        for color, (context, window) in zip(colors, combos):
            p = d[d.context.eq(context) & d.window.eq(window)]
            axes[0, col].plot(p.relative_time, p.expected_coverage, color=color,
                              label=f"GRU context {context}, calibration {window}")
            axes[1, col].plot(p.relative_time, p.width, color=color)
        oracle = p[["relative_time", "oracle_width"]]
        axes[1, col].plot(oracle.relative_time, oracle.oracle_width, "--", color="black")
        axes[0, col].axhline(.9, color="black", linestyle="--", linewidth=1)
        duration = 5 if scenario == "shock_5" else 128
        for ax in axes[:, col]:
            ax.axvspan(0, duration, color="gray", alpha=.12)
            ax.grid(alpha=.15)
        axes[0, col].set_title(f"Scale pulse duration {duration}: sigma 1 → 3 → 1")
        axes[1, col].set_xlabel("Time relative to onset")
    axes[0, 0].set_ylabel("Conditional coverage")
    axes[1, 0].set_ylabel("Interval width")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Exploratory GRU diagnostic: 3 training seeds × {n} paired test paths\n"
                 "Descriptive averages; bootstrap uncertainty reported separately")
    fig.tight_layout(rect=[0, .12, 1, .94])
    fig.savefig(OUT/"gru_context_response.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    if args.trajectories < 2 or args.epochs < 1:
        raise ValueError("trajectories >= 2 and epochs >= 1 are required")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.time()
    n, length, onset = args.trajectories, 768, 256
    noise = np.stack([np.random.default_rng(50000+i).normal(size=length) for i in range(n)])
    durations = {"no_change": 0, "shock_1": 1, "shock_5": 5,
                 "shift_32": 32, "shift_128": 128, "persistent": length-onset}
    metrics, curves, histories, training, checks, context_diagnostics = [], [], [], [], [], []
    for seed in TRAIN_SEEDS:
        seed_start = time.time()
        model, history, train_meta = fit(seed, args.epochs)
        histories.extend(history)
        torch.save(model.state_dict(), OUT/f"gru_seed_{seed}.pt")
        base_scale = {c: infer_targets(model, noise, c) for c in CONTEXTS}
        first_event, after_first = {}, {}
        for scenario, duration in durations.items():
            sigma = np.ones_like(noise)
            sigma[:, onset:onset+duration] = 3
            y = noise*sigma
            scenario_scales = {}
            for context in CONTEXTS:
                # Exact finite-context reuse: outside this span, the full
                # observed input window equals the no_change window.
                scale = base_scale[context].copy()
                if duration:
                    lo, hi = onset+1, min(length, onset+duration+context)
                    scale[:, lo:hi] = infer_targets(model, y, context, lo, hi)
                scenario_scales[context] = scale
                if scenario == "shift_128":
                    cut = onset+8
                    changed = y.copy()
                    changed[:, cut:] *= 97
                    observed = infer_targets(model, changed, context, cut, cut+1)
                    checks.append(bool(np.allclose(observed[:, 0], scale[:, cut], rtol=1e-6, atol=1e-7)))
                for window in (None, *WINDOWS):
                    calibrator = "native" if window is None else "rolling"
                    half = intervals(y, scale, window)
                    assert np.isfinite(half[:, 128:]).all()
                    key = (context, 0 if window is None else window)
                    if duration:
                        first_event.setdefault(key, half[:, onset].copy())
                        after_first.setdefault(key, half[:, onset+1].copy())
                        checks.append(bool(np.allclose(first_event[key], half[:, onset], rtol=1e-6)))
                        checks.append(bool(np.allclose(after_first[key], half[:, onset+1], rtol=1e-6)))
                    rows, prob = path_rows(seed, scenario, context, calibrator, window,
                                           y, sigma, half, onset, onset+duration)
                    metrics.extend(rows)
                    if window is not None:
                        for t in range(onset-24, onset+257):
                            curves.append({"model_seed": seed, "scenario": scenario,
                                           "context": context, "window": window,
                                           "relative_time": t-onset,
                                           "expected_coverage": prob[:, t].mean(),
                                           "width": (2*half[:, t]).mean(),
                                           "oracle_width": (2*Q90*sigma[:, t]).mean()})
            diagnostics_phases = {"pre": (onset-128, onset), "onset32": (onset, onset+32),
                                  "relative64_to128": (onset+64, onset+128),
                                  "relative128_to192": (onset+128, onset+192)}
            for phase, (lo, hi) in diagnostics_phases.items():
                a = scenario_scales[16][:, lo:hi]
                b = scenario_scales[64][:, lo:hi]
                dl = np.log(b/a)
                context_diagnostics.append({"model_seed": seed, "scenario": scenario, "phase": phase,
                                            "mean_scale_context16": a.mean(), "mean_scale_context64": b.mean(),
                                            "mean_abs_log_scale_difference": np.abs(dl).mean(),
                                            "q95_abs_log_scale_difference": np.quantile(np.abs(dl), .95),
                                            "max_abs_log_scale_difference": np.abs(dl).max(),
                                            "log_scale_correlation": np.corrcoef(np.log(a).ravel(), np.log(b).ravel())[0, 1]})
        train_meta["seed_wall_seconds"] = time.time()-seed_start
        training.append(train_meta)
        print(json.dumps({"finished_seed": seed, "best_validation_nll": train_meta["best_mean_validation_nll"],
                          "seed_wall_seconds": train_meta["seed_wall_seconds"]}), flush=True)
    assert all(checks)
    frame = pd.DataFrame(metrics)
    curve_frame = pd.DataFrame(curves)
    frame.to_csv(OUT/"trajectory_metrics.csv", index=False)
    curve_frame.to_csv(OUT/"event_curves.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT/"training_history.csv", index=False)
    pd.DataFrame(context_diagnostics).to_csv(OUT/"context_prediction_diagnostics.csv", index=False)
    measures = ["expected_coverage", "empirical_coverage", "width", "interval_score",
                "mean_coverage_deficit", "mean_excess_width_to_oracle"]
    groups = ["model_seed", "scenario", "context", "calibrator", "window", "phase"]
    means = frame.groupby(groups)[measures].mean()
    ses = frame.groupby(groups)[measures].std()/np.sqrt(n)
    means.join(ses.add_suffix("_test_path_se")).reset_index().to_csv(OUT/"summary_by_seed.csv", index=False)
    paths, adjusted, by_seed, pooled = interaction_tables(frame, n)
    paths.to_csv(OUT/"raw_interactions_by_path.csv", index=False)
    adjusted.to_csv(OUT/"adjusted_interactions_by_path.csv", index=False)
    by_seed.to_csv(OUT/"adjusted_interactions_by_seed.csv", index=False)
    pooled.to_csv(OUT/"adjusted_interactions_crossed_bootstrap.csv", index=False)
    make_plot(curve_frame, n)
    manifest = {"date": "2026-10-11", "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_generator_source_sha256": hashlib.sha256((ROOT/"scripts/run_pilot.py").read_bytes()).hexdigest(),
                "architecture": "one-layer GRU hidden24 plus Gaussian log-scale head; fixed zero mean",
                "training": training, "contexts": list(CONTEXTS), "calibration_windows": list(WINDOWS),
                "simulation": {"test_paths": n, "steps": length, "onset": onset,
                               "durations": durations, "sigma_event": 3,
                               "innovations": "iid Gaussian, deterministic known scale",
                               "test_seeds": "50000+i", "same_test_noise_across_models_and_scenarios": True},
                "checks": {"causal_suffix_and_matching_event_prefix_checks": all(checks),
                           "number_of_checks": len(checks), "finite_intervals_after128": True},
                "interaction": "[(ctx64,cal128)-(ctx64,cal32)]-[(ctx16,cal128)-(ctx16,cal32)] minus no_change same onset32 window",
                "uncertainty": "2000 paired test-path bootstrap reps per model; crossed model-seed/path bootstrap pooled, only 3 seeds",
                "limitations": ["exploratory fixed protocol, not confirmatory; no test-based hyperparameter tuning",
                                "context length is not necessarily effective learned memory",
                                "known sigma/CDF/oracle width used only for evaluation",
                                "fixed zero mean and Gaussian deterministic event scale",
                                "onset32 mixes event and recovery for short events",
                                "only three model seeds; crossed-bootstrap intervals approximate seed variation",
                                "no strong adaptive conformal/GARCH/NCC baselines or real data",
                                "Gaussian NLL predicts moments and need not yield calibrated mixture quantiles"],
                "wall_seconds": time.time()-start}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"wall_seconds": manifest["wall_seconds"], "all_checks": all(checks),
                      "adjusted_interval_score_interactions": pooled[pooled.metric.eq("interval_score")].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
