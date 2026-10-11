"""Causal paired simulation; diagnostic, not an originality or market claim.

The neural scale model uses causal summaries of past observations. Both memory
settings use exactly the same fitted weights. Calibration is forecast-time
normalization: old scores retain their old forecast scales. No oracle reset,
event identity or future duration is supplied to any predictor/calibrator.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pilot"
ALPHA = 0.1
Q90 = float(norm.ppf(1 - ALPHA / 2))


class ScaleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 24), nn.Tanh(),
                                 nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1).clamp(-3, 3)


def causal_features(y: np.ndarray, memory: int) -> np.ndarray:
    """x[:,t] is measurable before y[:,t]. Four fixed, causal summaries."""
    n, length = y.shape
    slow = np.ones(n)
    fast = np.ones(n)
    previous = np.zeros(n)
    result = np.empty((n, length, 4), dtype=np.float32)
    for t in range(length):
        log_s = 0.5 * np.log(np.maximum(slow, 1e-5))
        log_f = 0.5 * np.log(np.maximum(fast, 1e-5))
        result[:, t] = np.column_stack([
            log_s, log_f, log_f - log_s,
            np.log1p(np.abs(previous) / np.sqrt(np.maximum(slow, 1e-5))),
        ])
        slow += (y[:, t] ** 2 - slow) / memory
        fast += (y[:, t] ** 2 - fast) / max(2, memory // 4)
        previous = y[:, t]
    return result


def training_paths(seed: int, n: int = 160, length: int = 512):
    rng = np.random.default_rng(seed)
    persistent = np.ones(n)
    sigma = np.empty((n, length))
    for t in range(length):
        switches = rng.random(n) < 0.018
        persistent[switches] = np.exp(rng.uniform(np.log(0.5), np.log(3), switches.sum()))
        temporary = np.where(rng.random(n) < 0.015, 3.0, 1.0)
        sigma[:, t] = persistent * temporary
    return sigma * rng.normal(size=(n, length))


def fit_neural(seed: int, epochs: int):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    train_y = training_paths(seed + 100)
    valid_y = training_paths(seed + 200, n=40)
    xs, ys = [], []
    vx, vy = [], []
    for memory in [8, 64]:
        xs.append(causal_features(train_y, memory)[:, 64:].reshape(-1, 4))
        ys.append(train_y[:, 64:].reshape(-1))
        vx.append(causal_features(valid_y, memory)[:, 64:].reshape(-1, 4))
        vy.append(valid_y[:, 64:].reshape(-1))
    x = torch.from_numpy(np.concatenate(xs))
    y = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    val_x = torch.from_numpy(np.concatenate(vx))
    val_y = torch.tensor(np.concatenate(vy), dtype=torch.float32)
    model = ScaleMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    best_loss = float("inf")
    best = None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        for start in range(0, len(x), 2048):
            ix = order[start:start + 2048]
            log_s = model(x[ix])
            loss = (log_s + 0.5 * y[ix].square() * torch.exp(-2 * log_s)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            z = model(val_x)
            val_loss = (z + 0.5 * val_y.square() * torch.exp(-2 * z)).mean().item()
        history.append({"epoch": epoch + 1, "validation_gaussian_nll_without_constant": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    with torch.no_grad():
        baseline_nll = float(0.5 * val_y.square().mean())
    return model, {"seed": seed, "epochs": epochs, "history": history,
                   "best_validation_nll": best_loss, "fixed_unit_scale_nll": baseline_nll,
                   "train_samples": len(x), "validation_samples": len(val_x),
                   "parameters": sum(p.numel() for p in model.parameters())}


def higher_quantile(scores, alpha):
    """Finite-sample rank ceil((n+1)*(1-alpha)); caller enforces finite rank."""
    n = scores.shape[1]
    rank = np.ceil((n + 1) * (1 - np.asarray(alpha))).astype(int)
    rank = np.clip(rank, 1, n)
    return np.sort(scores, axis=1)[np.arange(len(scores)), rank - 1]


def forecast_intervals(y, scale, method, window):
    n, length = y.shape
    half = np.full_like(y, np.nan)
    scores = np.abs(y) / scale
    alpha_t = np.full(n, ALPHA)
    clipped = 0
    ewma = np.ones(n)
    for t in range(length):
        # Compute from y[:t] only; current score enters update after prediction.
        if t >= window:
            if method == "native":
                half[:, t] = Q90 * scale[:, t]
            elif method == "ewma":
                half[:, t] = Q90 * np.sqrt(np.maximum(ewma, 1e-8)) * scale[:, t]
            else:
                a = alpha_t if method.startswith("aci") else np.full(n, ALPHA)
                lower = 1.0 / (window + 1)
                a = np.clip(a, lower, 0.5)
                half[:, t] = higher_quantile(scores[:, t-window:t], a) * scale[:, t]
        if method.startswith("aci") and t >= window:
            gamma = float(method.split("_")[1])
            miss = np.abs(y[:, t]) > half[:, t]
            alpha_t += gamma * (ALPHA - miss)
            clipped += int(((alpha_t < 1.0/(window+1)) | (alpha_t > 0.5)).sum())
            alpha_t = np.clip(alpha_t, 1.0 / (window + 1), 0.5)
        if method == "ewma":
            ewma += (scores[:, t] ** 2 - ewma) / window
    return half, clipped


def interval_score(y, half):
    return 2 * half + 2 / ALPHA * np.maximum(np.abs(y) - half, 0)


def summarize_rows(scenario, predictor, method, window, y, sigma, half, onset, end):
    coverage_prob = 2 * ndtr(half / sigma) - 1
    hit = np.abs(y) <= half
    score = interval_score(y, half)
    oracle_half = Q90 * sigma
    phases = {"pre": slice(onset-128, onset),
              "onset32": slice(onset, onset+32),
              "event": slice(onset, min(end, y.shape[1]))}
    if end < y.shape[1]:
        phases["recovery64"] = slice(end, min(end+64, y.shape[1]))
    rows = []
    for phase, sl in phases.items():
        if sl.stop <= sl.start:
            continue
        for seed in range(len(y)):
            rows.append({"scenario": scenario, "predictor": predictor,
                         "calibrator": method, "window": window, "phase": phase,
                         "trajectory": seed, "n_times": sl.stop-sl.start,
                         "expected_coverage": float(np.mean(coverage_prob[seed, sl])),
                         "empirical_coverage": float(np.mean(hit[seed, sl])),
                         "width": float(np.mean(2*half[seed, sl])),
                         "interval_score": float(np.mean(score[seed, sl])),
                         "width_ratio_to_oracle": float(np.mean(half[seed, sl]/oracle_half[seed, sl])),
                         "mean_coverage_shortfall": float(np.mean(np.maximum(0.9-coverage_prob[seed, sl], 0))),
                         "excess_width": float(np.mean(np.maximum(2*half[seed, sl]-2*oracle_half[seed, sl], 0)))})
    return rows, coverage_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    model, training = fit_neural(731, args.epochs)
    pd.DataFrame(training["history"]).to_csv(OUT/"training_history.csv", index=False)
    torch.save(model.state_dict(), OUT/"neural_scale.pt")
    n, length, onset = args.trajectories, 896, 384
    noise = np.stack([np.random.default_rng(10000+i).normal(size=length) for i in range(n)])
    scenarios = {"no_change": 0, "shock_1": 1, "shock_5": 5, "shift_32": 32,
                 "shift_128": 128, "persistent": length-onset}
    calibrators = [("native", 128), ("rolling", 32), ("rolling", 128),
                   ("aci_0.005", 128), ("aci_0.05", 128), ("ewma", 8), ("ewma", 64)]
    rows, curves, all_half = [], [], {}
    counts = {}
    causal_checks = []
    for scenario, duration in scenarios.items():
        sigma = np.ones_like(noise)
        sigma[:, onset:onset+duration] = 3
        y = sigma*noise
        # Raw observed histories share the prefix, including first event outcome.
        for memory in [8, 64]:
            feat = causal_features(y, memory)
            # Direct suffix intervention check: features before cut must agree.
            modified = y.copy()
            modified[:, onset+8:] *= 97
            other_feat = causal_features(modified, memory)
            causal_checks.append(bool(np.array_equal(feat[:, :onset+9], other_feat[:, :onset+9])))
        predictors = {"fixed_scale": np.ones_like(y)}
        with torch.no_grad():
            for memory in [8, 64]:
                x = causal_features(y, memory)
                pred = np.exp(model(torch.from_numpy(x.reshape(-1, 4))).numpy()).reshape(y.shape)
                predictors[f"neural_memory_{memory}"] = pred
        for predictor, scale in predictors.items():
            for method, window in calibrators:
                half, clipped = forecast_intervals(y, scale, method, window)
                assert np.isfinite(half[:, 128:]).all()
                records, prob = summarize_rows(scenario, predictor, method, window,
                                              y, sigma, half, onset, onset+duration)
                rows.extend(records)
                key = (scenario, predictor, method, window)
                all_half[key] = half
                counts["|".join(map(str, key))] = clipped
                for t in range(onset-64, min(length, onset+320)):
                    curves.append({"scenario": scenario, "predictor": predictor,
                                   "calibrator": method, "window": window,
                                   "relative_time": t-onset,
                                   "expected_coverage": prob[:, t].mean(),
                                   "expected_coverage_se": prob[:, t].std(ddof=1)/np.sqrt(n),
                                   "width": (2*half[:, t]).mean(),
                                   "width_se": (2*half[:, t]).std(ddof=1)/np.sqrt(n),
                                   "oracle_width": (2*Q90*sigma[:, t]).mean()})
    assert all(causal_checks)
    # All event scenarios must issue the same first-event prediction.
    first_predictions_agree = True
    for predictor in ["fixed_scale", "neural_memory_8", "neural_memory_64"]:
        for method, window in calibrators:
            expected = all_half[("shock_1", predictor, method, window)][:, onset]
            for scenario in ["shock_5", "shift_32", "shift_128", "persistent"]:
                first_predictions_agree &= bool(np.array_equal(
                    expected, all_half[(scenario, predictor, method, window)][:, onset]))
    assert first_predictions_agree
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"trajectory_metrics.csv", index=False)
    curve = pd.DataFrame(curves)
    curve.to_csv(OUT/"event_curves.csv", index=False)
    measures = ["expected_coverage", "empirical_coverage", "width", "interval_score",
                "width_ratio_to_oracle", "mean_coverage_shortfall", "excess_width"]
    groups = ["scenario", "predictor", "calibrator", "window", "phase"]
    means = frame.groupby(groups)[measures].mean()
    se = frame.groupby(groups)[measures].std()/np.sqrt(n)
    summary = means.join(se.add_suffix("_se")).reset_index()
    summary.to_csv(OUT/"summary.csv", index=False)
    # Paired 2x2 predictor-memory x calibrator-memory interaction; no selection.
    interactions = []
    for (scenario, phase), part in frame.groupby(["scenario", "phase"]):
        part = part[part.calibrator.eq("rolling") & part.predictor.str.startswith("neural")]
        for measure in ["interval_score", "mean_coverage_shortfall", "excess_width"]:
            pivot = part.pivot(index="trajectory", columns=["predictor", "window"], values=measure)
            delta = ((pivot[("neural_memory_64", 128)]-pivot[("neural_memory_64", 32)])
                     -(pivot[("neural_memory_8", 128)]-pivot[("neural_memory_8", 32)]))
            se_delta = delta.std(ddof=1)/np.sqrt(n)
            interactions.append({"scenario": scenario, "phase": phase, "metric": measure,
                                 "interaction": delta.mean(), "se": se_delta,
                                 "ci95_low": delta.mean()-1.96*se_delta,
                                 "ci95_high": delta.mean()+1.96*se_delta,
                                 "definition": "(long predictor: long-short calibrator) - (short predictor: long-short calibrator)"})
    interaction_frame = pd.DataFrame(interactions)
    interaction_frame.to_csv(OUT/"memory_interactions.csv", index=False)
    # Remove any steady-state interaction using the exactly paired no-event path.
    adjusted = []
    for scenario in ["shock_1", "shock_5", "shift_32", "shift_128", "persistent"]:
        for metric in ["interval_score", "mean_coverage_shortfall", "excess_width"]:
            selected = frame[(frame.phase == "onset32") & (frame.calibrator == "rolling")
                             & frame.predictor.str.startswith("neural")]
            def deltas(name):
                p = selected[selected.scenario == name].pivot(index="trajectory", columns=["predictor","window"], values=metric)
                return ((p[("neural_memory_64",128)]-p[("neural_memory_64",32)])
                        -(p[("neural_memory_8",128)]-p[("neural_memory_8",32)]))
            delta = deltas(scenario)-deltas("no_change")
            se_d = delta.std(ddof=1)/np.sqrt(n)
            adjusted.append({"scenario": scenario, "phase": "first32_after_onset", "metric": metric,
                             "event_minus_nochange_interaction": delta.mean(), "se": se_d,
                             "ci95_low": delta.mean()-1.96*se_d, "ci95_high": delta.mean()+1.96*se_d})
    pd.DataFrame(adjusted).to_csv(OUT/"event_adjusted_interactions.csv", index=False)
    make_plots(curve, summary, n)
    manifest = {"date": "2026-10-11", "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__,
                "simulation": {"independent_trajectories": n, "steps": length,
                               "onset": onset, "sigma_before": 1, "sigma_event": 3,
                               "durations": scenarios, "innovations": "iid standard Gaussian",
                               "paired_noise_across_scenarios_and_methods": True},
                "nominal_coverage": 0.9, "training": training,
                "checks": {"causal_feature_suffix_intervention": all(causal_checks),
                           "identical_first_event_predictions": first_predictions_agree,
                           "finite_intervals_after_burn_in": True},
                "limitations": ["one neural training seed; reported SE is over independent test paths only",
                                "causal-summary MLP is a diagnostic surrogate, not a time-series foundation model; memory denotes EWMA input summaries, not learned recurrent memory",
                                "no strong adaptive conformal / GARCH / NCC replication yet",
                                "clipped ACI is a practical variant, not exact theoretical ACI",
                                "no real market data in this pilot",
                                "oracle sigma is used for evaluation only",
                                "multiple exploratory comparisons; unadjusted intervals"],
                "aci_clipping_counts": counts, "wall_seconds": time.time()-start_time,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"validation_nll": training["best_validation_nll"],
                      "fixed_scale_nll": training["fixed_unit_scale_nll"],
                      "independent_trajectories": n, "checks": manifest["checks"],
                      "wall_seconds": manifest["wall_seconds"]}, indent=2))


def make_plots(curve, summary, n):
    colors = ["#2669a5", "#bd6b36", "#778047", "#956b9f"]
    selected = [("neural_memory_8", 32), ("neural_memory_8", 128),
                ("neural_memory_64", 32), ("neural_memory_64", 128)]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", sharey="row")
    for col, scenario in enumerate(["shock_5", "shift_128"]):
        for i, (pred, window) in enumerate(selected):
            d = curve[(curve.scenario == scenario) & (curve.predictor == pred)
                      & (curve.calibrator == "rolling") & (curve.window == window)]
            d = d[(d.relative_time >= -24) & (d.relative_time <= 256)]
            label = f"Predictor memory {pred.rsplit('_',1)[1]}, calibration {window}"
            axes[0, col].plot(d.relative_time, d.expected_coverage, color=colors[i], label=label)
            axes[0, col].fill_between(d.relative_time,
                                     d.expected_coverage-1.96*d.expected_coverage_se,
                                     d.expected_coverage+1.96*d.expected_coverage_se,
                                     color=colors[i], alpha=.10, linewidth=0)
            axes[1, col].plot(d.relative_time, d.width, color=colors[i])
        oracle = d[["relative_time", "oracle_width"]]
        axes[1, col].plot(oracle.relative_time, oracle.oracle_width, "--", color="#333333", label="Oracle width")
        axes[0, col].axhline(.9, linestyle="--", color="#333333", linewidth=1)
        duration = 5 if scenario == "shock_5" else 128
        for ax in axes[:, col]:
            ax.axvspan(0, duration, color="#777777", alpha=.1)
            ax.set_xlim(-24, 256)
            ax.grid(alpha=.16)
        axes[0, col].set_title(f"Duration {duration}; sigma 1 → 3 → 1")
        axes[0, col].set_ylim(0.35, 1.01)
        axes[1, col].set_xlabel("Time relative to event onset")
    axes[0, 0].set_ylabel(f"Conditional coverage probability\n(mean over {n} independent paths)")
    axes[1, 0].set_ylabel("Mean prediction interval width")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Diagnostic simulation: input-summary and calibration memory\nCoverage bands: pointwise 95% Monte Carlo intervals", y=.99)
    fig.tight_layout(rect=[0,.12,1,.96])
    fig.savefig(OUT/"memory_response.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
