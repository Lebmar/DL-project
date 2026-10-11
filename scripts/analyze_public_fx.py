"""Exploratory ECB pilot summaries with paired circular date-block resampling.

All assets, methods and fitted seeds on a date are kept together. Fitted seeds
are averaged, not treated as independent observations. Intervals describe date
sampling variation conditional on the three fitted models; a block bootstrap
does not itself justify confidence intervals under arbitrary distribution shift.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "public_fx"


def method_name(row):
    suffix = row.calibrator if row.calibrator in {"native", "sfogd", "saocp"} else f"{row.calibrator}{int(row.window)}"
    return f"{row.predictor}{row.context}_{suffix}"


def bootstrap_means(values, block_length, reps=2000):
    """Paired circular blocks; identical index draws for every comparison."""
    n = len(values)
    rng = np.random.default_rng(2081 + block_length)
    means = np.empty((reps, values.shape[1]))
    for b in range(reps):
        starts = rng.integers(n, size=int(np.ceil(n / block_length)))
        ix = ((starts[:, None] + np.arange(block_length)) % n).ravel()[:n]
        means[b] = values[ix].mean(axis=0)
    return means


def main():
    prediction_path = OUT / "predictions.csv"
    if not prediction_path.exists():
        prediction_path = prediction_path.with_suffix(".csv.gz")
    frame = pd.read_csv(prediction_path, parse_dates=["date"])
    strong_path = ROOT / "outputs" / "strong_calibration" / "predictions.csv"
    if not strong_path.exists():
        strong_path = strong_path.with_suffix(".csv.gz")
    if strong_path.exists():
        strong = pd.read_csv(strong_path, parse_dates=["date"])
        strong["window"] = 0  # algorithm has no fixed score window
        frame = pd.concat([frame, strong], ignore_index=True)
    run = json.loads((OUT / "run_manifest.json").read_text())
    frame["normalized_interval_score"] = frame.interval_score / frame.asset.map(run["training_scales"])
    frame["normalized_width"] = 2 * frame.half_width / frame.asset.map(run["training_scales"])
    frame["method"] = frame.apply(method_name, axis=1)
    metrics = ["hit", "normalized_interval_score", "normalized_width"]
    # Every method has all four assets on every date. Equal-weight assets/seeds.
    date_means = frame.groupby(["date", "method"])[metrics].mean()
    assert date_means.groupby("method").size().nunique() == 1
    wide = {m: date_means[m].unstack("method").sort_index() for m in metrics}
    assert all(not v.isna().any().any() for v in wide.values())
    assert all(v.index.equals(wide["hit"].index) for v in wide.values())
    rows, contrasts = [], []
    comparisons = {
        "GRU64 minus EWMA16, native": {"gru64_native": 1, "ewma16_native": -1},
        "GRU64 minus EWMA64, native": {"gru64_native": 1, "ewma64_native": -1},
        "GRU64 minus GRU16, native": {"gru64_native": 1, "gru16_native": -1},
        "GRU context x calibration-window interaction": {
            "gru64_rolling128": 1, "gru64_rolling32": -1,
            "gru16_rolling128": -1, "gru16_rolling32": 1},
    }
    if strong_path.exists():
        comparisons.update({
            "SF-OGD minus rolling128, fixed GRU64": {"gru64_sfogd": 1, "gru64_rolling128": -1},
            "SAOCP minus rolling128, fixed GRU64": {"gru64_saocp": 1, "gru64_rolling128": -1},
        })
    for block in [20, 60]:
        for metric, table in wide.items():
            draws = bootstrap_means(table.to_numpy(), block)
            for j, method in enumerate(table.columns):
                lower, upper = np.quantile(draws[:, j], [.025, .975])
                rows.append({"method": method, "metric": metric, "block_length": block,
                             "mean": table[method].mean(), "ci_low": lower, "ci_high": upper,
                             "dates": len(table), "bootstrap_reps": len(draws)})
            for label, coefficients in comparisons.items():
                weights = np.array([coefficients.get(c, 0) for c in table.columns])
                lower, upper = np.quantile(draws @ weights, [.025, .975])
                contrasts.append({"comparison": label, "metric": metric, "block_length": block,
                                  "mean_difference": float(table.mean().to_numpy() @ weights),
                                  "ci_low": lower, "ci_high": upper})
    pd.DataFrame(rows).to_csv(OUT / "date_block_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "paired_date_block_comparisons.csv", index=False)
    # Descriptive calendar years are not labelled ex-post causal market regimes.
    frame["year"] = frame.date.dt.year
    frame.groupby(["year", "method"])[metrics].mean().reset_index().to_csv(OUT / "calendar_year_summary.csv", index=False)
    frame.groupby(["seed", "method"])[metrics].mean().reset_index().to_csv(OUT / "seed_summary.csv", index=False)

    selected = ["ewma16_native", "ewma64_native", "gru16_native", "gru64_native",
                "gru16_rolling128", "gru64_rolling128"]
    labels = ["EWMA 16 · native", "EWMA 64 · native", "GRU 16 · native", "GRU 64 · native",
              "GRU 16 · rolling 128", "GRU 64 · rolling 128"]
    if strong_path.exists():
        selected.extend(["gru64_sfogd", "gru64_saocp"])
        labels.extend(["GRU 64 · SF-OGD", "GRU 64 · SAOCP"])
    summary = pd.DataFrame(rows).query("block_length == 20").set_index(["method", "metric"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, metric, title in zip(axes, ["hit", "normalized_interval_score"],
                               ["Observed coverage (target 90%)", "Mean interval score / training std (lower is better)"]):
        values = summary.xs(metric, level="metric").loc[selected]
        ax.errorbar(values["mean"], range(len(selected)),
                    xerr=[values["mean"] - values.ci_low, values.ci_high - values["mean"]],
                    fmt="o", capsize=3, color="#226b83")
        ax.set_yticks(range(len(selected)), labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=.2)
        if metric == "hit":
            ax.axvline(.9, color="#b75c32", linestyle="--", linewidth=1)
    fig.suptitle("ECB FX pilot: forecast and calibration comparisons", fontsize=13)
    fig.text(.5, .01, "2021–2026-10-09 · 1,479 paired dates · four EUR-cross assets · three fixed GRU fits\n"
             "Exploratory 95% circular date-block intervals (20 dates); not a guarantee under arbitrary shift",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .93])
    fig.savefig(OUT / "public_fx_comparison.png", dpi=180)
    plt.close(fig)
    manifest = {"prediction_file_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                "prediction_file": str(prediction_path.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "assets_equal_weight": True, "seeds_average_not_independent": True,
                "normalization": "divide per-row width/interval score by train-only asset std",
                "resampling": "paired circular date blocks, all methods/assets/fits retained on a date",
                "block_lengths": [20, 60], "replicates": 2000,
                "dates": len(wide["hit"]), "limitations": [
                    "conditional on the three fitted models; not full training uncertainty",
                    "only four related EUR crosses; no broad finance generalization",
                    "bootstrap is exploratory and does not ensure validity under arbitrary nonstationarity",
                    "multiple comparisons are unadjusted; no confirmatory significance claim"]}
    if strong_path.exists():
        manifest["strong_prediction_file_sha256"] = hashlib.sha256(strong_path.read_bytes()).hexdigest()
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(contrasts).query("metric == 'normalized_interval_score'").to_string(index=False))


if __name__ == "__main__":
    main()
