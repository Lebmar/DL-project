"""Past-only constant-width-factor controls on previously viewed FX predictions.

Exploratory confound check, not new confirmation, method selection or training.
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
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/fx_static_scale_protocol.json"
OUT = ROOT / "outputs/fx_static_scale"
METRICS = ["coverage", "width", "interval_score"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, half, alpha=.1):
    return np.stack((np.abs(y) <= half, 2*half,
                     2*half+2/alpha*np.maximum(np.abs(y)-half, 0)), axis=-1)


def fit_factors(y, scale, alpha=.1):
    scores = np.abs(y)/scale
    order = np.argsort(scores, kind="stable")
    ordered, weights = scores[order], scale[order]
    rank = int(np.ceil((len(y)+1)*(1-alpha)))
    assert 1 <= rank <= len(y)
    index = int(np.searchsorted(np.cumsum(weights), (1-alpha)*weights.sum(), side="left"))
    return float(ordered[rank-1]), float(ordered[index])


def verify_is_minimum(y, scale, factor):
    # Independent exhaustive check of all piecewise-linear objective knots.
    candidates = np.r_[0., np.unique(np.abs(y)/scale)]
    scores = metrics(y[None, :], candidates[:, None]*scale[None, :])[:, :, 2].mean(axis=1)
    actual = metrics(y, factor*scale)[:, 2].mean()
    np.testing.assert_allclose(actual, scores.min(), rtol=1e-12, atol=1e-12)


def self_check():
    # Nonuniform scales make the IS-optimal factor differ from the usual quantile.
    scale = np.r_[100., np.ones(19)]
    y = scale*np.arange(1., 21.)
    ordinary, optimal = fit_factors(y, scale)
    assert ordinary == 19. and optimal == 9.
    verify_is_minimum(y, scale, optimal)
    for factor in [.5, 2., 3.]:
        np.testing.assert_allclose(np.array(fit_factors(y, factor*scale))*factor,
                                   [ordinary, optimal], rtol=1e-12, atol=1e-12)
    # Ties and zero residuals remain legitimate objective knots.
    verify_is_minimum(np.array([0., 0., 2., 2., 3., 3., 5., 5., 9., 9.]),
                      np.ones(10), fit_factors(np.array([0., 0., 2., 2., 3., 3., 5., 5., 9., 9.]), np.ones(10))[1])
    return {"weighted_quantile_minimizer": True, "scale_equivariance": True, "ties_and_zero_residuals": True}


def date_block_weights(n, block, repetitions, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(repetitions, int(np.ceil(n/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    weights = np.zeros((repetitions, n))
    np.add.at(weights, (np.arange(repetitions)[:, None], indices), 1/n)
    np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)
    return weights


def run():
    started = time.time()
    checks = self_check()
    p = json.loads(PROTOCOL.read_text())
    for relative, expected in p["inputs"].items():
        assert sha(ROOT/relative) == expected, relative
    assert not OUT.exists(), "Refusing to replace saved results"
    for path in [PROTOCOL, Path(__file__).resolve()]:
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(path), "Commit protocol and script before evaluation"
    source = ROOT / "outputs/fx_extension"
    stream_index = pd.read_csv(source/"forecast_index.csv").sort_values("stream_id")
    interval_index = pd.read_csv(source/"interval_index.csv")
    with np.load(source/"issued_forecasts.npz", allow_pickle=False) as z:
        dates = z["dates"].astype("datetime64[D]")
        assets = z["assets"].tolist()
        yy, scales, std = z["returns_pct"], z["sigma"], z["train_std"]
        np.testing.assert_array_equal(z["stream_ids"], stream_index.stream_id)
    with np.load(source/"calibration_streams.npz", allow_pickle=False) as z:
        qdates = z["dates"].astype("datetime64[D]")
        qs = z["q"]
        np.testing.assert_array_equal(z["interval_ids"], interval_index.interval_id)
    evaluate = dates >= np.datetime64(p["evaluation_start"])
    qeval = qdates >= np.datetime64(p["evaluation_start"])
    np.testing.assert_array_equal(dates[evaluate], qdates[qeval])
    assert evaluate.sum() == p["evaluation_dates"]
    rules = p["rules"]
    daily = np.empty((len(stream_index), len(rules), int(evaluate.sum()), 3))
    factor_rows = []
    max_invariance = 0.
    n_min_checks = 0
    for row in stream_index.itertuples():
        sid, ai = row.stream_id, assets.index(row.asset)
        y, scale = yy[ai], scales[sid]
        factors = {"native": float(norm.ppf(1-p["alpha"]/2))}
        for year in p["fit_years"]:
            fit = (dates >= np.datetime64(f"{year}-01-01")) & (dates <= np.datetime64(f"{year}-12-31"))
            assert fit.any() and dates[fit][-1] < dates[evaluate][0]
            q, optimal = fit_factors(y[fit], scale[fit], p["alpha"])
            verify_is_minimum(y[fit], scale[fit], optimal)
            n_min_checks += 1
            factors[f"frozen_quantile_{year}"] = q
            factors[f"frozen_is_{year}"] = optimal
            for c in [.5, 2.]:
                recovered = np.array(fit_factors(y[fit], c*scale[fit], p["alpha"]))*c
                np.testing.assert_allclose(recovered, [q, optimal], rtol=1e-12, atol=1e-12)
                max_invariance = max(max_invariance, float(np.max(np.abs(recovered-[q, optimal]))))
            factor_rows.append({"stream_id": sid, "predictor": row.predictor, "seed": row.seed,
                "context": row.context, "asset": row.asset, "fit_year": year, "fit_n": int(fit.sum()),
                "fit_last": str(dates[fit][-1]), "quantile_factor": q, "is_optimal_factor": optimal})
        for ri, rule in enumerate(rules):
            if rule in factors:
                half = scale[evaluate]*factors[rule]
            else:
                window = int(rule.removeprefix("rolling"))
                iid = interval_index[(interval_index.stream_id == sid) &
                    (interval_index.calibrator == "rolling") & (interval_index.window == window)].interval_id.item()
                half = scale[evaluate]*qs[iid, qeval]
                # Recompute past-only order statistic from doubled scale, independently of saved q.
                scores = np.abs(y)/(2*scale)
                windows = np.lib.stride_tricks.sliding_window_view(scores, window)[:-1]
                rank = int(np.ceil((window+1)*(1-p["alpha"])))
                rebuilt_q = np.partition(windows, rank-1, axis=1)[:, rank-1]
                rebuilt = 2*scale[window:]*rebuilt_q
                compared = rebuilt[evaluate[window:]]
                np.testing.assert_allclose(compared, half, rtol=1e-12, atol=1e-12)
                max_invariance = max(max_invariance, float(np.max(np.abs(compared-half))))
            daily[sid, ri] = metrics(y[evaluate], half, p["alpha"])
            daily[sid, ri, :, 1:] /= std[ai]
    assert np.isfinite(daily).all()
    method_keys = list(stream_index[["predictor", "context"]].drop_duplicates().itertuples(index=False, name=None))
    grouped = np.empty((len(method_keys), len(assets), len(rules), evaluate.sum(), 3))
    for mi, (model, context) in enumerate(method_keys):
        for ai, asset in enumerate(assets):
            ids = stream_index[(stream_index.predictor == model) & (stream_index.context == context) &
                (stream_index.asset == asset)]
            assert len(ids) == (5 if model in ["gru", "tcn"] else 1)
            grouped[mi, ai] = daily[ids.stream_id].mean(axis=0)
    pooled = grouped.mean(axis=1)
    summary_rows, asset_rows, year_rows, individual_rows = [], [], [], []
    eval_years = dates[evaluate].astype("datetime64[Y]").astype(int)+1970
    for mi, (model, context) in enumerate(method_keys):
        for ri, rule in enumerate(rules):
            key = {"predictor": model, "context": context, "rule": rule}
            summary_rows.append({**key, **dict(zip(METRICS, pooled[mi, ri].mean(axis=0)))})
            for ai, asset in enumerate(assets):
                asset_rows.append({**key, "asset": asset, **dict(zip(METRICS, grouped[mi, ai, ri].mean(axis=0)))})
            for year in np.unique(eval_years):
                subset = eval_years == year
                year_rows.append({**key, "year": int(year), "days": int(subset.sum()), "partial_year": bool(year == 2026),
                    **dict(zip(METRICS, pooled[mi, ri, subset].mean(axis=0)))})
    for row in stream_index.itertuples():
        for ri, rule in enumerate(rules):
            individual_rows.append({**row._asdict(), "rule": rule,
                **dict(zip(METRICS, daily[row.stream_id, ri].mean(axis=0)))})
    # Original native/rolling summaries must reproduce the prior saved analysis.
    original = pd.read_csv(ROOT/"outputs/fx_extension_analysis/method_summary.csv")
    reproduced = 0
    for record in summary_rows:
        rule = record["rule"]
        if rule not in ["native", "rolling32", "rolling128"]:
            continue
        old = original[(original.predictor == record["predictor"]) & (original.context == record["context"]) &
                       (original.calibrator == ("native" if rule == "native" else "rolling")) &
                       (original.window == (0 if rule == "native" else int(rule[7:])))]
        for metric in METRICS:
            selected = old[(old.metric == metric) & (old.block_length == 128)]
            assert len(selected) == 1
            np.testing.assert_allclose(record[metric], selected.iloc[0].estimate, rtol=1e-12, atol=1e-12)
        reproduced += 1
    contrasts = []
    for block in p["analysis"]["block_lengths"]:
        weights = date_block_weights(int(evaluate.sum()), block, p["analysis"]["bootstrap_replicates"], p["analysis"]["bootstrap_seed"]+block)
        for left, right in [(("gru",64),("garch11",0)), (("tcn",64),("garch11",0)), (("gru",64),("tcn",64))]:
            difference = pooled[method_keys.index(left)]-pooled[method_keys.index(right)]
            for ri, rule in enumerate(rules):
                samples = weights @ difference[ri]
                low, high = np.quantile(samples, [.025,.975], axis=0)
                contrasts.append({"left": f"{left[0]}_{left[1]}", "right": f"{right[0]}_{right[1]}", "rule": rule,
                    "block": block, **{f"{m}_{suffix}": float(v) for suffix, vals in
                    [("difference",difference[ri].mean(axis=0)),("low",low),("high",high)] for m,v in zip(METRICS,vals)}})
    checks.update(exhaustive_is_minima=n_min_checks, original_summaries_reproduced=reproduced,
                  maximum_scale_invariance_error=max_invariance, all_inputs_finite=True,
                  fit_years_strictly_precede_evaluation=True, evaluation_days=int(evaluate.sum()))
    OUT.mkdir(parents=True)
    for name, records in [("summary",summary_rows),("per_asset",asset_rows),("per_year",year_rows),
                          ("per_fit_asset",individual_rows),("factors",factor_rows),("paired_comparisons",contrasts)]:
        pd.DataFrame(records).to_csv(OUT/f"{name}.csv",index=False)
    manifest = {"status": "complete", "design_status": p["status"], "freeze_commit": subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "protocol_sha256": sha(PROTOCOL),
        "script_sha256": sha(__file__), "inputs": p["inputs"], "checks": checks,
        "wall_seconds": time.time()-started, "outputs": {path.name:sha(path) for path in sorted(OUT.iterdir())}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        run()
