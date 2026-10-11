"""Public ECB reference-rate pilot with a probabilistic GRU trained from scratch.

Train/validation/test dates are fixed before test evaluation. The test split is
exploratory after this run and must not subsequently be presented as blind.
Reference rates are informational, not executable trading prices. All FX series
share EUR; use analyze_public_fx.py for paired date-block uncertainty summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "public_fx"
DATA = ROOT / "data" / "public"
URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ASSETS = ["USD", "JPY", "GBP", "CHF"]
ALPHA = .1
Q90 = float(norm.ppf(1-ALPHA/2))
FEATURE_LOOKBACK = 64


class ProbabilisticGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 16, batch_first=True)
        self.head = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, x):
        _, hidden = self.gru(x)
        z = self.head(hidden[-1])
        return z[:, 0], z[:, 1].clamp(-3, 3)


def get_data():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "ecb_reference_rates.zip"
    if not archive.exists():
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                        "--max-time", "60", URL, "-o", str(archive)], check=True)
    with zipfile.ZipFile(archive) as z:
        contents = z.read("eurofxref-hist.csv")
    frame = pd.read_csv(io.BytesIO(contents), na_values=["N/A"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")[ASSETS]
    assert frame.index.is_unique
    assert not frame.isna().any().any()
    assert (frame > 0).all().all()
    # Return for an ECB publication date uses previous available publication.
    returns = 100*np.log(frame/frame.shift(1))
    returns = returns.iloc[1:]
    returns.to_csv(DATA/"ecb_log_returns_pct.csv")
    return returns, {"provider": "European Central Bank", "source_url": URL,
                     "copyright_url": "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
                     "retrieved_date": "2026-10-11", "first_date": str(frame.index.min().date()),
                     "last_date": str(frame.index.max().date()), "rows_rates": len(frame),
                     "transformation": "100 * log(rate_t / previous_available_rate); currency units per EUR",
                     "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                     "notes": ["ECB reference rates are for information, not trading execution",
                               "historical download is not a point-in-time revision-vintage dataset",
                               "all four series share EUR and are not independent assets"]}


def build_arrays(returns):
    train_mask = returns.index <= "2018-12-31"
    train_std = returns.loc[train_mask].std(ddof=1)
    x, y, dates, symbols = [], [], [], []
    for symbol in ASSETS:
        standardized = returns[symbol].to_numpy()/train_std[symbol]
        windows = np.lib.stride_tricks.sliding_window_view(standardized, FEATURE_LOOKBACK)[:-1].copy()
        features = np.stack([windows, np.clip(windows**2, 0, 100)], axis=-1).astype(np.float32)
        x.append(features)
        y.append(standardized[FEATURE_LOOKBACK:].astype(np.float32))
        dates.extend(returns.index[FEATURE_LOOKBACK:])
        symbols.extend([symbol]*len(features))
    metadata = pd.DataFrame({"date": dates, "asset": symbols})
    return torch.from_numpy(np.concatenate(x)), torch.from_numpy(np.concatenate(y)), metadata, train_std


def nll(model, x, y, context):
    mu, log_s = model(x[:, -context:])
    return (log_s + .5*(y-mu).square()*torch.exp(-2*log_s)).mean()


def fit(x, y, metadata, seed, epochs):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = np.flatnonzero((metadata.date <= "2018-12-31").to_numpy())
    val = np.flatnonzero(((metadata.date >= "2019-01-01") & (metadata.date <= "2020-12-31")).to_numpy())
    model = ProbabilisticGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, best_val = None, float("inf")
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), 256):
            ix = order[start:start+256]
            context = int(rng.choice([16, 64]))
            loss = nll(model, x[ix], y[ix], context)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total = []
            for context in [16, 64]:
                terms = []
                for start in range(0, len(val), 512):
                    ix = val[start:start+512]
                    terms.append((len(ix), nll(model, x[ix], y[ix], context).item()))
                total.append(sum(n*l for n,l in terms)/len(val))
            val_loss = sum(total)/len(total)
        history.append({"seed": seed, "epoch": epoch+1, "val_nll": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    return model, history, best_val


def predict(model, x, context):
    centers, scales = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 512):
            mu, log_s = model(x[start:start+512, -context:])
            centers.append(mu.numpy())
            scales.append(np.exp(log_s.numpy()))
    return np.concatenate(centers), np.concatenate(scales)


def calibrate(y, mu, sigma, window, method):
    normalized = np.abs(y-mu)/sigma
    width = np.full_like(y, np.nan)
    alpha = ALPHA
    clipped = 0
    for t in range(window, len(y)):
        if method == "native":
            width[t] = Q90*sigma[t]
        else:
            a = alpha if method == "aci_clipped" else ALPHA
            a = np.clip(a, 1/(window+1), .5)
            rank = min(window, int(np.ceil((window+1)*(1-a))))
            q = np.sort(normalized[t-window:t])[rank-1]
            width[t] = q*sigma[t]
        if method == "aci_clipped":
            alpha += .01*(ALPHA-(np.abs(y[t]-mu[t]) > width[t]))
            clipped += int(alpha < 1/(window+1) or alpha > .5)
            alpha = float(np.clip(alpha, 1/(window+1), .5))
    return width, clipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", nargs="+", type=int, default=[731, 1731, 2731])
    args = parser.parse_args()
    start_time = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    returns, source = get_data()
    x, y, metadata, train_std = build_arrays(returns)
    rows, histories, fits = [], [], []
    for seed in args.seeds:
        model, history, val_loss = fit(x, y, metadata, seed, args.epochs)
        histories.extend(history)
        fits.append({"seed": seed, "best_validation_nll": val_loss})
        torch.save(model.state_dict(), OUT/f"gru_seed_{seed}.pt")
        for context in [16, 64]:
            centers, scales = predict(model, x, context)
            for symbol in ASSETS:
                ix = np.flatnonzero((metadata.asset == symbol).to_numpy())
                actual = y[ix].numpy()*train_std[symbol]
                mu = centers[ix]*train_std[symbol]
                sigma = scales[ix]*train_std[symbol]
                dates = metadata.date.iloc[ix].to_numpy()
                # Begin online replay only after the fitted training period.
                begin = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
                actual, mu, sigma, dates = actual[begin:], mu[begin:], sigma[begin:], dates[begin:]
                for method, window in [("native", 128), ("rolling", 32), ("rolling", 128), ("aci_clipped", 128)]:
                    half, clipped = calibrate(actual, mu, sigma, window, method)
                    for i in np.flatnonzero(dates >= np.datetime64("2021-01-01")):
                        miss = max(abs(actual[i]-mu[i])-half[i], 0)
                        rows.append({"date": str(pd.Timestamp(dates[i]).date()), "asset": symbol,
                                     "seed": seed, "predictor": "gru", "context": context,
                                     "calibrator": method, "window": window,
                                     "actual": actual[i], "mu": mu[i], "sigma": sigma[i],
                                     "half_width": half[i], "hit": int(miss == 0),
                                     "interval_score": 2*half[i]+2/ALPHA*miss,
                                     "clipping_total_final": clipped})
        print(f"seed {seed}: best validation NLL={val_loss:.4f}", flush=True)
    # Statistical scale baseline uses the same training standardization.
    for symbol in ASSETS:
        r = returns[symbol].to_numpy()
        dates = returns.index.to_numpy()
        for memory in [16, 64]:
            variance = float(train_std[symbol]**2)
            sigma = np.empty_like(r)
            for i in range(len(r)):
                sigma[i] = np.sqrt(max(variance, 1e-8))
                variance += (r[i]**2-variance)/memory
            for method, window in [("native", 128), ("rolling", 32), ("rolling", 128), ("aci_clipped", 128)]:
                begin = int(np.searchsorted(dates, np.datetime64("2019-01-01")))
                active_r, active_sigma, active_dates = r[begin:], sigma[begin:], dates[begin:]
                half, clipped = calibrate(active_r, np.zeros_like(active_r), active_sigma, window, method)
                for i in np.flatnonzero(active_dates >= np.datetime64("2021-01-01")):
                    miss = max(abs(active_r[i])-half[i], 0)
                    rows.append({"date": str(pd.Timestamp(active_dates[i]).date()), "asset": symbol,
                                 "seed": -1, "predictor": "ewma", "context": memory,
                                 "calibrator": method, "window": window, "actual": active_r[i],
                                 "mu": 0., "sigma": active_sigma[i], "half_width": half[i],
                                 "hit": int(miss == 0), "interval_score": 2*half[i]+2/ALPHA*miss,
                                 "clipping_total_final": clipped})
    predictions = pd.DataFrame(rows)
    assert not predictions.isna().any().any()
    assert np.isfinite(predictions.select_dtypes(include="number").to_numpy()).all()
    predictions.to_csv(OUT/"predictions.csv", index=False)
    summary = predictions.groupby(["predictor", "seed", "context", "calibrator", "window", "asset"]).agg(
        n=("hit", "size"), coverage=("hit", "mean"), half_width=("half_width", "mean"),
        interval_score=("interval_score", "mean")).reset_index()
    summary["width"] = 2*summary.half_width
    summary.to_csv(OUT/"summary_by_asset_seed.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT/"training_history.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT/"validation_fits.csv", index=False)
    # A future-outcome intervention cannot change intervals already issued.
    probe_y = y[:500].numpy().copy()
    probe_mu = np.zeros_like(probe_y)
    probe_sigma = np.ones_like(probe_y)
    changed_y = probe_y.copy()
    changed_y[350:] += 1000
    for method in ["rolling", "aci_clipped"]:
        before, _ = calibrate(probe_y, probe_mu, probe_sigma, 128, method)
        after, _ = calibrate(changed_y, probe_mu, probe_sigma, 128, method)
        assert np.array_equal(before[128:351], after[128:351])
    manifest = {"date": "2026-10-11", "data": source, "train_end": "2018-12-31",
                "validation": ["2019-01-01", "2020-12-31"], "test_start": "2021-01-01",
                "test_status": "exploratory after first evaluation; no subsequent blind-test claim",
                "calibration_replay_start": "2019-01-01",
                "calibration_initialization": "replay validation period after checkpoint selection, then evaluate 2021+; not contemporaneous validation deployment",
                "aci": {"gamma": .01, "alpha_min": 1/129, "alpha_max": .5, "initial_alpha": ALPHA},
                "causal_calibration_future_suffix_check": True,
                "training_scales": train_std.to_dict(), "parameters": sum(p.numel() for p in model.parameters()),
                "neural_training_seeds": args.seeds, "epochs_max": args.epochs,
                "checkpoint_sha256": {str(s): hashlib.sha256((OUT/f"gru_seed_{s}.pt").read_bytes()).hexdigest() for s in args.seeds},
                "contexts": [16,64], "nominal_coverage": .9, "units": "percentage log-return",
                "weight_sharing": "same fitted GRU used for both context lengths; training randomly mixes lengths",
                "limitations": ["pilot only; no GARCH/AgACI/SAOCP/NCC comparison yet",
                                "GRU frozen after training; calibrators update sequentially after outcomes arrive",
                                "ACI uses explicit clipping and is not exact theoretical ACI",
                                "no true market regime labels; no causal shock attribution",
                                "cross-asset dependence and temporal dependence prohibit IID confidence intervals",
                                "normal likelihood may be misspecified for heavy tails",
                                "raw ECB history is not point-in-time revision data"],
                "wall_seconds": time.time()-start_time, "python": platform.python_version(),
                "torch": torch.__version__, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"seeds": args.seeds, "test_prediction_rows": len(predictions),
                      "last_date": source["last_date"], "wall_seconds": manifest["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
