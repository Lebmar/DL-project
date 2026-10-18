"""Frozen local-CPU ECB extension; ten fits, all currencies and paired contexts.

The 2021+ history has already been inspected. This is exploratory evaluation,
not a new blind test. No download, HPC access, test tuning, or silent overwrite.
--self-check uses synthetic arrays only and does not open the ECB data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import torch

import stage2_models as neural
import stage2_baselines as baseline

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/fx_extension_protocol.json"
OUT = ROOT / "outputs/fx_extension"
AUTHOR = ROOT / "outputs/strong_calibration/vendor/online_conformal"
NEW_FILES = ["configs/fx_extension_protocol.json", "scripts/run_fx_extension.py",
             "scripts/analyze_fx_extension.py", "research/fx_extension_protocol.md"]
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
METRICS = ("coverage", "width", "interval_score")
STREAM_KEYS = ["predictor", "seed", "context", "asset"]
INTERVAL_KEYS = STREAM_KEYS + ["calibrator", "window"]
COMPONENTS = ("total", "current_at_archive64", "archive_at_current16",
              "current_at_archive16", "archive_at_current64", "third_order")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def committed_files():
    hashes = {}
    for relative in NEW_FILES:
        content = subprocess.check_output(["git", "show", "HEAD:"+relative], cwd=ROOT)
        hashes[relative] = sha(ROOT / relative)
        assert hashlib.sha256(content).hexdigest() == hashes[relative], "Commit before execution: "+relative
    return hashes


def load_protocol():
    p = json.loads(PROTOCOL.read_text())
    assert p["frozen"] and not p["new_blind_test"]
    t, c, e = p["training"], p["calibration"], p["execution"]
    assert t["architectures"] == ["gru", "tcn"] and t["seeds"] == list(range(4301, 4306))
    assert t["contexts"] == list(CONTEXTS) and t["model"] == neural.DEFAULT_MODEL_CONFIG
    assert t["parameter_counts"] == {"gru": 2041, "tcn": 1897}
    assert {k: t[k] for k in ("batch_size", "batches_per_epoch", "max_epochs", "min_epochs", "patience",
        "min_delta", "learning_rate", "gradient_clip_norm", "validation_batch_size")} == {
        "batch_size": 128, "batches_per_epoch": 32, "max_epochs": 60, "min_epochs": 20, "patience": 8,
        "min_delta": .0001, "learning_rate": .002, "gradient_clip_norm": 5.0, "validation_batch_size": 512}
    assert p["data"]["assets"] == ["USD", "JPY", "GBP", "CHF"]
    assert (p["data"]["train_end"], p["data"]["checkpoint_start"], p["data"]["checkpoint_end"],
            p["data"]["calibration_replay_start"], p["data"]["evaluation_start"]) == (
            "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01", "2021-01-01")
    assert c["alpha"] == .1 and c["windows"] == list(WINDOWS) and c["aci_gamma"] == .02
    assert c["aci_window"] == 128 and c["saocp_lifetime"] == 8 and not c["score_clipping"]
    assert c["aci_clip"] == [1/129, .5] and c["aci_initial_alpha"] == .1
    assert c["all_stream_methods"] == ["native", "rolling32", "rolling128", "aci128"]
    assert c["strong_methods"] == ["sfogd", "saocp"]
    assert p["classical"]["models"] == ["constant", "ewma16", "ewma64", "garch11"]
    assert e["training_device"] == e["prediction_device"] == "cpu" and e["torch_threads"] == 4
    assert e["output"] == str(OUT.relative_to(ROOT))
    for relative, value in p["frozen_dependencies"].items():
        assert sha(ROOT / relative) == value, relative
    return p


def read_data(p):
    spec = p["data"]
    path = ROOT / spec["archive"]
    assert sha(path) == spec["archive_sha256"], "Frozen ECB archive mismatch; never download a replacement"
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read(spec["member"])), na_values=["N/A"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    rates = frame.sort_values("Date").set_index("Date")[spec["assets"]]
    assert rates.index.is_unique and rates.index.is_monotonic_increasing
    assert len(rates) == spec["rate_rows"]
    assert str(rates.index[0].date()) == spec["rate_first"] and str(rates.index[-1].date()) == spec["rate_last"]
    assert np.isfinite(rates.to_numpy()).all() and (rates.to_numpy() > 0).all()
    returns = (100*np.log(rates/rates.shift(1))).iloc[1:]
    train = returns.index <= spec["train_end"]
    std = returns.loc[train].std(ddof=1)
    assert np.isfinite(std).all() and (std > 0).all()
    raw = returns.to_numpy(dtype=np.float64).T.copy()
    standardized = raw / std.to_numpy()[:, None]
    dates = returns.index.to_numpy(dtype="datetime64[D]")
    masks = {
        "train": dates <= np.datetime64(spec["train_end"]),
        "checkpoint": (dates >= np.datetime64(spec["checkpoint_start"])) & (dates <= np.datetime64(spec["checkpoint_end"])),
        "replay": dates >= np.datetime64(spec["calibration_replay_start"]),
        "evaluation": dates >= np.datetime64(spec["evaluation_start"]),
        "forecast": dates >= np.datetime64(spec["checkpoint_start"]),
    }
    assert masks["evaluation"].sum() == spec["evaluation_dates"]
    assert str(dates[masks["evaluation"]][0]) == spec["evaluation_first_observed"]
    assert not (masks["train"] & masks["checkpoint"]).any()
    assert not (masks["checkpoint"] & masks["replay"]).any()
    assert masks["replay"].sum()-masks["evaluation"].sum() >= 128
    metadata = {
        "archive_sha256": sha(path), "assets": spec["assets"],
        "return_rows_per_asset": len(dates), "train_std_ddof1": dict(zip(spec["assets"], std.tolist())),
        "return_array_sha256": array_sha(raw), "standardized_array_sha256": array_sha(standardized),
        "train_standardized_sha256": array_sha(standardized[:, masks["train"]]),
        "splits": {name: {"first": str(dates[mask][0]), "last": str(dates[mask][-1]), "dates_per_asset": int(mask.sum())}
                   for name, mask in masks.items()},
        "neural_training_targets_per_asset": int(masks["train"].sum())-64,
        "archive_contains_future_rows": True,
        "fitting_inputs": "only targets through2018 and context windows ending before those targets; checkpoint selection only2019",
        "notes": ["No download or writes to source data", "No weekend imputation",
                  "Historical reference-rate snapshot, not real-time revision vintages",
                  "Shared EUR denominator and calendar create asset dependence"],
    }
    return raw, standardized, dates, std.to_numpy(), masks, metadata


def context_tensor(observations, assets, targets, context):
    indices = targets[:, None] + np.arange(-context, 0)[None, :]
    assert indices.min() >= 0
    return torch.tensor(observations[assets[:, None], indices], dtype=torch.float32)


def sampling_epoch(rng, asset_count, target_stop, opt):
    schedule = rng.permutation(np.resize(np.array(CONTEXTS), opt["batches_per_epoch"]))
    for context in schedule:
        assets = rng.integers(0, asset_count, opt["batch_size"])
        targets = rng.integers(64, target_stop, opt["batch_size"])
        yield int(context), assets, targets


def validation_nll(model, observations, target_indices, batch_size):
    assets = np.repeat(np.arange(len(observations)), len(target_indices))
    targets = np.tile(target_indices, len(observations))
    result = {}
    model.eval()
    with torch.no_grad():
        for context in CONTEXTS:
            total = 0.
            for begin in range(0, len(targets), batch_size):
                aa, tt = assets[begin:begin+batch_size], targets[begin:begin+batch_size]
                y = torch.tensor(observations[aa, tt], dtype=torch.float32)
                loss = neural.gaussian_nll(model(context_tensor(observations, aa, tt, context)), y)
                assert torch.isfinite(loss)
                total += float(loss)*len(tt)
            result[context] = total/len(targets)
    return result


def fit_one(architecture, seed, p, standardized, train_stop, validation_targets, protocol_hash):
    opt = p["training"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = neural.make_model(architecture, opt["model"]).float()
    assert sum(v.numel() for v in model.parameters()) == opt["parameter_counts"][architecture]
    before = neural.causal_prefix_audit(model, seed=674125101)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["learning_rate"])
    history, best, patience_reference, stale = [], float("inf"), float("inf"), 0
    checkpoint = OUT / "checkpoints" / f"{architecture}_seed_{seed}.pt"
    started = time.time()
    for epoch in range(1, opt["max_epochs"]+1):
        model.train()
        digest, losses = hashlib.sha256(), []
        for context, assets, targets in sampling_epoch(rng, len(standardized), train_stop, opt):
            digest.update(np.array([context], dtype="<i8").tobytes())
            digest.update(assets.astype("<i8").tobytes())
            digest.update(targets.astype("<i8").tobytes())
            y = torch.tensor(standardized[assets, targets], dtype=torch.float32)
            loss = neural.gaussian_nll(model(context_tensor(standardized, assets, targets, context)), y)
            assert torch.isfinite(loss), "Nonfinite training loss; no model exclusion permitted"
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["gradient_clip_norm"])
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in model.parameters())
            losses.append(float(loss.detach()))
        validation = validation_nll(model, standardized, validation_targets, opt["validation_batch_size"])
        mean = float(np.mean(list(validation.values())))
        if mean < best:
            best, selected_epoch = mean, epoch
            torch.save({"architecture": architecture, "model_seed": seed, "model_config": opt["model"],
                "contexts": list(CONTEXTS), "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "selected_epoch": epoch, "best_validation_mean_nll": best,
                "validation_nll_by_context": validation, "protocol_sha256": protocol_hash,
                "training_device": "cpu", "prediction_center": 0.0}, checkpoint)
        if mean < patience_reference-opt["min_delta"]:
            patience_reference, stale = mean, 0
        else:
            stale += 1
        history.append({"architecture": architecture, "seed": seed, "epoch": epoch,
            "train_nll_without_constant": float(np.mean(losses)),
            "validation_nll_context16": validation[16], "validation_nll_context64": validation[64],
            "mean_validation_nll": mean, "best_validation_nll": best,
            "patience_counter": stale, "minibatch_epoch_sha256": digest.hexdigest(),
            "elapsed_seconds": time.time()-started})
        if epoch % 5 == 0:
            print(json.dumps({"phase": "training", "architecture": architecture, "seed": seed,
                              "epoch": epoch, "validation_nll": mean}), flush=True)
        if epoch >= opt["min_epochs"] and stale >= opt["patience"]:
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    after = neural.causal_prefix_audit(model, seed=674125102)
    reproduced = validation_nll(model, standardized, validation_targets, opt["validation_batch_size"])
    np.testing.assert_allclose(list(reproduced.values()), list(saved["validation_nll_by_context"].values()), rtol=0, atol=0)
    pd.DataFrame(history).to_csv(OUT / "history" / f"{architecture}_seed_{seed}.csv", index=False)
    return {"architecture": architecture, "seed": seed, "epochs_executed": len(history),
        "selected_epoch": selected_epoch, "best_validation_mean_nll": best,
        "parameter_count": sum(v.numel() for v in model.parameters()),
        "checkpoint": str(checkpoint.relative_to(OUT)), "checkpoint_sha256": sha(checkpoint),
        "minibatch_epoch_sha256": [row["minibatch_epoch_sha256"] for row in history],
        "causal_audit_before": before, "causal_audit_selected": after,
        "validation_reproduced_exactly": True, "wall_seconds": time.time()-started,
        "stop_reason": "max_epochs" if len(history) == opt["max_epochs"] else "validation_patience"}


def predict_scales(model, standardized, target_indices, context, batch_size=512):
    """Reset at each exact past context; never use current/future target values."""
    result = np.empty((len(standardized), len(target_indices)), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for asset in range(len(standardized)):
            for begin in range(0, len(target_indices), batch_size):
                targets = target_indices[begin:begin+batch_size]
                assets = np.full(len(targets), asset)
                result[asset, begin:begin+len(targets)] = model(
                    context_tensor(standardized, assets, targets, context)).exp().numpy()
    assert np.isfinite(result).all() and (result > 0).all()
    return result


def expected_streams(p):
    rows = []
    for architecture in p["training"]["architectures"]:
        for seed in p["training"]["seeds"]:
            for context in CONTEXTS:
                for asset in p["data"]["assets"]:
                    rows.append(dict(predictor=architecture, seed=seed, context=context, asset=asset))
    for predictor in p["classical"]["models"]:
        for asset in p["data"]["assets"]:
            rows.append(dict(predictor=predictor, seed=-1, context=0, asset=asset))
    frame = pd.DataFrame(rows)
    frame.insert(0, "stream_id", np.arange(len(frame)))
    assert len(frame) == 96 and not frame.duplicated(STREAM_KEYS).any()
    return frame


def expected_intervals(streams):
    rows = []
    for row in streams.to_dict("records"):
        methods = [("native", 0), ("rolling", 32), ("rolling", 128), ("aci", 128)]
        if row["context"] == 64 or row["seed"] == -1:
            methods += [("sfogd", 0), ("saocp", 0)]
        for method, window in methods:
            rows.append({**row, "calibrator": method, "window": window})
    frame = pd.DataFrame(rows)
    frame.insert(0, "interval_id", np.arange(len(frame)))
    assert len(frame) == 496 and not frame.duplicated(INTERVAL_KEYS).any()
    return frame


def stream_quantiles(y, sigma, validation_mask, replay_mask, method, window, alpha=.1, gamma=.02):
    """One asset/fit/context only; no shared-asset warmup broadcast."""
    assert y.ndim == sigma.ndim == 1 and y.shape == sigma.shape
    assert validation_mask.any() and replay_mask.any() and not (validation_mask & replay_mask).any()
    assert np.flatnonzero(validation_mask)[-1] < np.flatnonzero(replay_mask)[0]
    maximum = float(max(np.max(np.abs(y[validation_mask])/sigma[validation_mask])*np.sqrt(3), 1e-12))
    yy, ss = y[replay_mask], sigma[replay_mask]
    if method == "native":
        return np.full(len(yy), norm.ppf(1-alpha/2)), {
            "max_scale": maximum, "method": "native", "score_truncation": False}
    spec = {"alpha": alpha, "gamma": gamma, "initial_window": 128, "max_scale": maximum}
    calibrated = baseline.calibrate_paths(yy, ss, spec, method, window if window else 128)
    meta = dict(calibrated["metadata"])
    if meta["author_source"]:
        meta["author_source"] = {**meta["author_source"], "vendor_path": str(AUTHOR.relative_to(ROOT))}
    meta["initialization"] = "asset-specific chronological replay from2020; no external/common warmup; continue through2021+"
    meta["max_scale_source"] = "same asset/fit/context 2019 normalized scores, sqrt(3)*maximum"
    return calibrated["q"], meta


def metrics(y, half, alpha=.1):
    return np.stack(((np.abs(y) <= half).astype(float), 2*half,
                     2*half+(2/alpha)*np.maximum(np.abs(y)-half, 0)), axis=-1)


def independent_rolling_q(y, sigma, window):
    """Separate previous-window indexing and sorted-order-statistic check."""
    scores = np.abs(y)/sigma
    q = np.full(len(y), np.nan)
    views = np.lib.stride_tricks.sliding_window_view(scores, window)[:-1]
    rank = int(np.ceil((window+1)*.9))
    q[window:] = np.sort(views, axis=1)[:, rank-1]
    return q


def decompose_fx(cells):
    """Input [..., current2, archive2, window2, dates, metrics]; no control world."""
    d = cells[..., 1, :, :]-cells[..., 0, :, :]
    d00, d01, d10, d11 = d[..., 0, 0, :, :], d[..., 0, 1, :, :], d[..., 1, 0, :, :], d[..., 1, 1, :, :]
    result = np.stack((d11-d00, d11-d01, d01-d00, d10-d00, d11-d10,
                       d11-d01-d10+d00), axis=-3)
    np.testing.assert_allclose(result[..., 0, :, :], result[..., 1, :, :]+result[..., 2, :, :], rtol=0, atol=1e-10)
    np.testing.assert_allclose(result[..., 0, :, :], result[..., 3, :, :]+result[..., 4, :, :], rtol=0, atol=1e-10)
    return result


def tiny_self_check():
    """Synthetic-only causal, own-asset replay, sampler and decomposition tests."""
    torch.set_num_threads(4)
    rng = np.random.default_rng(674125201)
    y = rng.normal(size=(4, 480))
    changed = y.copy()
    changed[:, 360:] += 40
    max_error = 0.
    for architecture in ("gru", "tcn"):
        torch.manual_seed(674125202)
        model = neural.make_model(architecture)
        neural.causal_prefix_audit(model, seed=674125203)
        for context in CONTEXTS:
            indices = np.arange(64, 480)
            before = predict_scales(model, y, indices, context)
            after = predict_scales(model, changed, indices, context)
            error = float(np.max(np.abs(before[:, indices <= 360]-after[:, indices <= 360])))
            assert error <= 2e-6
            max_error = max(max_error, error)
    vmask = np.arange(480) < 160
    rmask = ~vmask
    methods_checked = []
    for method, window in (("rolling", 32), ("rolling", 128), ("aci", 128), ("sfogd", 0), ("saocp", 0)):
        sig = np.exp(rng.normal(0, .2, 480))
        q, _ = stream_quantiles(y[0], sig, vmask, rmask, method, window)
        qq, _ = stream_quantiles(changed[0], sig, vmask, rmask, method, window)
        np.testing.assert_array_equal(q[128:201], qq[128:201])
        if method == "rolling":
            independent = independent_rolling_q(y[0,rmask],sig[rmask],window)
            np.testing.assert_array_equal(q[128:],independent[128:])
        if method == "aci":
            score = np.abs(y[0,rmask])/sig[rmask]
            expected, adaptive = np.full(len(score),np.nan), .1
            for t in range(128,len(score)):
                k = min(128,max(1,int(np.ceil(129*(1-adaptive)))))
                expected[t] = sorted(score[t-128:t])[k-1]
                adaptive = min(.5,max(1/129,adaptive+.02*(.1-(score[t] > expected[t]))))
            np.testing.assert_array_equal(q[128:],expected[128:])
        methods_checked.append([method, window])
    # Own-asset archive: constant positive outcomes lead to distinct expected q.
    own_q = []
    for multiplier in (1., 9.):
        q, _ = stream_quantiles(np.full(480, multiplier), np.ones(480), vmask, rmask, "rolling", 128)
        np.testing.assert_array_equal(q[128:], np.full(192, multiplier))
        own_q.append(float(q[-1]))
    # Independent scalar recursion checks full-history classical forecasting.
    for name, parameters in (("constant", {}), ("ewma16", {"weight": 1/16}),
                              ("garch11", {"omega": .03, "alpha": .08, "beta": .86})):
        model = baseline.ScaleModel(name, 1.2, parameters, {})
        a, b = model.scale_paths(y[0]), model.scale_paths(changed[0])
        np.testing.assert_array_equal(a[:361], b[:361])
        variance = np.empty(480)
        variance[0] = 1.2
        for t in range(1, 480):
            variance[t] = (1.2 if name == "constant" else
                (15/16*variance[t-1]+y[0,t-1]**2/16 if name == "ewma16" else
                 .03+.08*y[0,t-1]**2+.86*variance[t-1]))
        np.testing.assert_allclose(a*a, variance, rtol=1e-13, atol=1e-13)
    opt = {"batches_per_epoch": 32, "batch_size": 128}
    schedules = [list(sampling_epoch(np.random.default_rng(4301), 4, 400, opt)) for _ in range(2)]
    for one, two in zip(*schedules):
        assert one[0] == two[0]
        np.testing.assert_array_equal(one[1], two[1])
        np.testing.assert_array_equal(one[2], two[2])
    assert [x[0] for x in schedules[0]].count(16) == 16
    toy = rng.random((2, 4, 2, 2, 2, 11, 3))
    parts = decompose_fx(toy)
    direct = toy[..., 1,1,1,:,:]-toy[..., 1,1,0,:,:]-toy[..., 0,0,1,:,:]+toy[..., 0,0,0,:,:]
    np.testing.assert_allclose(parts[..., 0,:,:], direct, atol=1e-14)
    fake = {"training": {"architectures": ["gru","tcn"], "seeds": list(range(4301,4306))},
            "data": {"assets": ["USD","JPY","GBP","CHF"]},
            "classical": {"models": ["constant","ewma16","ewma64","garch11"]}}
    streams = expected_streams(fake)
    intervals = expected_intervals(streams)
    assert len(intervals.query("calibrator == 'sfogd'")) == 56
    return {"passed": True, "synthetic_only": True, "causal_scale_max_abs_error": max_error,
        "current_future_calibration_exclusion_methods": methods_checked, "distinct_asset_warmups": own_q,
        "classical_scalar_recursions": True, "matched_balanced_sampling": True,
        "independent_sorted_rolling_and_scalar_ACI": True,
        "factorial_routes": True, "stream_count": len(streams), "interval_count": len(intervals)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check(), indent=2))
        return
    sources = committed_files()
    p = load_protocol()
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError("Refusing any existing FX extension directory")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    raw, standardized, dates, std, masks, data = read_data(p)
    OUT.mkdir(parents=True)
    (OUT / "checkpoints").mkdir()
    (OUT / "history").mkdir()
    (OUT / "protocol.json").write_bytes(PROTOCOL.read_bytes())
    manifest = {"status": "training", "design_status": p["design_status"], "new_blind_test": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha(PROTOCOL),
        "new_sources": sources, "frozen_dependencies": p["frozen_dependencies"],
        "commit_before_execution": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "data": data, "training_device": "cpu", "prediction_device": "cpu", "paid_HPC_used": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "torch": str(torch.__version__)},
        "runtime": {"platform": platform.system(), "machine": platform.machine(), "threads": torch.get_num_threads(),
                    "interop_threads": torch.get_num_interop_threads(), "dtype": "float32", "deterministic_algorithms": True,
                    "float32_matmul_precision": "highest", "tf32": False},
        "self_check": tiny_self_check(), "models": [], "test_outcomes_used_for_fitting_or_selection": False,
        "evaluation_predictions_started": False}
    save_manifest = lambda: (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    save_manifest()
    validation_targets = np.flatnonzero(masks["checkpoint"])
    train_stop = int(masks["train"].sum())
    for architecture in p["training"]["architectures"]:
        for seed in p["training"]["seeds"]:
            record = fit_one(architecture, seed, p, standardized, train_stop, validation_targets, sha(PROTOCOL))
            manifest["models"].append(record)
            save_manifest()
    for seed in p["training"]["seeds"]:
        paired = [r for r in manifest["models"] if r["seed"] == seed]
        a, b = (r["minibatch_epoch_sha256"] for r in paired)
        assert a[:min(len(a),len(b))] == b[:min(len(a),len(b))]
    manifest["matching_minibatch_prefixes_verified"] = True
    classical, classical_scales = {}, {}
    for asset_index, asset in enumerate(p["data"]["assets"]):
        models = baseline.fit_baselines(raw[asset_index, masks["train"]])
        classical[asset] = {name: {"name": model.name, "variance_init": model.variance_init,
            "parameters": model.parameters, "metadata": model.metadata} for name,model in models.items()}
        for name, model in models.items():
            classical_scales[(name,asset)] = model.scale_paths(raw[asset_index])
    (OUT / "classical_models.json").write_text(json.dumps(classical, indent=2, allow_nan=False)+"\n")
    manifest["status"], manifest["evaluation_predictions_started"] = "evaluating", True
    save_manifest()
    target_indices = np.flatnonzero(masks["forecast"])
    forecast_dates = dates[masks["forecast"]]
    forecast_returns = raw[:, masks["forecast"]]
    streams = expected_streams(p)
    sigmas = np.full((len(streams),len(target_indices)), np.nan)
    for record in manifest["models"]:
        model, saved = neural.load_checkpoint(OUT / record["checkpoint"])
        assert saved["protocol_sha256"] == sha(PROTOCOL)
        assert sha(OUT / record["checkpoint"]) == record["checkpoint_sha256"]
        for context in CONTEXTS:
            scale = predict_scales(model, standardized, target_indices, context)*std[:,None]
            for ai, asset in enumerate(p["data"]["assets"]):
                sid = streams.index[(streams.predictor == record["architecture"]) & (streams.seed == record["seed"]) &
                                    (streams.context == context) & (streams.asset == asset)].item()
                sigmas[sid] = scale[ai]
    for row in streams[streams.seed == -1].itertuples():
        sigmas[row.stream_id] = classical_scales[(row.predictor,row.asset)][masks["forecast"]]
    assert np.isfinite(sigmas).all() and (sigmas > 0).all()
    validation_mask = forecast_dates <= np.datetime64(p["data"]["checkpoint_end"])
    replay_mask = forecast_dates >= np.datetime64(p["data"]["calibration_replay_start"])
    replay_dates = forecast_dates[replay_mask]
    evaluation_mask = replay_dates >= np.datetime64(p["data"]["evaluation_start"])
    intervals = expected_intervals(streams)
    qs = np.full((len(intervals),len(replay_dates)), np.nan)
    calibration_records = []
    for row in intervals.itertuples():
        ai = p["data"]["assets"].index(row.asset)
        q, metadata = stream_quantiles(forecast_returns[ai], sigmas[row.stream_id],
            validation_mask, replay_mask, row.calibrator, row.window, p["calibration"]["alpha"], p["calibration"]["aci_gamma"])
        qs[row.interval_id] = q
        assert np.isfinite(q[evaluation_mask]).all() and (q[evaluation_mask] >= 0).all()
        calibration_records.append({"interval_id": row.interval_id, **metadata})
    np.savez_compressed(OUT / "issued_forecasts.npz", dates=forecast_dates.astype(str),
        assets=np.array(p["data"]["assets"]), returns_pct=forecast_returns, train_std=std,
        sigma=sigmas, stream_ids=streams.stream_id.to_numpy())
    np.savez_compressed(OUT / "calibration_streams.npz", dates=replay_dates.astype(str),
        q=qs, interval_ids=intervals.interval_id.to_numpy())
    streams.to_csv(OUT / "forecast_index.csv", index=False)
    intervals.to_csv(OUT / "interval_index.csv", index=False)
    (OUT / "calibration_metadata.json").write_text(json.dumps(calibration_records, indent=2, allow_nan=False)+"\n")
    # Compact explicit eight-cell daily metric grid, independently reconstructible from sigma and q.
    model_keys = [(a,s) for a in p["training"]["architectures"] for s in p["training"]["seeds"]]
    values = np.empty((10,4,2,2,2,int(evaluation_mask.sum()),3), dtype=np.float64)
    index = intervals.set_index(INTERVAL_KEYS)
    diagonal_error = 0.
    for mi,(architecture,seed) in enumerate(model_keys):
        for ai,asset in enumerate(p["data"]["assets"]):
            yy = forecast_returns[ai,replay_mask][evaluation_mask]
            for ci,a in enumerate(CONTEXTS):
                sid = int(index.loc[(architecture,seed,a,asset,"native",0),"stream_id"])
                current = sigmas[sid,replay_mask][evaluation_mask]
                for bi,b in enumerate(CONTEXTS):
                    for wi,window in enumerate(WINDOWS):
                        ix = int(index.loc[(architecture,seed,b,asset,"rolling",window),"interval_id"])
                        half = current*qs[ix,evaluation_mask]
                        values[mi,ai,ci,bi,wi] = metrics(yy,half)
                        if a == b:
                            independently_issued = independent_rolling_q(
                                forecast_returns[ai,replay_mask],sigmas[sid,replay_mask],window)
                            np.testing.assert_array_equal(qs[ix,evaluation_mask],independently_issued[evaluation_mask])
                            diagonal = sigmas[sid,replay_mask][evaluation_mask]*independently_issued[evaluation_mask]
                            diagonal_error = max(diagonal_error,float(np.max(np.abs(half-diagonal))))
                            np.testing.assert_array_equal(values[mi,ai,ci,bi,wi],metrics(yy,diagonal))
    assert diagonal_error == 0 and np.isfinite(values).all()
    decompose_fx(values)
    np.savez_compressed(OUT / "crossed_metrics.npz", values=values, dates=replay_dates[evaluation_mask].astype(str),
        model_architectures=np.array([k[0] for k in model_keys]), model_seeds=np.array([k[1] for k in model_keys]),
        assets=np.array(p["data"]["assets"]), current_contexts=np.array(CONTEXTS),
        archive_contexts=np.array(CONTEXTS), windows=np.array(WINDOWS), metrics=np.array(METRICS))
    assert load_protocol() == p and committed_files() == sources
    assert sha(ROOT / p["data"]["archive"]) == p["data"]["archive_sha256"]
    for record in manifest["models"]:
        assert sha(OUT / record["checkpoint"]) == record["checkpoint_sha256"]
    files = [path for path in OUT.rglob("*") if path.is_file() and path.name != "manifest.json"]
    manifest.update(status="complete", models_completed=len(manifest["models"]), stream_count=len(streams),
        interval_count=len(intervals), test_dates=int(evaluation_mask.sum()), crossed_shape=list(values.shape),
        diagonal_reproduction_max_abs_error=diagonal_error, elapsed_seconds=time.time()-started,
        artifacts={str(path.relative_to(OUT)):sha(path) for path in sorted(files)})
    save_manifest()
    print(json.dumps({"status":"complete","models":10,"dates":int(evaluation_mask.sum()),
                      "elapsed_seconds":manifest["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
