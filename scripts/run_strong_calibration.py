"""Author-source SF-OGD/SAOCP on already-issued public ECB GRU forecasts.

The fixed Apache-2.0 author source is executed without algorithm edits. An AST
loader omits imports and unused ensemble classes, and supplies forbidden
Merlion type bindings. Only the documented model=None, external-score interface
is supported. This is an explicit source adapter, not a full package install.
No test-based hyperparameter selection or theoretical coverage claim is made.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "strong_calibration"
VENDOR = OUT / "vendor" / "online_conformal"
COMMIT = "95790a9b9e4061166e6d7a8a13282e019b1a8663"
AUTHOR_REPO = "https://github.com/salesforce/online_conformal"
COVERAGE = .9
ALPHA = 1-COVERAGE
SOURCE_HASHES = {
    "online_conformal/base.py": "83999e0444ef8868edd5235a17e806e4cbbab0ae1f94f977e6fcf964c65bd2f2",
    "online_conformal/ogd.py": "061604484f10f8a0465d45d69b80a441acfc41a46b1965378355110e27963b5d",
    "online_conformal/saocp.py": "83f8845559786452aafabfafdf6c136ceaf9b9b3557fc5f8f9267fcec04d386f",
    "online_conformal/utils.py": "88c8e7725f114ccd5c68ebd4adcd6804c523aa241fa62f2b29c2a95354d83fa1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UnavailableMerlion:
    """Forbidden binding used only to satisfy author type annotations."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Merlion forecasting is outside this source adapter")

    @staticmethod
    def from_pd(*args, **kwargs):
        raise RuntimeError("Merlion forecasting is outside this source adapter")


class UnavailableProgress:
    @staticmethod
    def trange(*args, **kwargs):
        raise RuntimeError("Author forecast() is disabled; use external scores")


def author_algorithms():
    VENDOR.mkdir(parents=True, exist_ok=True)
    for rel in [*SOURCE_HASHES, "LICENSE", "README.md"]:
        path = VENDOR / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/salesforce/online_conformal/{COMMIT}/{rel}"
            with urllib.request.urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        if rel in SOURCE_HASHES:
            assert sha(path) == SOURCE_HASHES[rel], f"Author source hash mismatch: {rel}"
    (VENDOR / "PINNED_COMMIT.txt").write_text(COMMIT+"\n", encoding="utf-8")
    (VENDOR / "ADAPTATION.md").write_text(
        "Source adapter for model=None, horizon=1 external scores only.\n"
        "Original source files are unmodified; four algorithms/support files are hash-checked.\n"
        "AST execution selects full original utility functions/classes, BasePredictor,\n"
        "ScaleFreeOGD, _OGD, and SAOCP; skips original import statements and Enb classes.\n"
        "UnavailableMerlion annotation bindings raise if a forecasting path is used.\n"
        "No SF-OGD/SAOCP update, projection, expert, gradient or initialization code is changed.\n"
        "Apache-2.0 LICENSE and copyright headers are retained. This pinned repo has no NOTICE file.\n"
        "This is not a full online_conformal package installation.\n", encoding="utf-8")
    ns = dict(np=np, pd=pd, copy=copy, math=math, ABC=ABC,
              abstractmethod=abstractmethod, Dict=Dict, Tuple=Tuple,
              defaultdict=defaultdict, ForecasterBase=UnavailableMerlion,
              TimeSeries=UnavailableMerlion, tqdm=UnavailableProgress)
    selections = [
        ("online_conformal/utils.py", None),
        ("online_conformal/base.py", {"BasePredictor"}),
        ("online_conformal/ogd.py", {"ScaleFreeOGD"}),
        ("online_conformal/saocp.py", {"_OGD", "SAOCP"}),
    ]
    for rel, selected in selections:
        path = VENDOR / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (selected is None or n.name in selected)]
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, str(path), "exec"), ns)
    return {"sfogd": ns["ScaleFreeOGD"], "saocp": ns["SAOCP"]}


def create(cls, method, maximum_scale):
    args = dict(model=None, train_data=None, horizon=1, coverage=COVERAGE,
                max_scale=float(maximum_scale))
    if method == "saocp":
        args["lifetime"] = 8  # fixed author default, not selected on test
    result = cls(**args)
    assert result.model is None
    assert result.predict(horizon=1) == (0., 0.)
    return result


def emit_update(calibrator, actual, mean, sigma):
    # The new true score is computed only after this interval has been issued.
    q = float(calibrator.predict(horizon=1)[1])
    assert np.isfinite(q) and q >= 0
    half = q*float(sigma)
    new_score = abs(float(actual)-float(mean))/float(sigma)
    calibrator.update(ground_truth=pd.Series([new_score]),
                      forecast=pd.Series([0.]), horizon=1)
    return q, half, new_score


def scalar_stream(cls, method, maximum_scale, warmup_scores, target_scores):
    calibrator = create(cls, method, maximum_scale)
    for score in warmup_scores:
        emit_update(calibrator, float(score), 0., 1.)
    qs = []
    for score in target_scores:
        q, _, _ = emit_update(calibrator, float(score), 0., 1.)
        qs.append(q)
    return np.asarray(qs)


def read_forecasts():
    path = ROOT / "outputs" / "public_fx" / "predictions.csv"
    if not path.exists():
        path = path.with_suffix(".csv.gz")
    frame = pd.read_csv(path, parse_dates=["date"])
    issued = frame[(frame.predictor == "gru") & (frame.calibrator == "native")].copy()
    key = ["seed", "context", "asset", "date"]
    assert not issued.duplicated(key).any()
    assert np.isfinite(issued[["actual", "mu", "sigma"]]).all().all()
    assert (issued.sigma > 0).all()
    assert issued.date.min() >= pd.Timestamp("2021-01-01")
    return issued.sort_values(key).reset_index(drop=True), path


def validation_forecasts(issued):
    path = ROOT / "scripts" / "run_public_fx_pilot.py"
    spec = importlib.util.spec_from_file_location("public_fx_author_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    returns = pd.read_csv(ROOT / "data" / "public" / "ecb_log_returns_pct.csv",
                          index_col=0, parse_dates=True)
    x, y, metadata, scales = module.build_arrays(returns)
    rows, checks, checkpoints = [], [], []
    torch.set_num_threads(4)
    for seed in sorted(issued.seed.unique()):
        checkpoint = ROOT / "outputs" / "public_fx" / f"gru_seed_{seed}.pt"
        model = module.ProbabilisticGRU()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        checkpoints.append({"seed": int(seed), "path": str(checkpoint.relative_to(ROOT)),
                            "sha256": sha(checkpoint)})
        for context in sorted(issued.context.unique()):
            mean, sigma = module.predict(model, x, int(context))
            for asset in sorted(issued.asset.unique()):
                ix = np.flatnonzero((metadata.asset == asset).to_numpy())
                all_forecasts = pd.DataFrame({
                    "date": metadata.date.iloc[ix].to_numpy(),
                    "actual": y[ix].numpy()*scales[asset],
                    "mu": mean[ix]*scales[asset], "sigma": sigma[ix]*scales[asset]})
                test = issued[(issued.seed == seed) & (issued.context == context)
                              & (issued.asset == asset)].set_index("date")
                rebuilt = all_forecasts.set_index("date").loc[test.index]
                errors = {}
                for col in ["actual", "mu", "sigma"]:
                    err = float(np.max(np.abs(test[col].to_numpy()-rebuilt[col].to_numpy())))
                    errors[col] = err
                    np.testing.assert_allclose(test[col], rebuilt[col], rtol=2e-6, atol=2e-6)
                checks.append({"seed": int(seed), "context": int(context), "asset": asset,
                               "max_abs_test_reconstruction_error": errors,
                               "rows_checked": len(test)})
                warm = all_forecasts[(all_forecasts.date >= "2019-01-01")
                                     & (all_forecasts.date <= "2020-12-31")].copy()
                warm["seed"], warm["context"], warm["asset"] = int(seed), int(context), asset
                rows.append(warm)
    return pd.concat(rows, ignore_index=True), checks, checkpoints, sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=["sfogd", "saocp"], default=["sfogd", "saocp"])
    args = parser.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    algorithms = author_algorithms()
    issued, issued_path = read_forecasts()
    warm, checks, checkpoints, parent_hash = validation_forecasts(issued)
    warm.to_csv(OUT / "reconstructed_validation_forecasts.csv", index=False)
    rows, validation_rows, initialization, causal_checks = [], [], [], []
    for (seed, context, asset), test in issued.groupby(["seed", "context", "asset"], sort=True):
        test = test.sort_values("date")
        initial = warm[(warm.seed == seed) & (warm.context == context) & (warm.asset == asset)].sort_values("date")
        initial_scores = np.abs(initial.actual.to_numpy()-initial.mu.to_numpy())/initial.sigma.to_numpy()
        init_2019 = initial.date < "2020-01-01"
        # Matches author's default scale heuristic, estimated only from 2019.
        # This is an expected scale, not a guaranteed bound on future FX scores.
        maximum_scale = float(np.max(initial_scores[init_2019.to_numpy()])*np.sqrt(3))
        assert maximum_scale > 0 and np.isfinite(maximum_scale)
        for method in args.methods:
            cls = algorithms[method]
            calibrator = create(cls, method, maximum_scale)
            for point in initial.itertuples(index=False):
                q, half, score = emit_update(calibrator, point.actual, point.mu, point.sigma)
                # 2019 is initialization. 2020 is diagnostic, not used to select settings.
                if point.date >= pd.Timestamp("2020-01-01"):
                    miss = max(abs(point.actual-point.mu)-half, 0.)
                    validation_rows.append({"date": point.date.date().isoformat(), "asset": asset,
                        "seed": int(seed), "context": int(context), "calibrator": method,
                        "q": q, "score": score, "hit": int(miss == 0), "width": 2*half,
                        "interval_score": 2*half+2/ALPHA*miss})
            initialization.append({"asset": asset, "seed": int(seed), "context": int(context),
                "calibrator": method, "max_scale": maximum_scale, "initial_q_before_replay": 0.,
                "q_at_test_start": float(calibrator.predict(horizon=1)[1]),
                "scale_estimation_start": "2019-01-01", "scale_estimation_end": "2019-12-31",
                "warmup_start": str(initial.date.min().date()), "warmup_end": str(initial.date.max().date()),
                "warmup_scores": len(initial), "validation_scale_exceedances": int((initial_scores > maximum_scale).sum())})
            for point in test.itertuples(index=False):
                q, half, score = emit_update(calibrator, point.actual, point.mu, point.sigma)
                miss = max(abs(point.actual-point.mu)-half, 0.)
                rows.append({"date": point.date.date().isoformat(), "asset": asset,
                    "seed": int(seed), "predictor": "gru", "context": int(context),
                    "calibrator": method, "actual": point.actual, "mu": point.mu,
                    "sigma": point.sigma, "q": q, "score": score,
                    "half_width": half, "width": 2*half, "hit": int(miss == 0),
                    "interval_score": 2*half+2/ALPHA*miss,
                    "score_exceeds_scale": int(score > maximum_scale), "max_scale": maximum_scale})
            # Actual future-target intervention on a representative stream.
            if seed == issued.seed.min() and context == issued.context.min() and asset == issued.asset.min():
                scores = np.abs(test.actual.to_numpy()-test.mu.to_numpy())/test.sigma.to_numpy()
                probe = scores[:300].copy()
                future = probe.copy()
                future[200:] = 100+20*future[200:]
                a = scalar_stream(cls, method, maximum_scale, initial_scores, probe)
                b = scalar_stream(cls, method, maximum_scale, initial_scores, future)
                assert np.array_equal(a[:201], b[:201])
                causal_checks.append({"method": method, "changed_targets_from_index": 200,
                    "intervals_equal_through_first_changed_target": True,
                    "post_update_state_change_observed": bool(np.any(a[201:] != b[201:]))})
        print(f"completed seed={seed}, context={context}, asset={asset}", flush=True)
    results = pd.DataFrame(rows)
    assert np.isfinite(results.select_dtypes(include="number").to_numpy()).all()
    assert not results.duplicated(["date", "asset", "seed", "context", "calibrator"]).any()
    results.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_diagnostics.csv", index=False)
    summary = results.groupby(["seed", "context", "asset", "calibrator"]).agg(
        n=("hit", "size"), coverage=("hit", "mean"), width=("width", "mean"),
        interval_score=("interval_score", "mean"),
        score_scale_exceedances=("score_exceeds_scale", "sum")).reset_index()
    summary.to_csv(OUT / "summary_by_asset_seed.csv", index=False)
    (OUT / "initialization.json").write_text(json.dumps(initialization, indent=2)+"\n")
    manifest = {
        "date": "2026-10-11", "author_repository": AUTHOR_REPO, "author_commit": COMMIT,
        "license": "Apache-2.0; original LICENSE and copyright headers preserved; no NOTICE in pinned tree",
        "source_adapter": "AST selects verbatim original classes/functions; Merlion bindings forbidden; model=None only",
        "author_source_sha256": SOURCE_HASHES, "methods": args.methods, "coverage": COVERAGE,
        "nonconformity": "abs(actual-issued_mu)/issued_sigma", "score_clipping": False,
        "initial_q": 0., "scale_rule": "sqrt(3)*max(2019 normalized score), fixed before test",
        "saocp_lifetime": 8, "hyperparameter_selection": "none on test; default settings; 2020 diagnostics only",
        "warmup": ["2019-01-01", "2020-12-31"], "test_start": "2021-01-01",
        "test_first_observed": results.date.min(), "test_last_observed": results.date.max(),
        "future_target_intervention": causal_checks, "test_reconstruction": checks,
        "checkpoints": checkpoints, "issued_forecasts_sha256": sha(issued_path),
        "parent_predictor_source_sha256": parent_hash, "script_sha256": sha(__file__),
        "prediction_rows": len(results), "wall_seconds": time.time()-start,
        "limitations": [
            "maximum expected score estimated from 2019 is not a hard future bound; no theoretical guarantee claimed",
            "GRU checkpoint was selected using 2019-2020, so validation replay is retrospective initialization",
            "existing 2021+ test has already been explored; not a fresh blind-test or test-selected method",
            "Gaussian GRU scales may be misspecified; normalized-score weighting differs from absolute-return calibration",
            "same EUR-denominated currencies and repeated seeds are dependent; no IID confidence interval",
            "descriptive baseline pilot, not novelty or superiority evidence; 2020 diagnostics not used for tuning",
        ],
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"prediction_rows": len(results), "wall_seconds": manifest["wall_seconds"],
                      "total_scale_exceedances": int(results.score_exceeds_scale.sum())}, indent=2))


if __name__ == "__main__":
    main()
