"""Past-only non-neural scale controls and online calibration for stage 2.

This module accepts observed returns and already-issued scales; it never accepts
the simulator's true volatility, regime labels, event duration, or test results.
Fit on the common training paths, select calibration settings on the independent
validation paths, freeze the returned specs, then evaluate. Pure API calls write
no files; the explicit --fit-protocol CLI saves only the new controls directory.

SF-OGD/SAOCP use the existing, hash-pinned Apache-2.0 author-source AST adapter,
read-only. This supports model=None external scores only, not a Merlion install.
Validation-estimated max_scale is a heuristic, not a hard score bound. Scores
remain untruncated and no bounded-score theoretical coverage claim is made.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter


DEFAULT_ALPHA = .1
DEFAULT_GAMMA = .02  # Fixed practical setting unless validation grid is invoked.
GAMMA_GRID = (.005, .02, .05)
PERSISTENCE_LIMIT = .999 - 1e-8
_AUTHOR_CACHE = None


def _paths(value, label="paths"):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in (1, 2) or array.size == 0:
        raise ValueError(f"{label} must be a nonempty 1-D or 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite observations")
    return (array[None, :] if array.ndim == 1 else array), array.ndim == 1


def _restore(array, was_1d):
    return array[0] if was_1d else array


def _garch_variance(y, variance_init, omega, alpha, beta):
    """v[t] = omega + alpha*y[t-1]^2 + beta*v[t-1]; v[0] is train-only."""
    inputs = np.empty_like(y)
    inputs[:, 0] = variance_init
    inputs[:, 1:] = omega + alpha * np.square(y[:, :-1])
    return lfilter([1.], [1., -beta], inputs, axis=1)


def gaussian_nll(y, issued_sigma):
    values, _ = _paths(y, "y")
    sigma, _ = _paths(issued_sigma, "issued_sigma")
    if values.shape != sigma.shape or (sigma <= 0).any():
        raise ValueError("y and positive issued_sigma must have matching shapes")
    variance = np.square(sigma)
    return float(np.mean(.5 * (np.log(2 * np.pi * variance)
                              + np.square(values) / variance)))


@dataclass(frozen=True)
class ScaleModel:
    name: str
    variance_init: float
    parameters: dict
    metadata: dict

    def scale_paths(self, y):
        """Return sigma[t] before observing y[t]; independently reset each path."""
        values, was_1d = _paths(y, "observed returns")
        if self.name == "constant":
            variance = np.full_like(values, self.variance_init)
        elif self.name.startswith("ewma"):
            weight = float(self.parameters["weight"])
            # Input zero's special value gives the common train-only initial v.
            inputs = np.empty_like(values)
            inputs[:, 0] = self.variance_init
            inputs[:, 1:] = weight * np.square(values[:, :-1])
            variance = lfilter([1.], [1., -(1 - weight)], inputs, axis=1)
        elif self.name == "garch11":
            variance = _garch_variance(values, self.variance_init,
                                       **self.parameters)
        else:
            raise ValueError(f"Unknown scale model: {self.name}")
        if not np.isfinite(variance).all() or (variance <= 0).any():
            raise FloatingPointError(f"Nonpositive or nonfinite variance: {self.name}")
        return _restore(np.sqrt(variance), was_1d)


def _fit_garch(train, variance_init):
    """Segmented zero-mean Gaussian conditional likelihood with train resets.

    omega is optimized in units of the training second moment. The initial
    variance is the same pooled training unconditional second moment used by
    the other controls; it is fixed throughout optimization and evaluation.
    The fitted stationary omega/(1-alpha-beta) is recorded separately.
    """
    y2 = np.square(train)
    n, length = train.shape

    def objective(theta):
        omega_unit, alpha, beta = theta
        variance = _garch_variance(train, variance_init,
                                   omega_unit * variance_init, alpha, beta)
        if not np.isfinite(variance).all() or (variance <= 0).any():
            return 1e100, np.zeros(3)
        loss = float(np.mean(.5 * (np.log(2 * np.pi * variance)
                                   + y2 / variance)))
        coefficient = .5 * (1 / variance - y2 / np.square(variance))
        derivative_input = np.zeros_like(train)
        gradient = []
        # Derivatives also reset to zero at every segment's fixed initial v.
        for component in range(3):
            if component == 0:
                derivative_input[:, 1:] = variance_init
            elif component == 1:
                derivative_input[:, 1:] = y2[:, :-1]
            else:
                derivative_input[:, 1:] = variance[:, :-1]
            derivative = lfilter([1.], [1., -beta], derivative_input, axis=1)
            gradient.append(float(np.mean(coefficient * derivative)))
        return loss, np.asarray(gradient)

    attempts, converged = [], []
    for alpha0, beta0 in ((.05, .90), (.15, .70), (.20, .30), (.001, .95)):
        result = minimize(objective, [1 - alpha0 - beta0, alpha0, beta0],
                          jac=True, method="SLSQP",
                          bounds=((1e-8, None), (0., PERSISTENCE_LIMIT),
                                  (0., PERSISTENCE_LIMIT)),
                          constraints=[{"type": "ineq",
                                        "fun": lambda x: PERSISTENCE_LIMIT - x[1] - x[2],
                                        "jac": lambda x: np.array([0., -1., -1.])}],
                          options={"maxiter": 400, "ftol": 1e-10})
        omega_unit, alpha, beta = map(float, result.x)
        feasible = (omega_unit > 0 and alpha >= 0 and beta >= 0
                    and alpha + beta < .999 and np.isfinite(result.fun))
        attempts.append({"start_alpha": alpha0, "start_beta": beta0,
                         "success": bool(result.success), "feasible": bool(feasible),
                         "message": str(result.message), "iterations": int(result.nit),
                         "train_gaussian_nll": float(result.fun),
                         "omega_in_train_variance_units": omega_unit,
                         "alpha": alpha, "beta": beta})
        if result.success and feasible:
            converged.append(result)
    if not converged:
        raise RuntimeError(f"No feasible converged Gaussian GARCH fit: {attempts}")
    # Select multistart runs using training likelihood only.
    best = min(converged, key=lambda result: result.fun)
    omega_unit, alpha, beta = map(float, best.x)
    params = {"omega": omega_unit * variance_init, "alpha": alpha, "beta": beta}
    fit = {"objective": "segmented zero-mean Gaussian conditional NLL",
           "selection": "minimum training NLL among converged feasible multistart fits",
           "train_gaussian_nll": float(best.fun), "attempts": attempts,
           "converged": True, "alpha_plus_beta": alpha + beta,
           "stationary_variance_implied_by_parameters": params["omega"] / (1 - alpha - beta),
           "near_boundary": {"omega_unit_at_floor": omega_unit <= 1e-7,
                             "alpha_near_zero": alpha <= 1e-5,
                             "beta_near_zero": beta <= 1e-5,
                             "persistence_near_cap": alpha + beta >= .9989},
           "training_path_count": n, "training_path_length": length}
    return params, fit


def fit_baselines(train_paths, validation_paths=None):
    """Fit common constant/EWMA16/EWMA64/Gaussian-GARCH controls.

    All parameters and initial variances use train_paths. validation_paths adds
    comparable Gaussian NLL diagnostics and never selects a GARCH fit. EWMA
    memories are fixed weights 1/16 and 1/64; neither EWMA is a fitted GARCH.
    """
    train, _ = _paths(train_paths, "training paths")
    variance_init = float(np.mean(np.square(train)))
    if not np.isfinite(variance_init) or variance_init <= 1e-12:
        raise ValueError("Training second moment must be finite and exceed 1e-12")
    common = {"mean_forecast": 0., "train_shape": list(train.shape),
              "train_array_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
              "variance_initialization": "pooled training mean(y**2), independently reset per path",
              "initial_variance": variance_init,
              "inputs": "observed past y only; no true volatility/regime inputs"}
    models = {"constant": ScaleModel("constant", variance_init, {},
                                     {**common, "kind": "constant train RMS"})}
    for memory in (16, 64):
        name = f"ewma{memory}"
        models[name] = ScaleModel(name, variance_init, {"weight": 1 / memory},
                                 {**common, "kind": "fixed EWMA squared-return recursion",
                                  "memory": memory, "weight": 1 / memory,
                                  "fitted_GARCH": False})
    parameters, diagnostic = _fit_garch(train, variance_init)
    models["garch11"] = ScaleModel("garch11", variance_init, parameters,
                                  {**common, "kind": "fitted Gaussian GARCH(1,1)",
                                   "parameters": parameters, "fit": diagnostic})
    if validation_paths is not None:
        val, _ = _paths(validation_paths, "independent validation paths")
        for model in models.values():
            model.metadata["validation_gaussian_nll"] = gaussian_nll(val, model.scale_paths(val))
            model.metadata["validation_used_for_fit_selection"] = False
    return models


def load_author_calibrators():
    """Read-only import of the existing exact, pinned author-source adapter.

    Original files, Apache LICENSE, copyright headers, PINNED_COMMIT and
    ADAPTATION remain in the earlier vendor directory. No download or rewrite
    occurs here. Original repository has no NOTICE at this fixed commit.
    Algorithm class/function bodies are executed whole without math changes.
    """
    global _AUTHOR_CACHE
    if _AUTHOR_CACHE is not None:
        return _AUTHOR_CACHE
    path = Path(__file__).with_name("run_strong_calibration.py")
    spec = importlib.util.spec_from_file_location("stage2_pinned_author_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if (adapter.VENDOR / "PINNED_COMMIT.txt").read_text().strip() != adapter.COMMIT:
        raise RuntimeError("Author vendor commit does not match pinned adapter")
    if not (adapter.VENDOR / "LICENSE").exists():
        raise FileNotFoundError("Preserved Apache-2.0 author LICENSE is required")
    for relative, expected in adapter.SOURCE_HASHES.items():
        if adapter.sha(adapter.VENDOR / relative) != expected:
            raise RuntimeError(f"Author source hash mismatch: {relative}")
    ns = dict(np=adapter.np, pd=adapter.pd, copy=adapter.copy, math=adapter.math,
              ABC=adapter.ABC, abstractmethod=adapter.abstractmethod,
              Dict=adapter.Dict, Tuple=adapter.Tuple, defaultdict=adapter.defaultdict,
              ForecasterBase=adapter.UnavailableMerlion,
              TimeSeries=adapter.UnavailableMerlion, tqdm=adapter.UnavailableProgress)
    selections = [("online_conformal/utils.py", None),
                  ("online_conformal/base.py", {"BasePredictor"}),
                  ("online_conformal/ogd.py", {"ScaleFreeOGD"}),
                  ("online_conformal/saocp.py", {"_OGD", "SAOCP"})]
    for relative, selected in selections:
        source_path = adapter.VENDOR / relative
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and (selected is None or node.name in selected)]
        tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
        exec(compile(tree, str(source_path), "exec"), ns)
    _AUTHOR_CACHE = ({"sfogd": ns["ScaleFreeOGD"], "saocp": ns["SAOCP"]},
                     {"repository": adapter.AUTHOR_REPO, "commit": adapter.COMMIT,
                      "source_hashes": dict(adapter.SOURCE_HASHES), "license": "Apache-2.0",
                      "vendor_path": str(adapter.VENDOR),
                      "adapter": "existing AST source adapter, read-only model=None external scores",
                      "initial_q": 0., "saocp_lifetime": 8,
                      "NOTICE": "none in pinned original repository"})
    return _AUTHOR_CACHE


def _quantile(scores, alpha):
    """Practical finite rank ceil((n+1)*(1-alpha)), capped to observed n."""
    n = scores.shape[1]
    rank = np.ceil((n + 1) * (1 - np.asarray(alpha))).astype(int)
    rank = np.clip(rank, 1, n)
    return np.sort(scores, axis=1)[np.arange(len(scores)), rank - 1]


def _warmup(warmup_scores, n):
    if warmup_scores is None:
        return np.empty((n, 0), dtype=np.float64)
    warm, _ = _paths(warmup_scores, "warmup scores")
    if (warm < 0).any():
        raise ValueError("Warmup conformity scores must be nonnegative")
    if warm.shape[0] == 1:
        return np.broadcast_to(warm, (n, warm.shape[1]))
    if warm.shape[0] != n:
        raise ValueError("Warmup needs one common score path or one score path per target path")
    return warm


def calibrate_paths(y, issued_sigma, spec, method="rolling", window=128,
                    warmup_scores=None):
    """Issue intervals before observation/update, with independent path states.

    spec is fitted/frozen by fit_calibration_spec on validation only. Callers
    must pass causally issued sigma, plus each predictor's own 128 validation
    warmup scores for the same raw observed warmup segment. Warmup is observed
    before the target path and updated prequentially. With no warmup, reported
    intervals begin after initial_window observations. Every method also keeps
    learning through the target's unscored pre-event prefix (e.g. 384 steps).

    Return q, half_width, scores, covered, valid arrays of original input shape,
    and metadata including untruncated scores exceeding validation max_scale.
    """
    values, was_1d = _paths(y, "y")
    sigma, _ = _paths(issued_sigma, "issued_sigma")
    if values.shape != sigma.shape or (sigma <= 0).any():
        raise ValueError("y and positive issued_sigma must have matching shapes")
    if method not in ("rolling", "aci", "sfogd", "saocp"):
        raise ValueError("method must be rolling, aci, sfogd or saocp")
    if window not in (32, 128):
        raise ValueError("Predeclared rolling/ACI window must be 32 or 128")
    alpha = float(spec.get("alpha", DEFAULT_ALPHA))
    gamma = float(spec.get("gamma", DEFAULT_GAMMA))
    initial_window = int(spec.get("initial_window", 128))
    maximum_scale = float(spec["max_scale"])
    if not 0 < alpha < .5 or gamma <= 0 or maximum_scale <= 0:
        raise ValueError("Invalid frozen calibration spec")
    if initial_window < window:
        raise ValueError("initial_window must be at least the selected rolling window")
    n, length = values.shape
    warm = _warmup(warmup_scores, n)
    scores = np.abs(values) / sigma  # archived actual scores, never truncated
    if not np.isfinite(scores).all():
        raise FloatingPointError("Normalized target scores overflowed")
    all_scores = np.concatenate((warm, scores), axis=1)
    q_all = np.full_like(all_scores, np.nan)
    clipped_alpha = 0
    provenance = None
    if method in ("rolling", "aci"):
        adaptive_alpha = np.full(n, alpha)
        for t in range(all_scores.shape[1]):
            # Buffer excludes score[t]. Predict first, then observe miss/update.
            if t >= window:
                used_alpha = adaptive_alpha if method == "aci" else np.full(n, alpha)
                q_all[:, t] = _quantile(all_scores[:, t - window:t], used_alpha)
                if method == "aci":
                    miss = all_scores[:, t] > q_all[:, t]
                    adaptive_alpha += gamma * (alpha - miss)
                    floor = 1 / (window + 1)
                    clipped_alpha += int(((adaptive_alpha < floor)
                                          | (adaptive_alpha > .5)).sum())
                    adaptive_alpha = np.clip(adaptive_alpha, floor, .5)
    else:
        algorithms, provenance = load_author_calibrators()
        args = dict(model=None, train_data=None, horizon=1,
                    coverage=1 - alpha, max_scale=maximum_scale)
        if method == "saocp":
            args["lifetime"] = 8  # pinned author default, never test-selected
        for row in range(n):
            calibrator = algorithms[method](**args)
            if calibrator.predict(horizon=1) != (0., 0.):
                raise RuntimeError("Pinned author's q0 changed")
            for t, score in enumerate(all_scores[row]):
                q = float(calibrator.predict(horizon=1)[1])
                if not np.isfinite(q) or q < 0:
                    raise FloatingPointError(f"Invalid author quantile: {q}")
                q_all[row, t] = q
                calibrator.update(ground_truth=pd.Series([float(score)]),
                                  forecast=pd.Series([0.]), horizon=1)
    # Common reporting burn-in, though each method naturally learns earlier.
    q_all[:, :initial_window] = np.nan
    q = q_all[:, warm.shape[1]:]
    half_width = q * sigma
    valid = np.isfinite(half_width)
    covered = valid & (np.abs(values) <= half_width)
    metadata = {"method": method, "window": window if method in ("rolling", "aci") else None,
                "alpha": alpha, "gamma": gamma if method == "aci" else None,
                "warmup_length": int(warm.shape[1]), "initial_window": initial_window,
                "initialization": "same observed validation warmup; independent states per target path",
                "ACI_initial_alpha": alpha if method == "aci" else None,
                "ACI_alpha_clip": [1 / (window + 1), .5] if method == "aci" else None,
                "ACI_clipping_total_final": clipped_alpha,
                "max_scale": maximum_scale, "max_scale_is_hard_bound": False,
                "score_truncation": False,
                "target_score_exceedance_count": int((scores > maximum_scale).sum()),
                "warmup_score_exceedance_count": int((warm > maximum_scale).sum()),
                "causality": "q[t] before score[t]; sigma[t] must be issued by caller before y[t]",
                "author_source": provenance}
    return {"q": _restore(q, was_1d), "half_width": _restore(half_width, was_1d),
            "scores": _restore(scores, was_1d), "covered": _restore(covered, was_1d),
            "valid": _restore(valid, was_1d), "metadata": metadata}


def fit_calibration_spec(validation_y, validation_sigmas: Mapping[str, np.ndarray],
                         gamma=DEFAULT_GAMMA, alpha=DEFAULT_ALPHA,
                         evaluation_start=384, initial_window=128,
                         warmup_length=128, gamma_selection=None):
    """Freeze per-predictor scale heuristics and common observed warmup.

    validation path 0's last warmup_length outcomes are common across predictors,
    but normalized by each predictor's own causally issued scale. max_scale is
    sqrt(3)*max(score) over all independent validation paths, not any test path.
    gamma is fixed .02 by default; gamma_selection can record a prior validation
    grid decision returned by select_global_aci_gamma. No automatic test tuning.
    """
    values, _ = _paths(validation_y, "independent validation y")
    if warmup_length < initial_window or warmup_length > values.shape[1]:
        raise ValueError("Validation warmup must cover initial_window and fit in path 0")
    if not 0 <= evaluation_start < values.shape[1]:
        raise ValueError("evaluation_start must lie within validation path")
    if not validation_sigmas:
        raise ValueError("At least one validation predictor is required")
    result = {}
    for name, issued in validation_sigmas.items():
        sigma, _ = _paths(issued, f"validation sigma: {name}")
        if sigma.shape != values.shape or (sigma <= 0).any():
            raise ValueError(f"Validation forecast shape/positivity mismatch: {name}")
        scores = np.abs(values) / sigma
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Validation normalized scores overflowed: {name}")
        maximum_scale = float(max(np.max(scores) * np.sqrt(3), 1e-12))
        result[name] = {"predictor": name, "alpha": float(alpha), "gamma": float(gamma),
                        "initial_window": int(initial_window), "max_scale": maximum_scale,
                        "max_scale_source": "sqrt(3)*max(all independent validation normalized scores)",
                        "max_scale_is_hard_bound": False, "score_truncation": False,
                        "warmup_scores": scores[0, -warmup_length:].tolist(),
                        "warmup_source": "validation path 0 last observations, same raw segment across predictors",
                        "warmup_length": int(warmup_length), "validation_shape": list(values.shape),
                        "validation_score_max": float(np.max(scores)),
                        "validation_evaluation_start": int(evaluation_start),
                        "gamma_selection": gamma_selection or {
                            "rule": "fixed practical gamma; not optimized", "gamma": float(gamma)},
                        "frozen_before_test": True}
    return result


def interval_score(y, half_width, alpha=DEFAULT_ALPHA):
    y = np.asarray(y, dtype=np.float64)
    half_width = np.asarray(half_width, dtype=np.float64)
    return 2 * half_width + (2 / alpha) * np.maximum(np.abs(y) - half_width, 0)


def select_global_aci_gamma(validation_y, validation_sigmas: Mapping[str, np.ndarray],
                            candidates: Sequence[float] = GAMMA_GRID,
                            evaluation_start=384, window=128, alpha=DEFAULT_ALPHA):
    """Optional validation-only grid, one global gamma across fixed predictors.

    Validation paths initialize from their own first 128 past observations;
    they do not replay their own future final segment as a warmup. Average raw
    IS after evaluation_start within each predictor, then average predictors
    equally. Main factorial does not use ACI; this choice is supplemental.
    Candidates/forecasters/evaluation_start must be fixed before looking at test.
    """
    values, _ = _paths(validation_y, "independent validation y")
    if not validation_sigmas or not 128 <= evaluation_start < values.shape[1]:
        raise ValueError("Need predictors and evaluation_start >=128 within validation path")
    candidate_values = sorted(set(float(value) for value in candidates))
    if not candidate_values or any(value <= 0 for value in candidate_values):
        raise ValueError("Candidate gammas must be positive")
    rows = []
    for gamma in candidate_values:
        predictor_scores = {}
        for name, sigma in validation_sigmas.items():
            spec = {"alpha": alpha, "gamma": gamma, "initial_window": 128,
                    "max_scale": 1.}  # unused by ACI; raw exceedances remain diagnostic
            result = calibrate_paths(values, sigma, spec, "aci", window=window)
            score = interval_score(values, result["half_width"], alpha)
            predictor_scores[name] = float(np.mean(score[:, evaluation_start:]))
        rows.append({"gamma": gamma, "per_predictor_mean_IS": predictor_scores,
                     "equal_predictor_mean_IS": float(np.mean(list(predictor_scores.values())))})
    best = min(rows, key=lambda row: (row["equal_predictor_mean_IS"], row["gamma"]))
    return {"rule": "minimum equal-predictor mean raw interval score on independent validation",
            "gamma": best["gamma"], "candidates": candidate_values, "diagnostics": rows,
            "predictors": list(validation_sigmas), "window": window,
            "evaluation_start": evaluation_start,
            "validation_warmup": "none external; each path first 128 past observations",
            "test_used": False, "tie_break": "smaller gamma"}


def causal_self_check(include_author=True):
    """Tiny synthetic code checks only; does not load project test data/results."""
    rng = np.random.default_rng(73199)
    train = rng.normal(size=(6, 192))
    validation = rng.normal(size=(3, 256))
    models = fit_baselines(train, validation)
    y = rng.normal(size=(2, 256))
    cutoff = 193
    changed = y.copy()
    changed[:, cutoff + 1:] = rng.normal(50., 20., size=changed[:, cutoff + 1:].shape)
    methods = ["rolling", "aci"] + (["sfogd", "saocp"] if include_author else [])
    checked = []
    val_sigmas = {name: model.scale_paths(validation) for name, model in models.items()}
    specs = fit_calibration_spec(validation, val_sigmas, evaluation_start=128)
    for name, model in models.items():
        sigma, sigma_changed = model.scale_paths(y), model.scale_paths(changed)
        # Independent scalar recursion checks units, reset and forecast indexing.
        expected_variance = np.full_like(y, model.variance_init)
        for row in range(len(y)):
            for t in range(1, y.shape[1]):
                if name.startswith("ewma"):
                    weight = model.parameters["weight"]
                    expected_variance[row, t] = ((1 - weight) * expected_variance[row, t - 1]
                                                 + weight * y[row, t - 1] ** 2)
                elif name == "garch11":
                    p = model.parameters
                    expected_variance[row, t] = (p["omega"] + p["alpha"] * y[row, t - 1] ** 2
                                                 + p["beta"] * expected_variance[row, t - 1])
        np.testing.assert_allclose(sigma ** 2, expected_variance, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(sigma[:, :cutoff + 1], sigma_changed[:, :cutoff + 1])
        for method in methods:
            for window in ((32, 128) if method in ("rolling", "aci") else (128,)):
                spec = specs[name]
                one = calibrate_paths(y, sigma, spec, method, window, spec["warmup_scores"])
                two = calibrate_paths(changed, sigma_changed, spec, method, window, spec["warmup_scores"])
                np.testing.assert_array_equal(one["half_width"][:, :cutoff + 1],
                                              two["half_width"][:, :cutoff + 1])
                checked.append(f"{name}/{method}/{window}")
        # Changing the current observation also cannot alter its issued scale/q.
        current_changed = y.copy()
        current_changed[:, cutoff:] += 25
        current_sigma = model.scale_paths(current_changed)
        np.testing.assert_array_equal(sigma[:, :cutoff + 1], current_sigma[:, :cutoff + 1])
        spec = specs[name]
        one = calibrate_paths(y, sigma, spec, "aci", 128, spec["warmup_scores"])
        two = calibrate_paths(current_changed, current_sigma, spec, "aci", 128, spec["warmup_scores"])
        np.testing.assert_array_equal(one["half_width"][:, :cutoff + 1],
                                      two["half_width"][:, :cutoff + 1])
    return {"status": "passed", "synthetic_only": True, "suffix_cutoff": cutoff,
            "checked_model_method_windows": checked,
            "garch_fit": models["garch11"].metadata["fit"]}


def fit_frozen_protocol(protocol_path, output_directory):
    """Fit/save controls on the exact frozen shared train/validation generator.

    No test data/results are constructed or opened. No ACI gamma grid is run.
    Output models[name] can be loaded directly with ScaleModel(**entry).
    """
    import time
    import train_stage2 as training

    protocol_path, output_directory = Path(protocol_path), Path(output_directory)
    if (output_directory / "models.json").exists():
        raise FileExistsError("A saved controls panel already exists; refusing silent replacement")
    config, protocol, raw, protocol_hash = training.read_protocol(protocol_path)
    baseline_path = Path(__file__).resolve()
    generator_path = Path(training.__file__).resolve()
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    generator_hash = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    gamma = float(protocol["secondary"]["aci_gamma"])
    if gamma != DEFAULT_GAMMA:
        raise ValueError("This frozen controls run expects the specified fixed ACI gamma .02")
    started = time.time()
    train = training.generate_paths(config["training_data"], config["generator"])
    validation = training.generate_paths(config["validation_data"], config["generator"])
    models = fit_baselines(train, validation)
    val_sigmas = {name: model.scale_paths(validation) for name, model in models.items()}
    specs = fit_calibration_spec(validation, val_sigmas, gamma=gamma,
                                alpha=DEFAULT_ALPHA, evaluation_start=384,
                                gamma_selection={"rule": "protocol-fixed practical gamma; not optimized",
                                                 "gamma": gamma})
    tiny_check = causal_self_check(include_author=True)
    provenance = {"protocol_path": str(protocol_path.resolve()), "protocol_sha256": protocol_hash,
                  "baseline_source_path": str(baseline_path), "baseline_source_sha256": baseline_hash,
                  "generator_source_path": str(generator_path), "generator_source_sha256": generator_hash,
                  "train_array_sha256": training.array_sha256(train),
                  "validation_array_sha256": training.array_sha256(validation),
                  "training_data": config["training_data"], "validation_data": config["validation_data"],
                  "generator": config["generator"], "test_data_accessed": False,
                  "ACI_gamma_grid_run": False, "ACI_gamma": gamma}
    payload = {"schema": "stage2_scale_controls_v1", "provenance": provenance,
               "models": {name: {"name": model.name, "variance_init": model.variance_init,
                                 "parameters": model.parameters, "metadata": model.metadata}
                          for name, model in models.items()}, "calibration_specs": specs}
    diagnostics = {"provenance": provenance,
                   "fit_rule": "Gaussian GARCH multistart selection on training NLL only",
                   "variance_initialization": "common pooled training second moment, not latent generator sigma",
                   "garch_fit": models["garch11"].metadata["fit"],
                   "validation_gaussian_nll": {name: model.metadata["validation_gaussian_nll"]
                                               for name, model in models.items()},
                   "tiny_self_check_status": tiny_check["status"],
                   "elapsed_seconds": time.time() - started}
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_hash:
        raise RuntimeError("Frozen protocol changed while fitting; refusing stale panel")
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != baseline_hash:
        raise RuntimeError("Baseline source changed while fitting; refusing stale provenance")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("models.json", payload), ("fit_diagnostics.json", diagnostics),
                        ("tiny_self_check.json", tiny_check)):
        with (output_directory / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--skip-author", action="store_true")
    parser.add_argument("--fit-protocol", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "stage2_controls")
    args = parser.parse_args()
    if args.fit_protocol:
        print(json.dumps(fit_frozen_protocol(args.fit_protocol, args.output), indent=2))
    elif args.self_check:
        print(json.dumps(causal_self_check(not args.skip_author), indent=2))
    else:
        parser.print_help()
