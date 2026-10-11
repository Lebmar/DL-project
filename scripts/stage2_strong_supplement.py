"""Single-seed, prespecified online-calibration supplement for Stage 2.

Prepare predictor-specific settings from validation before reading either test
reference file. Evaluation uses only frozen seed4101 scales for Gaussian/t5,
duration128, amplitude3. This is descriptive supplementary evidence; it cannot
support claims about training-seed stability or select a new deployment method.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch

import stage2_baselines as baselines
import stage2_models as models
import train_stage2 as training


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2_strong_supplement"
REFERENCES = ROOT / "outputs" / "stage2_evaluation"
SEED = 4101
PREDICTORS = ("gru_c16", "gru_c64", "tcn_c16", "tcn_c64")
METHODS = ("native", "rolling32", "rolling128", "aci128", "sfogd", "saocp")
FAMILIES = ("gaussian", "student_t5")
PHASES = {"entry": (384, 416), "recovery64": (512, 576),
          "recovery192": (512, 704), "late": (704, 768)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_panel(protocol_path):
    config, protocol, raw, protocol_hash = training.read_protocol(Path(protocol_path))
    manifest_path = ROOT / "outputs" / "stage2_training" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["completed_models"] != 10
            or not manifest["matching_minibatches_verified"]):
        raise ValueError("The complete, matched frozen training panel is required")
    if config != manifest["training_config"] or protocol_hash != manifest["protocol_sha256"]:
        raise RuntimeError("Frozen protocol does not match the trained panel")
    saved_protocol = manifest_path.with_name("frozen_protocol.json")
    if saved_protocol.read_bytes() != raw:
        raise RuntimeError("Training protocol snapshot differs from current protocol")
    for filename, expected in manifest["source_sha256"].items():
        if sha(ROOT / "scripts" / filename) != expected:
            raise RuntimeError(f"Frozen training source changed: {filename}")
    if (protocol["test"]["alpha"] != .1 or protocol["secondary"]["aci_gamma"] != .02
            or protocol["test"]["onset"] != 384 or config["contexts"] != [16, 64]):
        raise ValueError("This supplement requires the prospectively specified settings")
    checkpoints, loaded = {}, {}
    for architecture in ("gru", "tcn"):
        matches = [m for m in manifest["models"]
                   if m["architecture"] == architecture and m["model_seed"] == SEED]
        if len(matches) != 1:
            raise ValueError("Missing or duplicate seed4101 checkpoint")
        entry = matches[0]
        checkpoint = ROOT / entry["checkpoint"]
        if sha(checkpoint) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {checkpoint}")
        model, payload = models.load_checkpoint(checkpoint)
        if (payload["protocol_sha256"] != protocol_hash or payload["model_seed"] != SEED
                or payload["architecture"] != architecture
                or payload["model_config"] != config["model"]
                or payload["selected_epoch"] != entry["selected_epoch"]):
            raise RuntimeError("Checkpoint metadata differs from the frozen training panel")
        checkpoints[architecture] = {"path": str(checkpoint.relative_to(ROOT)),
                                     "sha256": entry["checkpoint_sha256"],
                                     "selected_epoch": entry["selected_epoch"]}
        loaded[architecture] = model
    return config, protocol, protocol_hash, manifest_path, checkpoints, loaded


def predict_points(model, y, context, paths, targets, batch_size=2048):
    """Issue sigma[t] using y[max(0,t-context):t], with left-zero initialization.

    Padding is applied before the same pointwise feature transform used during
    training. No current outcome is visible; t=0 has an all-zero context.
    """
    result = np.empty(len(targets), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            stop = min(start + batch_size, len(targets))
            indices = targets[start:stop, None] + np.arange(-context, 0)[None, :]
            x = y[paths[start:stop, None], np.maximum(indices, 0)]
            x = np.where(indices < 0, 0., x)
            log_scale = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
            result[start:stop] = np.exp(log_scale.astype(np.float64))
    return result


def predict_paths(model, y, context):
    n, length = y.shape
    targets = np.tile(np.arange(length), n)
    paths = np.repeat(np.arange(n), length)
    return predict_points(model, y, context, paths, targets).reshape(y.shape)


def _author_chunk(payload):
    """Each worker calls the unchanged author wrapper on independent paths."""
    y, sigma, spec, method = payload
    torch.set_num_threads(1)
    return baselines.calibrate_paths(y, sigma, spec, method, window=128,
                                     warmup_scores=spec["warmup_scores"])


def parallel_author_paths(y, sigma, spec, method):
    if method not in ("sfogd", "saocp"):
        raise ValueError("Only independent-path author algorithms use this execution wrapper")
    indices = [chunk for chunk in np.array_split(np.arange(len(y)), 4) if len(chunk)]
    payloads = [(y[chunk], sigma[chunk], spec, method) for chunk in indices]
    # Spawn avoids forking an already multithreaded Torch runtime. No worker
    # shares an algorithm state; each path keeps exactly its own warmup/update.
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        chunks = list(pool.map(_author_chunk, payloads))
    result = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
              for key in ("q", "half_width", "scores", "covered", "valid")}
    metadata = chunks[0]["metadata"].copy()
    for count in ("ACI_clipping_total_final", "target_score_exceedance_count",
                  "warmup_score_exceedance_count"):
        metadata[count] = sum(chunk["metadata"][count] for chunk in chunks)
    for chunk in chunks[1:]:
        for key, value in chunks[0]["metadata"].items():
            if key not in ("ACI_clipping_total_final", "target_score_exceedance_count",
                           "warmup_score_exceedance_count") and chunk["metadata"][key] != value:
                raise RuntimeError(f"Worker algorithm provenance differs: {key}")
    metadata["execution"] = "four spawned processes, independent path chunks; unchanged author mathematics"
    metadata["worker_path_chunks"] = [len(chunk) for chunk in indices]
    result["metadata"] = metadata
    return result


def parallel_self_check():
    """Synthetic execution equivalence only; no course/test data loaded."""
    y = np.linspace(-2, 2, 4 * 129).reshape(4, 129)
    sigma = np.linspace(.6, 1.8, 4 * 129).reshape(4, 129)
    spec = {"alpha": .1, "gamma": .02, "initial_window": 128,
            "max_scale": 20., "warmup_scores": np.linspace(.05, 2., 128).tolist()}
    rows = []
    for method in ("sfogd", "saocp"):
        serial = baselines.calibrate_paths(y, sigma, spec, method, window=128,
                                           warmup_scores=spec["warmup_scores"])
        parallel = parallel_author_paths(y, sigma, spec, method)
        for key in ("q", "half_width", "scores", "covered", "valid"):
            np.testing.assert_array_equal(serial[key], parallel[key])
        for key, value in serial["metadata"].items():
            if parallel["metadata"][key] != value:
                raise AssertionError(f"Merged metadata differs from serial: {key}")
        rows.append({"method": method, "arrays_exactly_equal": True,
                     "merged_original_metadata_equal": True, "paths": 4,
                     "observations_per_path": 129, "warmup_per_path": 128})
    return {"passed": True, "checks": rows, "test_data_accessed": False}


def source_hashes():
    return {path.name: sha(path) for path in
            (Path(__file__), Path(training.__file__), Path(models.__file__),
             Path(baselines.__file__))}


def prepare(protocol_path, output):
    """Validation-only preparation; never opens a test reference file."""
    if (output / "validation_calibration_specs.json").exists():
        raise FileExistsError("Validation preparation already exists; do not silently replace it")
    config, protocol, protocol_hash, manifest_path, checkpoints, loaded = read_panel(protocol_path)
    torch.set_num_threads(config["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.time()
    validation = training.generate_paths(config["validation_data"], config["generator"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if training.array_sha256(validation) != manifest["data_sha256"]["validation"]:
        raise RuntimeError("Reconstructed independent validation data hash changed")
    validation_sigmas = {}
    for predictor in PREDICTORS:
        architecture, context = predictor.split("_c")
        validation_sigmas[predictor] = predict_paths(loaded[architecture], validation, int(context))
        print(json.dumps({"stage": "validation_only", "predictor": predictor,
                          "elapsed_seconds": time.time() - started}), flush=True)
    specs = baselines.fit_calibration_spec(
        validation, validation_sigmas, gamma=.02, alpha=.1,
        evaluation_start=384, initial_window=128, warmup_length=128,
        gamma_selection={"rule": "protocol-fixed practical gamma; not optimized", "gamma": .02})
    _, author_source = baselines.load_author_calibrators()
    payload = {"schema": "stage2_neural_strong_supplement_validation_v1",
               "protocol_sha256": protocol_hash, "training_manifest_sha256": sha(manifest_path),
               "validation_array_sha256": training.array_sha256(validation),
               "source_sha256": source_hashes(), "checkpoint_metadata": checkpoints,
               "model_seed": SEED, "validation_prediction_initialization":
               "finite observed context, left zero raw padding, target outcome excluded",
               "validation_sigma_sha256": {k: training.array_sha256(v)
                                             for k, v in validation_sigmas.items()},
               "calibration_specs": specs, "author_source": author_source,
               "test_data_accessed": False, "prepared_wall_seconds": time.time() - started,
               "scope": "single model seed supplement; no test parameter selection"}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation_calibration_specs.json", payload)
    return payload


def committed_source():
    """Require this exact supplement source to be committed before test access."""
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    saved = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if hashlib.sha256(saved).hexdigest() != sha(__file__):
        raise RuntimeError("Commit this exact supplement script before opening test references")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def read_reference(path, protocol):
    with np.load(path, allow_pickle=False) as data:
        required = ["y_event", "y_control", "true_sigma_event", "true_sigma_control"]
        required += [f"{predictor}_{variant}" for predictor in PREDICTORS
                     for variant in ("event", "control")]
        if not set(required).issubset(data.files):
            raise ValueError(f"Reference missing keys: {set(required) - set(data.files)}")
        result = {key: np.asarray(data[key], dtype=np.float64) for key in required}
        for key, expected in (("onset", 384), ("exit", 512)):
            if key in data.files and int(np.asarray(data[key]).item()) != expected:
                raise ValueError(f"Reference {key} differs from the specified reference event")
    expected_shape = (protocol["test"]["paths"], protocol["test"]["length"])
    for key, value in result.items():
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"Reference array shape/nonfinite error: {key}")
        if key != "y_event" and key != "y_control" and (value <= 0).any():
            raise ValueError(f"Reference scales must be positive: {key}")
    if not np.array_equal(result["y_event"][:, :384], result["y_control"][:, :384]):
        raise RuntimeError("Reference event/control do not share the pre-event observations")
    for predictor in PREDICTORS:
        if not np.array_equal(result[f"{predictor}_event"][:, :385],
                              result[f"{predictor}_control"][:, :385]):
            raise RuntimeError("Same-prefix scales must coincide through the event onset forecast")
    return result


def check_reference_predictions(reference, loaded):
    """Fixed numerical provenance check, independent of interval outcomes."""
    chosen_paths = np.array([0, 7, 63, 127])
    chosen_targets = np.array([0, 16, 64, 384, 385, 512, 704, 1023])
    paths = np.repeat(chosen_paths, len(chosen_targets))
    targets = np.tile(chosen_targets, len(chosen_paths))
    errors = {}
    for predictor in PREDICTORS:
        architecture, context = predictor.split("_c")
        for variant in ("event", "control"):
            actual = predict_points(loaded[architecture], reference[f"y_{variant}"],
                                    int(context), paths, targets)
            saved = reference[f"{predictor}_{variant}"][paths, targets]
            error = float(np.max(np.abs(actual - saved)))
            if not np.allclose(actual, saved, rtol=2e-6, atol=2e-6):
                raise RuntimeError(f"Reference scales disagree with frozen checkpoint: {predictor} {variant}")
            errors[f"{predictor}_{variant}"] = error
    return errors


def evaluate(protocol_path, output, references):
    if (output / "manifest.json").exists():
        raise FileExistsError("Supplement evaluation already exists; refusing silent replacement")
    config, protocol, protocol_hash, manifest_path, checkpoints, loaded = read_panel(protocol_path)
    prepared = json.loads((output / "validation_calibration_specs.json").read_text(encoding="utf-8"))
    if (prepared["protocol_sha256"] != protocol_hash
            or prepared["training_manifest_sha256"] != sha(manifest_path)
            or prepared["checkpoint_metadata"] != checkpoints
            or prepared["source_sha256"] != source_hashes()
            or prepared["test_data_accessed"] is not False):
        raise RuntimeError("Validation preparation provenance changed before evaluation")
    code_commit = committed_source()
    torch.set_num_threads(config["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.time()
    rows, metadata, reference_hashes, forecast_checks = [], [], {}, {}
    for family in FAMILIES:
        reference_path = references / f"reference_{family}.npz"
        reference_hashes[family] = sha(reference_path)
        reference = read_reference(reference_path, protocol)
        forecast_checks[family] = check_reference_predictions(reference, loaded)
        for predictor in PREDICTORS:
            architecture, context = predictor.split("_c")
            spec = prepared["calibration_specs"][predictor]
            for variant in ("event", "control"):
                y = reference[f"y_{variant}"]
                sigma = reference[f"{predictor}_{variant}"]
                # True simulator sigma is retained only for reference integrity;
                # it is not passed to a predictor, calibration spec, or update.
                scores = np.abs(y) / sigma
                for method in METHODS:
                    if method == "native":
                        half_width = norm.ppf(1 - .1 / 2) * sigma
                        method_meta = {"method": "native", "native_quantile":
                                       float(norm.ppf(1 - .1 / 2)), "valid_count": int(y.size)}
                    else:
                        algorithm, window = (("rolling", 32) if method == "rolling32" else
                                             ("rolling", 128) if method == "rolling128" else
                                             ("aci", 128) if method == "aci128" else (method, 128))
                        result = (parallel_author_paths(y, sigma, spec, algorithm)
                                  if algorithm in ("sfogd", "saocp") else
                                  baselines.calibrate_paths(
                                      y, sigma, spec, algorithm, window=window,
                                      warmup_scores=spec["warmup_scores"]))
                        half_width, method_meta = result["half_width"], result["metadata"]
                    score = baselines.interval_score(y, half_width, alpha=.1)
                    coverage = (np.abs(y) <= half_width).astype(float)
                    for phase, (lo, hi) in PHASES.items():
                        if not np.isfinite(half_width[:, lo:hi]).all():
                            raise RuntimeError(f"Unissued interval in reporting phase: {method} {phase}")
                        for path in range(len(y)):
                            rows.append({"family": family, "architecture": architecture,
                                         "model_seed": SEED, "context": int(context),
                                         "method": method, "variant": variant, "phase": phase,
                                         "path": path, "phase_start": lo, "phase_stop": hi,
                                         "interval_score": float(score[path, lo:hi].mean()),
                                         "coverage": float(coverage[path, lo:hi].mean()),
                                         "width": float((2 * half_width[path, lo:hi]).mean())})
                    metadata.append({"family": family, "predictor": predictor, "variant": variant,
                                     "label": method, **method_meta,
                                     "validation_score_max": spec["validation_score_max"],
                                     "score_over_validation_max_count":
                                     int((scores > spec["validation_score_max"]).sum()),
                                     "score_over_max_scale_count": int((scores > spec["max_scale"]).sum()),
                                     "phase_score_over_validation_max_count": {
                                         phase: int((scores[:, lo:hi] > spec["validation_score_max"]).sum())
                                         for phase, (lo, hi) in PHASES.items()},
                                     "score_truncation": False})
                    print(json.dumps({"stage": "single_seed_supplement", "family": family,
                                      "predictor": predictor, "variant": variant, "method": method,
                                      "elapsed_seconds": time.time() - started}), flush=True)
    metrics = pd.DataFrame(rows)
    groups = ["family", "architecture", "model_seed", "context", "method", "variant", "phase"]
    summary = metrics.groupby(groups, as_index=False).agg(
        paths=("path", "size"), mean_interval_score=("interval_score", "mean"),
        sd_path_interval_score=("interval_score", "std"),
        mean_coverage=("coverage", "mean"), mean_width=("width", "mean"))
    paired = metrics.pivot(index=[g for g in groups if g != "variant"] + ["path"],
                           columns="variant", values=["interval_score", "coverage", "width"])
    adjusted_rows = []
    for metric in ("interval_score", "coverage", "width"):
        delta = (paired[(metric, "event")] - paired[(metric, "control")]).rename("delta")
        item = delta.reset_index().groupby([g for g in groups if g != "variant"], as_index=False).agg(
            paired_paths=("path", "size"), mean_event_minus_control=("delta", "mean"),
            sd_path_event_minus_control=("delta", "std"))
        item["metric"] = metric
        adjusted_rows.append(item)
    metrics.to_csv(output / "path_metrics.csv.gz", index=False,
                   compression={"method": "gzip", "mtime": 0})
    summary.to_csv(output / "phase_summary.csv", index=False)
    pd.concat(adjusted_rows, ignore_index=True).to_csv(output / "paired_event_minus_control_summary.csv", index=False)
    write_json(output / "calibrator_metadata.json", metadata)
    manifest = {"schema": "stage2_strong_supplement_v1", "status": "complete",
                "model_seed": SEED, "protocol_sha256": protocol_hash,
                "training_manifest_sha256": sha(manifest_path), "source_sha256": source_hashes(),
                "supplement_code_commit_before_test_read": code_commit,
                "validation_preparation_sha256": sha(output / "validation_calibration_specs.json"),
                "reference_sha256": reference_hashes, "checkpoint_metadata": checkpoints,
                "reference_frozen_prediction_spotcheck_max_abs_error": forecast_checks,
                "reference_scenario": {"families": list(FAMILIES), "duration": 128,
                                       "amplitude": 3., "onset": 384, "exit": 512},
                "methods": list(METHODS), "phases": PHASES, "path_rows": len(metrics),
                "author_execution": "four independent path chunks in spawned processes; no shared states or parameter changes",
                "paths_per_family": int(protocol["test"]["paths"]),
                "online_order": "issued sigma/q before current observation, then feedback",
                "warmup": "predictor-specific scores from common last128 observations of validation path0; sequential unscored test pre384",
                "validation_max_scale_is_a_hard_bound": False, "scores_truncated": False,
                "oracle_scale_used_by_methods": False,
                "scope": "descriptive single seed4101 supplement; no generalized seed-stability or superiority claim",
                "limitations": ["SF-OGD/SAOCP validation max_scale is a heuristic, no bounded-score theorem claimed",
                                "ACI is the clipped practical variant; gamma=.02 fixed before test",
                                "native Gaussian quantile is misspecified for t5",
                                "summary path SD is descriptive, not a confidence interval",
                                "recovery64 is contained within recovery192; phases are dependent"],
                "wall_seconds": time.time() - started}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "stage2_protocol.json")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true", help="validation-only freeze; no test access")
    action.add_argument("--evaluate", action="store_true", help="requires prior preparation and committed source")
    action.add_argument("--self-check", action="store_true", help="synthetic exact serial/parallel equivalence, no test access")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(parallel_self_check(), indent=2))
        return
    result = (prepare(args.protocol, args.output) if args.prepare_only else
              evaluate(args.protocol, args.output, args.references))
    print(json.dumps({"status": result.get("status", "validation_prepared"),
                      "test_data_accessed": result.get("test_data_accessed", True),
                      "elapsed_seconds": result.get("wall_seconds", result.get("prepared_wall_seconds"))}), flush=True)


if __name__ == "__main__":
    main()
