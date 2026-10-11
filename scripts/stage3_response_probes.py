"""Frozen validation-only dependence and pulse-response features for all30 fits."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_stage2 import predict_scales
from stage2_memory_diagnostics import diagnose_model
from stage2_models import load_checkpoint
from train_stage2 import array_sha256, context_tensor, generate_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage3_response_probes"
CONFIG = ROOT / "configs/stage3_response_probes.json"
PROTOCOL = ROOT / "configs/stage3_replication_protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads(CONFIG.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert config["frozen"] and protocol["frozen"]
    for path in (Path(__file__).resolve(), CONFIG):
        committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(path)
    for name, digest in protocol["sources"].items():
        assert sha(ROOT / "scripts" / name) == digest
    training_path = ROOT / "outputs/stage3_training/manifest.json"
    training = json.loads(training_path.read_text())
    assert training["status"] == "complete" and training["completed_models"] == 30
    assert training["protocol_sha256"] == sha(PROTOCOL)
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Response probes already exist; refusing overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    sources = {name: sha(ROOT / "scripts" / name) for name in
               ["stage3_response_probes.py", "stage2_memory_diagnostics.py", "stage2_models.py",
                "evaluate_stage2.py", "train_stage2.py"]}
    manifest = {"status": "running", "config_sha256": sha(CONFIG),
                "protocol_sha256": sha(PROTOCOL), "training_manifest_sha256": sha(training_path),
                "sources": sources, "code_commit_before_execution": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "replication_test_data_accessed": False,
                "checkpoint_selection_or_updates": False, "validation_panel_hashes": {},
                "interpretation": config["interpretation"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows, gradients = [], []
    pulse = config["paired_pulse"]
    onset, exit_time = pulse["onset"], pulse["onset"]+pulse["duration"]
    for panel in training["panels"]:
        panel_path = ROOT / panel["manifest"]
        assert sha(panel_path) == panel["manifest_sha256"]
        fitted = json.loads(panel_path.read_text())
        tc = fitted["training_config"]
        val_y = generate_paths(tc["validation_data"], tc["generator"])
        assert array_sha256(val_y) == fitted["data_sha256"]["validation"]
        with np.load(panel_path.parent / "validation_target_pairs.npz") as pairs:
            paths, targets = pairs["paths"][:256].copy(), pairs["targets"][:256].copy()
        contexts = {c: context_tensor(val_y, paths, targets, c, "cpu") for c in (16, 64)}
        permutation = np.random.default_rng(920110).permutation(256)
        rng = np.random.default_rng(2718000+1000*panel["panel"])
        control = rng.normal(size=(pulse["paths"], pulse["length"]))
        event = control.copy()
        event[:, onset:exit_time] *= pulse["amplitude"]
        manifest["validation_panel_hashes"][str(panel["panel"])] = {
            "training_manifest": sha(panel_path), "validation_observations": array_sha256(val_y),
            "pair_paths_first256": array_sha256(paths), "pair_targets_first256": array_sha256(targets),
            "pulse_control": array_sha256(control), "pulse_event": array_sha256(event)}
        for entry in fitted["models"]:
            checkpoint = ROOT / entry["checkpoint"]
            assert sha(checkpoint) == entry["checkpoint_sha256"]
            model, _ = load_checkpoint(checkpoint)
            profile, features, _ = diagnose_model(model, contexts[16], contexts[64], permutation)
            row = {"panel": panel["panel"], "architecture": entry["architecture"],
                   "seed": entry["model_seed"], "validation_nll": entry["best_validation_mean_nll"],
                   "checkpoint_sha256": sha(checkpoint), **features}
            log_responses = {}
            for context in (16, 64):
                # Only the affected issued forecast interval is necessary;
                # the model is still reset with all required past inputs.
                base_sigma = predict_scales(model, control, context, onset, exit_time+64)
                event_sigma = predict_scales(model, event, context, onset, exit_time+64)
                np.testing.assert_array_equal(base_sigma[:, 0], event_sigma[:, 0])
                log_responses[context] = np.log(event_sigma)-np.log(base_sigma)
            for phase, start, stop in (("entry", 0, 32), ("recovery64", 128, 192)):
                for context in (16, 64):
                    row[f"{phase}_context{context}_mean_log_scale_response"] = float(
                        log_responses[context][:, start:stop].mean())
                contrast = log_responses[64][:, start:stop]-log_responses[16][:, start:stop]
                row[f"{phase}_pulse_context_contrast_mean_signed"] = float(contrast.mean())
                row[f"{phase}_pulse_context_contrast_mean_absolute"] = float(np.abs(contrast).mean())
            rows.append(row)
            gradients.extend({"panel": panel["panel"], "architecture": entry["architecture"],
                              "seed": entry["model_seed"], "lag": lag,
                              "mean_abs_raw_input_gradient": float(value)}
                             for lag, value in enumerate(profile, start=1))
            print(json.dumps({"panel": panel["panel"], "architecture": entry["architecture"],
                              "seed": entry["model_seed"], "status": "probed"}), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "model_features.csv", index=False)
    pd.DataFrame(gradients).to_csv(OUT / "gradient_by_lag.csv", index=False)
    manifest.update(status="complete", model_rows=len(rows),
                    wall_seconds=time.time()-started,
                    artifacts={p.name: sha(p) for p in OUT.iterdir() if p.name != "manifest.json"})
    assert len(rows) == 30
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "complete", "models": len(rows),
                      "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
