"""Run three frozen training-data repeats using the unchanged Stage2 trainer.

Only output-directory and fully specified train/validation seeds vary. No test
paths, test metrics or Stage2 results are loaded. Completed panels may be
verified and skipped with --resume; a partial panel is never silently rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import train_stage2 as trainer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stage3_training"
PROTOCOL = ROOT / "configs/stage3_replication_protocol.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(payload):
    (OUT / "manifest.json").write_text(json.dumps(payload, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["frozen"] is True
    for name, digest in protocol["sources"].items():
        assert sha(ROOT / "scripts" / name) == digest, name
    for path in (Path(__file__).resolve(), PROTOCOL):
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(path), relative
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Existing Stage3 manifest: use --resume for verified complete panels")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["protocol_sha256"] == sha(PROTOCOL)
        assert manifest["driver_sha256"] == sha(__file__)
    else:
        manifest = {"status": "training", "protocol_sha256": sha(PROTOCOL),
                    "driver_sha256": sha(__file__), "sources": protocol["sources"],
                    "code_commit_before_training": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "panels": [], "test_data_accessed": False,
                    "adapter": "unchanged trainer.main with trainer.OUT redirected to independent panel directory",
                    "limitations": protocol["limitations"]}
        save(manifest)
    started = time.time()
    original_out, original_argv = trainer.OUT, sys.argv
    try:
        for item in protocol["panels"]:
            panel = item["panel"]
            panel_out = OUT / f"panel{panel}"
            panel_manifest_path = panel_out / "manifest.json"
            recorded = next((p for p in manifest["panels"] if p["panel"] == panel), None)
            if recorded is not None:
                assert sha(panel_manifest_path) == recorded["manifest_sha256"]
                selected = json.loads(panel_manifest_path.read_text())
                assert selected["status"] == "complete" and selected["matching_minibatches_verified"]
                assert selected["training_config"] == item["training_config"]
                for model in selected["models"]:
                    assert sha(ROOT / model["checkpoint"]) == model["checkpoint_sha256"]
                print(json.dumps({"panel": panel, "status": "verified skip"}), flush=True)
                continue
            if panel_manifest_path.exists():
                raise FileExistsError(f"Unrecorded/partial panel {panel}; preserve it before restarting")
            panel_out.mkdir(parents=True, exist_ok=True)
            panel_protocol = panel_out / "input_protocol.json"
            panel_protocol.write_text(json.dumps({"frozen": True,
                "training_config": item["training_config"], "stage3_panel": panel,
                "parent_protocol_sha256": sha(PROTOCOL)}, indent=2)+"\n")
            trainer.OUT = panel_out
            sys.argv = ["train_stage2.py", "--protocol", str(panel_protocol)]
            trainer.main()
            result = json.loads(panel_manifest_path.read_text())
            assert result["status"] == "complete" and result["completed_models"] == 10
            manifest["panels"].append({"panel": panel,
                "manifest": str(panel_manifest_path.relative_to(ROOT)),
                "manifest_sha256": sha(panel_manifest_path),
                "train_seed": item["training_config"]["training_data"]["seed"],
                "validation_seed": item["training_config"]["validation_data"]["seed"],
                "models": result["completed_models"], "wall_seconds": result["wall_seconds"]})
            manifest["wall_seconds_this_invocation"] = time.time()-started
            save(manifest)
    finally:
        trainer.OUT, sys.argv = original_out, original_argv
    manifest["status"] = "complete"
    manifest["completed_models"] = sum(p["models"] for p in manifest["panels"])
    manifest["wall_seconds_this_invocation"] = time.time()-started
    save(manifest)
    print(json.dumps({"status": "complete", "models": manifest["completed_models"],
                      "wall_seconds": manifest["wall_seconds_this_invocation"]}), flush=True)


if __name__ == "__main__":
    main()
