"""Train the conditional 12-fit neural bridge with the unchanged Stage 2 trainer.

No training or bridge-test observations are generated until a committed frozen
protocol pins a completed, passing finite-mechanism manifest.  Resume verifies
complete panels byte-for-byte; partial or unrecorded files are never overwritten.
The outer manifest records actual timestamps because the frozen legacy trainer
retains its historical manifest date.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

import train_stage2 as trainer


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/neural_compatibility_protocol.json"
OUT = ROOT / "outputs/neural_compatibility_training"
FROZEN_SOURCES = {
    "train_stage2.py": "d93d1ecbb6ce30ccba88c2566be2632d769db6c2968717c3f6c36f24fbc49850",
    "stage2_models.py": "c93e127dd43a3f9f70fed3fbd4c63bed8fa0c74b261c33dbd0a327889bd10f2e",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def repo_path(value):
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT), f"Path outside repository: {value}")
    return resolved


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def require_committed(path):
    content = subprocess.check_output(["git", "show", f"HEAD:{relative(path)}"], cwd=ROOT)
    require(hashlib.sha256(content).hexdigest() == sha(path),
            f"Commit the exact source before new training: {relative(path)}")


def save(path, payload):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def directory_hashes(path):
    return {relative(p): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def validate_panels(protocol):
    panels = protocol.get("panels")
    require(isinstance(panels, list) and len(panels) == 3,
            "The small neural bridge requires exactly three training panels")
    require([p.get("panel") for p in panels] == [1, 2, 3],
            "Panels must be specified once, in order 1, 2, 3")
    all_seeds = []
    for item in panels:
        config = item.get("training_config")
        require(isinstance(config, dict), "Each panel requires the complete training_config")
        normalized = copy.deepcopy(config)
        require(set(normalized) == set(trainer.DEFAULT_CONFIG), "Training config schema changed")
        seeds = normalized.get("model_seeds")
        require(isinstance(seeds, list) and len(seeds) == 2 and len(set(seeds)) == 2,
                "Each panel requires exactly two distinct model/sampler seeds")
        values = [*seeds, normalized["training_data"].get("seed"),
                  normalized["validation_data"].get("seed"),
                  normalized["validation_data"].get("pair_seed")]
        require(all(type(v) is int and 0 <= v < 2**32 for v in values),
                "All seed values must be explicit nonnegative 32-bit integers")
        require(not (set(seeds) & set(trainer.DEFAULT_CONFIG["model_seeds"])),
                "Bridge model seeds must be new rather than the original Stage 2/3 seeds")
        all_seeds.extend(values)
        normalized["model_seeds"] = trainer.DEFAULT_CONFIG["model_seeds"]
        normalized["training_data"]["seed"] = trainer.DEFAULT_CONFIG["training_data"]["seed"]
        normalized["validation_data"]["seed"] = trainer.DEFAULT_CONFIG["validation_data"]["seed"]
        normalized["validation_data"]["pair_seed"] = trainer.DEFAULT_CONFIG["validation_data"]["pair_seed"]
        require(normalized == trainer.DEFAULT_CONFIG,
                "Only train/validation/pair seeds and the two model seeds may vary from the frozen trainer")
    require(len(all_seeds) == len(set(all_seeds)),
            "Data seeds and nested model seeds must be distinct within and across panels")


def verified_gate(protocol):
    specification = protocol.get("finite_mechanism_gate")
    require(isinstance(specification, dict), "Protocol must pin finite_mechanism_gate")
    path = repo_path(specification["manifest"])
    require(sha(path) == specification["sha256"], "Finite-mechanism manifest hash mismatch")
    gate = json.loads(path.read_text())
    require(gate.get("status") == "complete", "Finite-mechanism study is not complete")
    require(gate.get("neural_bridge_gate", {}).get("passed") is True,
            "Finite-mechanism bridge gate did not pass; neural training is prohibited")
    checks = gate.get("checks", {})
    require(all(checks.get(key) is True for key in
                ("independent_sort", "strict_past_timing", "actual_phase_native_multiset")),
            "Finite-mechanism implementation checks did not pass")
    require(checks.get("numerical_engine", {}).get("passed") is True,
            "Finite-mechanism numerical-engine checks did not pass")
    tolerance = gate["protocol"]["numerics"]["maximum_phase_metric_difference"]
    require(gate["maximum_quadrature_metric_change"] <= tolerance,
            "Finite-mechanism quadrature did not meet its frozen tolerance")
    require(isinstance(gate.get("sources"), dict) and bool(gate["sources"]),
            "Finite-mechanism source hashes are missing")
    require(isinstance(gate.get("outputs"), dict) and bool(gate["outputs"]),
            "Finite-mechanism artifact hashes are missing")
    for source, digest in gate["sources"].items():
        require(sha(repo_path(source)) == digest, f"Finite-mechanism source changed: {source}")
    for filename, digest in gate["outputs"].items():
        artifact = (path.parent / filename).resolve()
        require(artifact.is_relative_to(path.parent), "Finite artifact escapes its results directory")
        require(sha(artifact) == digest, f"Finite-mechanism artifact changed: {filename}")
    return {"manifest": relative(path), "sha256": sha(path),
            "freeze_commit": gate["freeze_commit"], "verdict": gate["neural_bridge_gate"]}


def verified_protocol(path):
    require_committed(Path(__file__).resolve())
    require_committed(path)
    protocol = json.loads(path.read_text())
    require(protocol.get("frozen") is True, "Bridge protocol must be frozen before training")
    require(protocol.get("sources") == FROZEN_SOURCES, "Frozen trainer/model hashes differ")
    for name, digest in FROZEN_SOURCES.items():
        source = ROOT / "scripts" / name
        require(sha(source) == digest, f"Frozen source changed: {name}")
        require_committed(source)
    validate_panels(protocol)
    require(isinstance(protocol.get("limitations"), list), "Protocol limitations must be explicit")
    return protocol, verified_gate(protocol)


def panel_payload(item, protocol_path):
    return {"frozen": True, "training_config": item["training_config"],
            "neural_compatibility_panel": item["panel"],
            "parent_protocol_sha256": sha(protocol_path)}


def verify_panel(item, panel_out, protocol_path):
    manifest_path = panel_out / "manifest.json"
    result = json.loads(manifest_path.read_text())
    config = item["training_config"]
    expected_grid = {(architecture, seed) for architecture in config["architectures"]
                     for seed in config["model_seeds"]}
    require(len(expected_grid) == 4, "Expected exactly four fits per panel")
    require(result.get("status") == "complete" and result.get("completed_models") == 4,
            f"Panel {item['panel']} is partial or does not contain four complete fits")
    require(result.get("matching_minibatches_verified") is True, "Unverified minibatch pairing")
    require(result.get("training_config") == config, "Saved training config differs from protocol")
    require(result.get("source_sha256") == FROZEN_SOURCES, "Saved panel source hashes differ")
    input_path = panel_out / "input_protocol.json"
    frozen_path = panel_out / "frozen_protocol.json"
    require(json.loads(input_path.read_text()) == panel_payload(item, protocol_path),
            "Panel input protocol does not match the frozen parent")
    require(sha(input_path) == sha(frozen_path) == result["protocol_sha256"],
            "Panel protocol copies or hashes differ")
    models = result.get("models", [])
    require(len(models) == 4 and {(m["architecture"], m["model_seed"]) for m in models} == expected_grid,
            "Panel model grid has missing, extra, or duplicate fits")
    for model in models:
        checkpoint = repo_path(model["checkpoint"])
        require(checkpoint.is_relative_to(panel_out), "Checkpoint must belong to its bridge panel")
        require(sha(checkpoint) == model["checkpoint_sha256"], "Saved checkpoint hash mismatch")
        require(model["causal_audit_before"].get("passed") is True and
                model["causal_audit_selected_checkpoint"].get("passed") is True,
                "Selected checkpoint did not pass causal audit")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require((saved["architecture"], saved["model_seed"]) ==
                (model["architecture"], model["model_seed"]), "Checkpoint fit identity differs")
        require(saved["protocol_sha256"] == sha(input_path) and saved["model_config"] == config["model"],
                "Checkpoint provenance or architecture differs")
        require(saved["selected_epoch"] == model["selected_epoch"], "Selected checkpoint epoch differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify frozen sources, protocol and gate; generate no data or outputs")
    args = parser.parse_args()
    protocol_path, output = repo_path(args.protocol), repo_path(args.output)
    require(output.parent == ROOT / "outputs" and output.name.startswith("neural_compatibility_training"),
            "Use a dedicated outputs/neural_compatibility_training* directory")
    protocol, gate = verified_protocol(protocol_path)
    if args.check_only:
        print(json.dumps({"status": "ready", "models": 12, "protocol_sha256": sha(protocol_path),
                          "finite_mechanism_gate": gate, "training_data_generated": False,
                          "bridge_test_data_accessed": False}), flush=True)
        return

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Bridge manifest already exists; --resume verifies complete panels only")
        manifest = json.loads(manifest_path.read_text())
        require(manifest["protocol_sha256"] == sha(protocol_path), "Resume protocol changed")
        require(manifest["driver_sha256"] == sha(__file__), "Resume driver changed")
        require(manifest["sources"] == FROZEN_SOURCES and manifest["finite_mechanism_gate"] == gate,
                "Resume frozen inputs changed")
        require(manifest["test_data_accessed"] is False, "Training manifest records test access")
        require(manifest["output"] == relative(output), "Resume output directory changed")
        records = manifest.get("panels", [])
        require(len({p["panel"] for p in records}) == len(records) and
                all(p["panel"] in (1, 2, 3) for p in records), "Invalid saved panel records")
        allowed = {"manifest.json", *(f"panel{p['panel']}" for p in records)}
        require(all(p.name in allowed for p in output.iterdir()),
                "Unrecorded/partial panel or output file exists; preserve it before any restart")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing nonempty output directory without a complete outer manifest")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "training", "started_at_utc": utc_now(), "output": relative(output),
            "protocol": relative(protocol_path), "protocol_sha256": sha(protocol_path),
            "driver_sha256": sha(__file__), "sources": FROZEN_SOURCES,
            "finite_mechanism_gate": gate,
            "code_commit_before_training": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "panels": [], "invocations": [], "test_data_accessed": False,
            "adapter": "unchanged trainer.main; only trainer.OUT and full input-protocol path redirected",
            "legacy_date_note": "Frozen trainer panel manifests retain historical date 2026-10-11; actual execution timestamps are recorded in this outer manifest.",
            "seed_design": "three independent data panels, two unique nested paired model/sampler seeds per panel",
            "limitations": protocol["limitations"],
        }

    started = time.monotonic()
    invocation = {"started_at_utc": utc_now(), "resume": bool(args.resume), "status": "running"}
    manifest["invocations"].append(invocation)
    manifest["status"] = "training"
    save(manifest_path, manifest)
    original_out, original_argv = trainer.OUT, sys.argv
    try:
        for item in protocol["panels"]:
            panel = item["panel"]
            panel_out = output / f"panel{panel}"
            recorded = next((p for p in manifest["panels"] if p["panel"] == panel), None)
            if recorded is not None:
                require(directory_hashes(panel_out) == recorded["artifact_sha256"],
                        f"Saved panel {panel} artifacts changed")
                verify_panel(item, panel_out, protocol_path)
                print(json.dumps({"panel": panel, "status": "verified skip"}), flush=True)
                continue
            if panel_out.exists():
                raise FileExistsError(f"Unrecorded/partial panel {panel}; never overwrite or resume its fits")
            panel_out.mkdir()
            panel_protocol = panel_out / "input_protocol.json"
            panel_protocol.write_text(json.dumps(panel_payload(item, protocol_path), indent=2) + "\n")
            panel_started_at = utc_now()
            invocation["current_panel"] = panel
            save(manifest_path, manifest)
            trainer.OUT = panel_out
            sys.argv = ["train_stage2.py", "--protocol", str(panel_protocol)]
            trainer.main()
            result = verify_panel(item, panel_out, protocol_path)
            panel_manifest = panel_out / "manifest.json"
            config = item["training_config"]
            manifest["panels"].append({
                "panel": panel, "manifest": relative(panel_manifest),
                "manifest_sha256": sha(panel_manifest), "artifact_sha256": directory_hashes(panel_out),
                "started_at_utc": panel_started_at, "completed_at_utc": utc_now(),
                "train_seed": config["training_data"]["seed"],
                "validation_seed": config["validation_data"]["seed"],
                "validation_pair_seed": config["validation_data"]["pair_seed"],
                "model_seeds": config["model_seeds"], "models": result["completed_models"],
                "wall_seconds": result["wall_seconds"],
            })
            save(manifest_path, manifest)
        require(sum(p["models"] for p in manifest["panels"]) == 12, "Complete bridge must have 12 fits")
        manifest.update(status="complete", completed_models=12, completed_at_utc=utc_now())
        invocation.update(status="complete", completed_at_utc=utc_now(), wall_seconds=time.monotonic()-started)
        invocation.pop("current_panel", None)
        save(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        invocation.update(status="failed", stopped_at_utc=utc_now(), wall_seconds=time.monotonic()-started,
                          error_type=type(error).__name__, error=str(error))
        save(manifest_path, manifest)
        raise
    finally:
        trainer.OUT, sys.argv = original_out, original_argv
    print(json.dumps({"status": "complete", "models": 12,
                      "wall_seconds_this_invocation": invocation["wall_seconds"],
                      "test_data_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
