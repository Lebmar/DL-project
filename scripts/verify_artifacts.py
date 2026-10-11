"""Verify published source, checkpoint, data and compressed-output provenance."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(value).hexdigest()


def published_bytes(path):
    path = ROOT / path
    if path.exists():
        return path.read_bytes()
    gz = path.with_suffix(path.suffix + ".gz")
    return gzip.decompress(gz.read_bytes())


def main():
    manifests = {}
    checks = 0
    for name, script, key in [
        ("pilot", "run_pilot.py", "source_sha256"),
        ("gru_simulation", "run_gru_simulation.py", "source_sha256"),
        ("archive_intervention", "run_archive_intervention.py", "source_sha256"),
        ("public_fx", "run_public_fx_pilot.py", "source_sha256"),
        ("strong_calibration", "run_strong_calibration.py", "script_sha256"),
    ]:
        manifest = json.loads((ROOT / "outputs" / name / "run_manifest.json").read_text())
        assert manifest[key] == sha((ROOT / "scripts" / script).read_bytes()), (name, "source hash")
        manifests[name] = manifest
        checks += 1
    analysis = json.loads((ROOT / "outputs/public_fx/analysis_manifest.json").read_text())
    assert analysis["source_sha256"] == sha((ROOT / "scripts/analyze_public_fx.py").read_bytes())
    assert analysis["prediction_file_sha256"] == sha(published_bytes(analysis["prediction_file"]))
    assert analysis["strong_prediction_file_sha256"] == sha(published_bytes("outputs/strong_calibration/predictions.csv"))
    checks += 3
    assert manifests["gru_simulation"]["training_generator_source_sha256"] == manifests["pilot"]["source_sha256"]
    assert manifests["archive_intervention"]["gru_source_sha256"] == manifests["gru_simulation"]["source_sha256"]
    assert manifests["strong_calibration"]["parent_predictor_source_sha256"] == manifests["public_fx"]["source_sha256"]
    assert manifests["strong_calibration"]["issued_forecasts_sha256"] == sha(published_bytes("outputs/public_fx/predictions.csv"))
    checks += 4
    for name in ["archive_intervention", "public_fx"]:
        directory = "gru_simulation" if name == "archive_intervention" else name
        for seed, expected in manifests[name]["checkpoint_sha256"].items():
            assert sha((ROOT / "outputs" / directory / f"gru_seed_{seed}.pt").read_bytes()) == expected
            checks += 1
    for item in manifests["strong_calibration"]["checkpoints"]:
        assert sha((ROOT / item["path"]).read_bytes()) == item["sha256"]
        checks += 1
    for name, expected in manifests["strong_calibration"]["author_source_sha256"].items():
        assert sha((ROOT / "outputs/strong_calibration/vendor/online_conformal" / name).read_bytes()) == expected
        checks += 1
    assert sha((ROOT / "data/public/ecb_reference_rates.zip").read_bytes()) == manifests["public_fx"]["data"]["archive_sha256"]
    checks += 1
    archives = json.loads((ROOT / "outputs/archive_manifest.json").read_text())
    for item in archives:
        compressed = (ROOT / item["distributed"]).read_bytes()
        assert sha(compressed) == item["compressed_sha256"]
        assert sha(gzip.decompress(compressed)) == item["uncompressed_sha256"]
        checks += 2
    print(json.dumps({"provenance_checks_passed": checks, "compressed_files": len(archives)}))


if __name__ == "__main__":
    main()
