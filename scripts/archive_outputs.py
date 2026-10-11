"""Deterministically compress large generated CSVs for repository distribution."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "outputs/pilot/trajectory_metrics.csv",
    "outputs/pilot/event_curves.csv",
    "outputs/gru_simulation/trajectory_metrics.csv",
    "outputs/gru_simulation/event_curves.csv",
    "outputs/gru_simulation/raw_interactions_by_path.csv",
    "outputs/gru_simulation/adjusted_interactions_by_path.csv",
    "outputs/archive_intervention/factorial_metrics_by_path.csv",
    "outputs/archive_intervention/interaction_effects_by_path.csv",
    "outputs/archive_intervention/log_decomposition_by_path.csv",
    "outputs/public_fx/predictions.csv",
    "outputs/strong_calibration/predictions.csv",
    "outputs/strong_calibration/validation_diagnostics.csv",
    "outputs/strong_calibration/reconstructed_validation_forecasts.csv",
]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    records = []
    for name in FILES:
        original = ROOT / name
        target = original.with_suffix(".csv.gz")
        if original.exists():
            raw = original.read_bytes()
            compressed = gzip.compress(raw, compresslevel=9, mtime=0)
            target.write_bytes(compressed)
        elif target.exists():
            compressed = target.read_bytes()
            raw = gzip.decompress(compressed)
        else:
            continue
        assert gzip.decompress(compressed) == raw
        records.append({"original": name, "distributed": str(target.relative_to(ROOT)),
                        "uncompressed_sha256": digest(raw), "compressed_sha256": digest(compressed),
                        "uncompressed_bytes": len(raw), "compressed_bytes": len(compressed),
                        "roundtrip_identical": True})
    (ROOT / "outputs" / "archive_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"files": len(records), "raw_bytes": sum(x["uncompressed_bytes"] for x in records),
                      "archived_bytes": sum(x["compressed_bytes"] for x in records)}))


if __name__ == "__main__":
    main()
