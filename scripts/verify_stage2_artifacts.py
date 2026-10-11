"""Verify provenance, complete panel and saved stage2 outputs without retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    protocol = ROOT / "configs/stage2_protocol.json"
    payload = json.loads(protocol.read_text())
    training_path = ROOT / "outputs/stage2_training/manifest.json"
    training = json.loads(training_path.read_text())
    check("training complete", training["status"] == "complete")
    check("protocol unchanged", training["protocol_sha256"] == sha(protocol))
    check("frozen snapshot identical", (training_path.parent/"frozen_protocol.json").read_bytes() == protocol.read_bytes())
    check("common minibatches", training["matching_minibatches_verified"])
    for name, expected in training["source_sha256"].items():
        check(f"training source {name}", sha(ROOT/"scripts"/name) == expected)
    actual_pairs = {(m["architecture"], m["model_seed"]) for m in training["models"]}
    expected_pairs = {(a, s) for a in payload["training_config"]["architectures"]
                      for s in payload["training_config"]["model_seeds"]}
    check("ten distinct fits", len(training["models"]) == 10 and actual_pairs == expected_pairs)
    for model in training["models"]:
        name = f"{model['architecture']}_{model['model_seed']}"
        check(f"checkpoint {name}", sha(ROOT/model["checkpoint"]) == model["checkpoint_sha256"])
        history = pd.read_csv(training_path.parent/"history"/f"{model['architecture']}_seed_{model['model_seed']}.csv")
        check(f"selected epoch minimizes validation {name}",
              int(history.loc[history.mean_validation_nll.idxmin(), "epoch"]) == model["selected_epoch"])
    controls_path = ROOT/"outputs/stage2_controls/models.json"
    controls = json.loads(controls_path.read_text())
    check("controls protocol", controls["provenance"]["protocol_sha256"] == sha(protocol))
    check("controls source", controls["provenance"]["baseline_source_sha256"] == sha(ROOT/"scripts/stage2_baselines.py"))
    check("controls no test fitting", not controls["provenance"]["test_data_accessed"])
    check("fixed ACI gamma", controls["provenance"]["ACI_gamma"] == .02 and not controls["provenance"]["ACI_gamma_grid_run"])
    evaluation_path = ROOT/"outputs/stage2_evaluation/manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    check("evaluation complete", evaluation["status"] == "complete" and evaluation["cases"] == 30)
    check("evaluation training provenance", evaluation["training_manifest_sha256"] == sha(training_path))
    check("evaluation controls provenance", evaluation["controls_sha256"] == sha(controls_path))
    for name, expected in evaluation["sources"].items():
        check(f"evaluation source {name}", sha(ROOT/"scripts"/name) == expected)
    for name, expected in evaluation["artifacts"].items():
        check(f"evaluation artifact {name}", sha(evaluation_path.parent/name) == expected)
    negative = pd.read_csv(evaluation_path.parent/"negative_controls.csv")
    check("1800 neural finite-memory controls", len(negative) == 30*10*2*3)
    check("late widths coincide", negative.late_max_abs_width_error.max() <= 4e-6)
    interactions = pd.read_csv(evaluation_path.parent/"paired_interactions.csv.gz")
    check("complete interaction grid", len(interactions) == 30*(10+1)*4*128)
    check("no duplicate interaction units", not interactions.duplicated(
        ["family", "duration", "amplitude", "architecture", "seed", "phase", "path"]).any())
    check("finite metrics", np.isfinite(interactions.iloc[:, 7:].to_numpy()).all())
    for family in ("gaussian", "student_t5"):
        with np.load(evaluation_path.parent/f"reference_{family}.npz", allow_pickle=False) as reference:
            check(f"{family} paired observations before and after event",
                  np.array_equal(reference["y_event"][:, :384], reference["y_control"][:, :384])
                  and np.array_equal(reference["y_event"][:, 512:], reference["y_control"][:, 512:]))
            for arch in ("gru", "tcn"):
                for context in (16, 64):
                    event, control = reference[f"{arch}_c{context}_event"], reference[f"{arch}_c{context}_control"]
                    check(f"{family}/{arch}/{context} same-context scale equality",
                          np.array_equal(event[:, :385], control[:, :385])
                          and np.array_equal(event[:, 512+context:], control[:, 512+context:]))
    print(json.dumps({"status": "passed", "checks": len(checks),
                      "scope": "saved training/controls/evaluation; analysis and supplements retain own manifests",
                      "checked": checks}, indent=2))


if __name__ == "__main__":
    main()
