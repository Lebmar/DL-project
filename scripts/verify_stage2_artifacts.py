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
    memory_path = ROOT/"outputs/stage2_memory/manifest.json"
    memory = json.loads(memory_path.read_text())
    check("memory validation source", memory["source_sha256"] == sha(ROOT/"scripts/stage2_memory_diagnostics.py"))
    check("memory training provenance", memory["training_manifest_sha256"] == sha(training_path))
    for name, expected in memory["output_sha256"].items():
        check(f"memory artifact {name}", sha(memory_path.parent/name) == expected)
    analysis_path = ROOT/"outputs/stage2_analysis/manifest.json"
    analysis = json.loads(analysis_path.read_text())
    check("analysis complete", analysis["status"] == "complete")
    check("analysis source", analysis["analysis_source_sha256"] == sha(ROOT/"scripts/analyze_stage2.py"))
    check("analysis evaluation provenance", analysis["evaluation_manifest_sha256"] == sha(evaluation_path))
    check("analysis report", analysis["report_sha256"] == sha(ROOT/"research/stage2_results.md"))
    for name, expected in analysis["artifact_sha256"].items():
        check(f"analysis artifact {name}", sha(analysis_path.parent/name) == expected)
    for revision in analysis.get("rendering_revisions", []):
        check("rendering preserves statistics", revision["statistical_csv_unchanged"])
        check("rendering source", revision["source_sha256"] == sha(ROOT/revision["script"]))
    supplement_path = ROOT/"outputs/stage2_strong_supplement/manifest.json"
    supplement = json.loads(supplement_path.read_text())
    check("supplement complete single-seed", supplement["status"] == "complete" and supplement["model_seed"] == 4101)
    check("supplement training provenance", supplement["training_manifest_sha256"] == sha(training_path))
    for name, expected in supplement["source_sha256"].items():
        check(f"supplement source {name}", sha(ROOT/"scripts"/name) == expected)
    for name, expected in supplement.get("artifact_sha256", {}).items():
        check(f"supplement artifact {name}", sha(supplement_path.parent/name) == expected)
    for family, expected in supplement["reference_sha256"].items():
        check(f"supplement reference {family}", sha(evaluation_path.parent/f"reference_{family}.npz") == expected)
    paths = pd.read_csv(supplement_path.parent/"path_metrics.csv.gz")
    check("complete supplementary path rows", len(paths) == 49152 and supplement["path_rows"] == 49152)
    archive_path = ROOT/"outputs/stage2_archive_mechanism/manifest.json"
    archive = json.loads(archive_path.read_text())
    check("archive exploratory complete", archive["status"] == "complete" and not archive["new_blind_test"])
    check("archive source", archive["source_sha256"] == sha(ROOT/"scripts/run_stage2_archive_diagnostic.py"))
    check("archive training provenance", archive["training_manifest_sha256"] == sha(training_path))
    check("archive evaluation provenance", archive["original_evaluation_manifest_sha256"] == sha(evaluation_path))
    for name, expected in archive["artifacts"].items():
        check(f"archive artifact {name}", sha(archive_path.parent/name) == expected)
    diagonal = archive["diagonal_check"]
    check("archive diagonal performance replication", diagonal["performance_rows"] == 640
          and diagonal["maximum_performance_absolute_error"] < 1e-10)
    check("archive diagonal path replication", diagonal["interaction_path_rows"] == 10240
          and diagonal["maximum_path_interaction_absolute_error"] < 1e-10)
    check("crossed bounds", archive["negative_control_checks"] == 160
          and archive["maximum_half_width_error_after_each_crossed_bound"] < 1e-10)
    components = pd.read_csv(archive_path.parent/"paired_component_effects.csv.gz")
    check("complete archive path rows", len(components) == 51200)
    keys = ["family", "architecture", "seed", "phase", "path"]
    values = list(components.columns[6:])
    parts = {c: g.set_index(keys)[values].sort_index() for c, g in components.groupby("component")}
    check("both decomposition paths reproduce total",
          np.allclose(parts["total"], parts["current_at_archive64"]+parts["archive_at_current16"], atol=1e-10, rtol=0)
          and np.allclose(parts["total"], parts["current_at_archive16"]+parts["archive_at_current64"], atol=1e-10, rtol=0))
    geometry_path = ROOT/"outputs/stage2_score_geometry/manifest.json"
    geometry = json.loads(geometry_path.read_text())
    check("score geometry complete exploratory", geometry["status"] == "complete"
          and not geometry["new_blind_test"] and not geometry["primary_analysis_replaced"])
    check("score geometry source", geometry["source_sha256"] == sha(ROOT/"scripts/stage2_score_geometry.py"))
    check("score geometry input provenance", geometry["evaluation_manifest_sha256"] == sha(evaluation_path))
    for name, expected in geometry["artifacts"].items():
        check(f"score geometry artifact {name}", sha(geometry_path.parent/name) == expected)
    check("score geometry realized diagonal replication", geometry["diagonal_realized_max_absolute_error"] < 1e-10)
    errors = geometry["self_check"]["maximum_errors"]
    check("expected score integrates Gaussian and t5 tails", errors["quadrature_absolute_error"] < 1e-10)
    check("mixed derivative finite differences", errors["mixed_derivative_absolute_error"] < 1e-4)
    check("oriented rectangle identity", errors["rectangle_integral_absolute_error"] < 1e-10)
    print(json.dumps({"status": "passed", "checks": len(checks),
                      "scope": "saved stage2 training, controls, evaluation, memory, analysis, supplements and score geometry",
                      "checked": checks}, indent=2))


if __name__ == "__main__":
    main()
