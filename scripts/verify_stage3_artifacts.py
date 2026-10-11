"""Verify saved Stage3 provenance and paired outputs without fitting or test generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["interval_score", "coverage", "width", "coverage_deficit", "excess_width_to_oracle"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    checks = []

    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    protocol_path = ROOT / "configs/stage3_replication_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    training_path = ROOT / "outputs/stage3_training/manifest.json"
    training = json.loads(training_path.read_text())
    check("frozen protocol", protocol["frozen"] and training["protocol_sha256"] == sha(protocol_path))
    check("complete three training panels", training["status"] == "complete"
          and training["completed_models"] == 30 and len(training["panels"]) == 3)
    check("training driver", training["driver_sha256"] == sha(ROOT / "scripts/train_stage3_repeats.py"))
    check("training excludes test", not training["test_data_accessed"])
    for name, expected in training["sources"].items():
        check(f"frozen source {name}", protocol["sources"][name] == expected == sha(ROOT / "scripts" / name))
    checkpoints, model_ids = {}, set()
    for item in protocol["panels"]:
        panel = item["panel"]
        record = next(p for p in training["panels"] if p["panel"] == panel)
        panel_path = ROOT / record["manifest"]
        fitted = json.loads(panel_path.read_text())
        check(f"panel{panel} manifest", sha(panel_path) == record["manifest_sha256"])
        check(f"panel{panel} config", fitted["training_config"] == item["training_config"])
        check(f"panel{panel} complete paired batches", fitted["status"] == "complete"
              and fitted["completed_models"] == 10 and fitted["matching_minibatches_verified"])
        original = panel_path.parent / "input_protocol.json"
        copy = panel_path.parent / "frozen_protocol.json"
        check(f"panel{panel} frozen copy", sha(original) == sha(copy) == fitted["protocol_sha256"])
        check(f"panel{panel} parent protocol", json.loads(original.read_text())["parent_protocol_sha256"] == sha(protocol_path))
        with np.load(panel_path.parent / "validation_target_pairs.npz", allow_pickle=False) as pairs:
            check(f"panel{panel} saved path pairs", array_sha(pairs["paths"]) == fitted["data_sha256"]["validation_pair_paths"])
            check(f"panel{panel} saved target pairs", array_sha(pairs["targets"]) == fitted["data_sha256"]["validation_pair_targets"])
        histories = {}
        for model in fitted["models"]:
            key = (panel, model["architecture"], model["model_seed"])
            model_ids.add(key)
            checkpoint = ROOT / model["checkpoint"]
            checkpoints[key] = sha(checkpoint)
            check(f"checkpoint {key}", checkpoints[key] == model["checkpoint_sha256"])
            history = pd.read_csv(panel_path.parent / "history" / f"{model['architecture']}_seed_{model['model_seed']}.csv")
            histories[(model["architecture"], model["model_seed"])] = history
            check(f"validation-selected epoch {key}", model["selected_epoch"] == int(history.loc[history.mean_validation_nll.idxmin(), "epoch"]))
        for seed in item["training_config"]["model_seeds"]:
            a, b = histories[("gru", seed)], histories[("tcn", seed)]
            n = min(len(a), len(b))
            check(f"panel{panel} matched sampler prefix seed{seed}",
                  a.minibatch_epoch_sha256.iloc[:n].tolist() == b.minibatch_epoch_sha256.iloc[:n].tolist())
    check("exact thirty model identities", model_ids == {(p["panel"], a, s) for p in protocol["panels"]
          for a in p["training_config"]["architectures"] for s in p["training_config"]["model_seeds"]})
    probe_path = ROOT / "outputs/stage3_response_probes/manifest.json"
    probes = json.loads(probe_path.read_text())
    check("complete validation probes", probes["status"] == "complete" and probes["model_rows"] == 30)
    check("probe protocol", probes["protocol_sha256"] == sha(protocol_path))
    check("probe training", probes["training_manifest_sha256"] == sha(training_path))
    check("probe config", probes["config_sha256"] == sha(ROOT / "configs/stage3_response_probes.json"))
    check("probes neither access replication test nor update checkpoints", not probes["replication_test_data_accessed"]
          and not probes["checkpoint_selection_or_updates"])
    for name, expected in probes["sources"].items():
        check(f"probe source {name}", sha(ROOT / "scripts" / name) == expected)
    for name, expected in probes["artifacts"].items():
        check(f"probe artifact {name}", sha(probe_path.parent / name) == expected)
    features = pd.read_csv(probe_path.parent / "model_features.csv")
    check("all fits retained in features", len(features) == 30 and not features.duplicated(["panel", "architecture", "seed"]).any()
          and set(map(tuple, features[["panel", "architecture", "seed"]].to_numpy())) == model_ids)
    check("features use unchanged checkpoints", all(row.checkpoint_sha256 == checkpoints[(row.panel, row.architecture, row.seed)]
          for row in features.itertuples()))
    evaluation_path = ROOT / "outputs/stage3_evaluation/manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    check("complete four-case evaluation", evaluation["status"] == "complete" and evaluation["cases"] == 4
          and evaluation["models_evaluated"] == 30 and evaluation["model_exclusions"] == "none")
    check("evaluation protocol and new test specification", evaluation["protocol_sha256"] == sha(protocol_path)
          and evaluation["test"] == protocol["test"])
    check("evaluation training", evaluation["training_manifest_sha256"] == sha(training_path))
    check("evaluation source", evaluation["source_sha256"] == sha(ROOT / "scripts/evaluate_stage3.py"))
    for hashes in evaluation["panel_artifact_sha256"].values():
        for name, expected in hashes.items():
            check(f"evaluation training artifact {name}", sha(ROOT / name) == expected)
    for name, expected in evaluation["artifacts"].items():
        check(f"evaluation artifact {name}", sha(evaluation_path.parent / name) == expected)
    check("diagonal reproduction", evaluation["diagonal_check_count"] == 480
          and evaluation["maximum_diagonal_all_five_metric_error"] < 1e-10)
    check("crossed finite-memory controls", evaluation["negative_control_check_count"] == 960
          and evaluation["maximum_crossed_half_width_error_after_bound"] <= 2e-6)
    effects = pd.read_csv(evaluation_path.parent / "paired_interactions.csv.gz")
    keys = ["panel", "family", "duration", "amplitude", "architecture", "seed", "phase", "path"]
    check("complete paired interactions", len(effects) == 61440 and not effects.duplicated(keys).any())
    check("finite interaction metrics", np.isfinite(effects[METRICS].to_numpy()).all())
    parts = pd.read_csv(evaluation_path.parent / "paired_component_effects.csv.gz")
    check("complete crossed component rows", len(parts) == 368640 and not parts.duplicated(keys+["component"]).any())
    arrays = {c: g.set_index(keys)[METRICS].sort_index() for c, g in parts.groupby("component")}
    check("diagonal J equals crossed total", np.allclose(effects.set_index(keys)[METRICS].sort_index(), arrays["total"], atol=1e-10, rtol=0))
    check("both component routes sum to J", np.allclose(arrays["total"], arrays["current_at_archive64"]+arrays["archive_at_current16"], atol=1e-10, rtol=0)
          and np.allclose(arrays["total"], arrays["current_at_archive16"]+arrays["archive_at_current64"], atol=1e-10, rtol=0))
    analysis_path = ROOT / "outputs/stage3_analysis/manifest.json"
    analysis = json.loads(analysis_path.read_text())
    check("complete analysis", analysis["status"] == "complete")
    check("analysis protocol", analysis["protocol_sha256"] == sha(protocol_path))
    check("analysis training", analysis["training_manifest_sha256"] == sha(training_path))
    check("analysis evaluation", analysis["evaluation_manifest_sha256"] == sha(evaluation_path))
    check("analysis probes", analysis["probe_manifest_sha256"] == sha(probe_path))
    check("analysis source", analysis["analysis_source_sha256"] == sha(ROOT / "scripts/analyze_stage3.py"))
    for field in ("input_sha256", "dependency_sha256"):
        for name, expected in analysis[field].items():
            check(f"analysis {field} {name}", sha(ROOT / name) == expected)
    for name, expected in analysis["artifact_sha256"].items():
        check(f"analysis artifact {name}", sha(analysis_path.parent / name) == expected)
    check("analysis report", analysis["report_sha256"] == sha(ROOT / "research/stage3_results.md"))
    check("bootstrap/grid/decomposition checks", analysis["checks"]["passed"]
          and analysis["checks"]["decomposition_max_error"] < 1e-10
          and analysis["checks"]["identical_entry_across_durations_max_error"] < 1e-10)
    check("no fitted meta-model", not analysis["validation_scatter"]["fitted_meta_model"]
          and analysis["validation_scatter"]["all30_models_retained"])
    print(json.dumps({"status": "passed", "checks": len(checks),
                      "scope": "saved Stage3 artifacts; no new training/test generation",
                      "checked": checks}, indent=2))


if __name__ == "__main__":
    main()
