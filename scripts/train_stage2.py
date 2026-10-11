"""Train one preregistered matched GRU/TCN panel after protocol freeze.

Requires --protocol JSON with `frozen: true` and the full `training_config` below.
Use --print-config to inspect the exact default training configuration. This
script never loads a test set and makes no comparison based on test performance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage2_models import DEFAULT_MODEL_CONFIG, causal_prefix_audit, gaussian_nll, make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stage2_training"
DEFAULT_CONFIG = {
    "architectures": ["gru", "tcn"],
    "model_seeds": [4101, 4102, 4103, 4104, 4105],
    "contexts": [16, 64],
    "model": copy.deepcopy(DEFAULT_MODEL_CONFIG),
    "training_data": {"seed": 314159, "paths": 256, "length": 1024},
    "validation_data": {"seed": 314160, "paths": 64, "length": 1024,
                        "target_pairs": 8192, "pair_seed": 314161},
    "generator": {"switch_probability": 0.018, "temporary_probability": 0.015,
                  "persistent_scale_min": 0.5, "persistent_scale_max": 3.0,
                  "temporary_multiplier": 3.0},
    "optimization": {"batch_size": 128, "batches_per_epoch": 32,
                     "max_epochs": 60, "min_epochs": 20, "patience": 8,
                     "min_delta": 0.0001, "learning_rate": 0.002,
                     "gradient_clip_norm": 5.0, "validation_batch_size": 512},
    "execution": {"device": "cpu", "torch_threads": 4},
}


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def generate_paths(config, generator):
    """The explicitly specified piecewise Gaussian pilot generator, no oracle labels returned."""
    rng = np.random.default_rng(config["seed"])
    n, length = config["paths"], config["length"]
    persistent = np.ones(n)
    sigma = np.empty((n, length))
    for t in range(length):
        switches = rng.random(n) < generator["switch_probability"]
        persistent[switches] = np.exp(rng.uniform(np.log(generator["persistent_scale_min"]),
                                                 np.log(generator["persistent_scale_max"]), switches.sum()))
        temporary = np.where(rng.random(n) < generator["temporary_probability"],
                             generator["temporary_multiplier"], 1.0)
        sigma[:, t] = persistent*temporary
    return sigma*rng.normal(size=(n, length))


def context_tensor(observations, paths, targets, context, device):
    indices = targets[:, None] + np.arange(-context, 0)[None, :]
    return torch.tensor(observations[paths[:, None], indices], dtype=torch.float32, device=device)


def validate(model, val_x, val_target, batch_size):
    model.eval()
    values = {}
    with torch.no_grad():
        for context, x in val_x.items():
            total = 0.0
            for start in range(0, len(x), batch_size):
                stop = min(start+batch_size, len(x))
                loss = gaussian_nll(model(x[start:stop]), val_target[start:stop])
                total += float(loss)*(stop-start)
            values[context] = total/len(x)
    return values


def read_protocol(path):
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("frozen") is not True:
        raise ValueError("protocol must explicitly contain frozen: true before training")
    config = payload.get("training_config")
    if config is None:
        raise ValueError("protocol must contain the full training_config; inspect --print-config")
    # No silent default substitution: all parameters are in the frozen file.
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("training_config keys must match --print-config")
    for section in ["model", "training_data", "validation_data", "generator", "optimization", "execution"]:
        if set(config[section]) != set(DEFAULT_CONFIG[section]):
            raise ValueError(f"section {section} keys must match --print-config")
    if config["contexts"] != [16, 64] or config["architectures"] != ["gru", "tcn"]:
        raise ValueError("matched panel must use contexts [16,64] and architectures [gru,tcn]")
    opt = config["optimization"]
    if not (1 <= opt["min_epochs"] <= opt["max_epochs"]):
        raise ValueError("invalid epoch limits")
    if opt["batches_per_epoch"] % len(config["contexts"]):
        raise ValueError("batches must balance both contexts exactly")
    if config["execution"]["device"] != "cpu":
        raise ValueError("this preregistered local implementation currently uses CPU")
    return config, payload, raw, hashlib.sha256(raw).hexdigest()


def train_one(architecture, seed, config, train_y, val_x, val_target, protocol_hash):
    opt = config["optimization"]
    device = config["execution"]["device"]
    torch.manual_seed(seed)
    model = make_model(architecture, config["model"]).to(device)
    audit_before = causal_prefix_audit(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt["learning_rate"])
    # This RNG is separate from torch/model initialization. Reinitializing it
    # for each architecture gives exact matched contexts/path/target batches.
    sampling_rng = np.random.default_rng(seed)
    best_loss, stopping_reference, stale = float("inf"), float("inf"), 0
    best_epoch, best_context_losses = None, None
    checkpoint = OUT / "checkpoints" / f"{architecture}_seed_{seed}.pt"
    history = []
    started = time.time()
    for epoch in range(1, opt["max_epochs"]+1):
        model.train()
        schedule = sampling_rng.permutation(np.resize(np.array(config["contexts"]),
                                                      opt["batches_per_epoch"]))
        batch_digest = hashlib.sha256()
        training_losses = []
        for context in schedule:
            paths = sampling_rng.integers(0, len(train_y), opt["batch_size"])
            targets = sampling_rng.integers(max(config["contexts"]), train_y.shape[1], opt["batch_size"])
            batch_digest.update(np.array([context], dtype="<i8").tobytes())
            batch_digest.update(paths.astype("<i8").tobytes())
            batch_digest.update(targets.astype("<i8").tobytes())
            x = context_tensor(train_y, paths, targets, int(context), device)
            y = torch.tensor(train_y[paths, targets], dtype=torch.float32, device=device)
            loss = gaussian_nll(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss: {architecture} seed{seed}, epoch{epoch}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["gradient_clip_norm"])
            optimizer.step()
            training_losses.append(float(loss.detach()))
        context_losses = validate(model, val_x, val_target, opt["validation_batch_size"])
        mean_loss = float(np.mean(list(context_losses.values())))
        # Selection always uses the absolute minimum validation mean, even
        # when the improvement is smaller than the patience min_delta.
        if mean_loss < best_loss:
            best_loss, best_epoch, best_context_losses = mean_loss, epoch, context_losses.copy()
            torch.save({"architecture": architecture, "model_seed": seed,
                        "model_config": config["model"], "contexts": config["contexts"],
                        "state_dict": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        "selected_epoch": epoch, "best_validation_mean_nll": best_loss,
                        "validation_nll_by_context": context_losses,
                        "protocol_sha256": protocol_hash}, checkpoint)
        if mean_loss < stopping_reference-opt["min_delta"]:
            stopping_reference, stale = mean_loss, 0
        else:
            stale += 1
        row = {"architecture": architecture, "model_seed": seed, "epoch": epoch,
               "train_nll_without_constant": float(np.mean(training_losses)),
               "validation_nll_context16": context_losses[16],
               "validation_nll_context64": context_losses[64],
               "mean_validation_nll": mean_loss, "best_validation_nll": best_loss,
               "patience_counter": stale, "minibatch_epoch_sha256": batch_digest.hexdigest(),
               "elapsed_seconds": time.time()-started}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / "history" / f"{architecture}_seed_{seed}.csv", index=False)
        print(json.dumps({"architecture": architecture, "seed": seed, "epoch": epoch,
                          "validation_nll": mean_loss, "best_epoch": best_epoch,
                          "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        if epoch >= opt["min_epochs"] and stale >= opt["patience"]:
            break
    saved = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    audit_after = causal_prefix_audit(model)
    return history, {"architecture": architecture, "model_seed": seed,
                     "epochs_executed": len(history), "selected_epoch": best_epoch,
                     "best_validation_mean_nll": best_loss,
                     "selected_validation_nll_by_context": best_context_losses,
                     "checkpoint": str(checkpoint.relative_to(ROOT)),
                     "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "causal_audit_before": audit_before, "causal_audit_selected_checkpoint": audit_after,
                     "wall_seconds": time.time()-started,
                     "stop_reason": "max_epochs" if len(history) == opt["max_epochs"] else "validation_patience"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, help="frozen JSON with training_config")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return
    if args.protocol is None:
        parser.error("training requires --protocol after the protocol has been frozen")
    config, protocol, protocol_raw, protocol_hash = read_protocol(args.protocol)
    # Avoid silently replacing a previous panel or treating a partial panel as
    # complete. A fresh output directory is required for this training run.
    if (OUT/"manifest.json").exists():
        raise FileExistsError("stage2_training already has a manifest; inspect it before a new run")
    for folder in [OUT, OUT/"checkpoints", OUT/"history"]:
        folder.mkdir(parents=True, exist_ok=True)
    (OUT/"frozen_protocol.json").write_bytes(protocol_raw)
    torch.set_num_threads(config["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.time()
    train_y = generate_paths(config["training_data"], config["generator"])
    val_y = generate_paths(config["validation_data"], config["generator"])
    val_config = config["validation_data"]
    pair_rng = np.random.default_rng(val_config["pair_seed"])
    pair_paths = pair_rng.integers(0, len(val_y), val_config["target_pairs"])
    pair_targets = pair_rng.integers(max(config["contexts"]), val_y.shape[1], val_config["target_pairs"])
    val_x = {c: context_tensor(val_y, pair_paths, pair_targets, c, config["execution"]["device"])
             for c in config["contexts"]}
    val_target = torch.tensor(val_y[pair_paths, pair_targets], dtype=torch.float32)
    np.savez_compressed(OUT/"validation_target_pairs.npz", paths=pair_paths, targets=pair_targets)
    source_paths = [Path(__file__), ROOT/"scripts/stage2_models.py"]
    manifest = {"status": "training", "date": "2026-10-11", "python": platform.python_version(),
                "numpy": np.__version__, "torch": torch.__version__,
                "protocol_source": str(args.protocol.resolve()), "protocol_sha256": protocol_hash,
                "training_config": config,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                "data_sha256": {"train": array_sha256(train_y), "validation": array_sha256(val_y),
                                "validation_pair_paths": array_sha256(pair_paths),
                                "validation_pair_targets": array_sha256(pair_targets)},
                "unit_scale_validation_nll_without_constant": float(.5*val_target.square().mean()),
                "selection": "minimum mean validation NLL across contexts16/64, no test data loaded",
                "early_stop": "min_delta affects patience only; counter starts at epoch1, stopping enabled after min_epochs",
                "matching": "independent NumPy sampler reset to model seed for each architecture; identical minibatch prefixes",
                "models": [], "matching_minibatches_verified": False,
                "limitations": ["fixed zero mean and a simulation-trained Gaussian scale model",
                                "context length does not establish effective learned memory",
                                "GRU and TCN parameter counts are close, not equal",
                                "validation targets are shared draws within64 independent paths, not8192 independent series"]}
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    histories = {}
    for seed in config["model_seeds"]:
        for architecture in config["architectures"]:
            history, result = train_one(architecture, seed, config, train_y, val_x, val_target, protocol_hash)
            histories[(architecture, seed)] = history
            manifest["models"].append(result)
            manifest["wall_seconds"] = time.time()-started
            (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        gru_history = histories[("gru", seed)]
        tcn_history = histories[("tcn", seed)]
        for a,b in zip(gru_history,tcn_history):
            if a["minibatch_epoch_sha256"] != b["minibatch_epoch_sha256"]:
                raise AssertionError(f"unmatched minibatches at seed{seed}, epoch{a['epoch']}")
    manifest["matching_minibatches_verified"] = True
    manifest["status"] = "complete"
    manifest["wall_seconds"] = time.time()-started
    manifest["completed_models"] = len(manifest["models"])
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    all_history = [r for h in histories.values() for r in h]
    pd.DataFrame(all_history).to_csv(OUT/"training_history.csv", index=False)
    pd.DataFrame([{k:v for k,v in model.items() if not isinstance(v,(dict,list))}
                  for model in manifest["models"]]).to_csv(OUT/"model_summary.csv", index=False)
    print(json.dumps({"status": "complete", "models": manifest["completed_models"],
                      "matched_minibatches": True, "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
