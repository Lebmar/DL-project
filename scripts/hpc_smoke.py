"""Small synthetic runtime check for the frozen Stage 2 neural model module.

This is an environment smoke test, not a scientific experiment or benchmark.
It never reads project datasets, checkpoints, or saved experimental results.
Only an explicitly named, previously nonexistent JSON output is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import stage2_models


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SEED = 674023119  # Synthetic smoke only; distinct from project science seeds.
STEPS = 5
BATCH_SIZE = 8
EXPECTED_PARAMETERS = {"gru": 2041, "tcn": 1897}
CPU_CUDA_RTOL = 2e-4
CPU_CUDA_ATOL = 2e-5
SLURM_FIELDS = (
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_NNODES",
    "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_GPUS_ON_NODE",
)


def bounded_threads(text):
    value = int(text)
    if not 1 <= value <= 16:
        raise argparse.ArgumentTypeError("--threads must be between 1 and 16")
    return value


def module_provenance():
    """Read only the relevant module's bytes and its latest Git commit."""
    module = Path(stage2_models.__file__).resolve()
    relative = module.relative_to(ROOT).as_posix()
    result = {"path": relative,
              "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
              "last_committed_change": None}
    if shutil.which("git"):
        try:
            git = subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", relative],
                cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
            )
            value = git.stdout.strip()
            if git.returncode == 0 and len(value) == 40 and all(
                    character in "0123456789abcdef" for character in value):
                result["last_committed_change"] = value
        except (OSError, subprocess.TimeoutExpired):
            pass
    return result


def nvidia_inventory():
    """Benign hardware/driver query only: no process, user, UUID, or path data."""
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    fields = ["name", "driver_version", "memory.total"]
    command = ["nvidia-smi", "--query-gpu=" + ",".join(fields),
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=10, check=False)
    except subprocess.TimeoutExpired:
        return {"available": True, "status": "timeout", "query_fields": fields}
    except OSError:
        return {"available": True, "status": "query_failed", "query_fields": fields}
    if result.returncode:
        return {"available": True, "status": "query_failed",
                "return_code": result.returncode, "query_fields": fields}
    devices = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            return {"available": True, "status": "unexpected_query_format",
                    "query_fields": fields}
        devices.append(dict(zip(fields, values)))
    return {"available": True, "status": "ok", "query_fields": fields,
            "memory_unit": "MiB", "visible_to_nvidia_smi": devices,
            "note": "Inventory may include GPUs outside the process CUDA allocation."}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def assert_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite " + label)


def train_synthetic(architecture, context, seed, device, batches, targets):
    torch.manual_seed(seed)
    model = stage2_models.make_model(architecture).to(device=device, dtype=torch.float32)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETERS[architecture]:
        raise AssertionError("Unexpected frozen model parameter count: " + architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    losses, gradient_maxima = [], []
    model.train()
    for step in range(STEPS):
        x = torch.tensor(batches[step], device=device, dtype=torch.float32)
        y = torch.tensor(targets[step], device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        if prediction.dtype != torch.float32 or prediction.shape != y.shape:
            raise AssertionError("Unexpected forward dtype or shape")
        assert_finite(prediction, "forward prediction")
        loss = stage2_models.gaussian_nll(prediction, y)
        assert_finite(loss, "loss")
        loss.backward()
        maxima = []
        for parameter in model.parameters():
            if parameter.grad is None:
                raise AssertionError("A trainable parameter has no gradient")
            assert_finite(parameter.grad, "gradient")
            maxima.append(float(parameter.grad.detach().abs().max().cpu()))
        optimizer.step()
        for parameter in model.parameters():
            assert_finite(parameter, "updated parameter")
        losses.append(float(loss.detach().cpu()))
        gradient_maxima.append(max(maxima))
    model.eval()
    return model, {"losses_without_gaussian_constant": losses,
                   "max_abs_gradient_by_step": gradient_maxima,
                   "parameter_count": count, "context": context,
                   "steps": STEPS, "batch_size": BATCH_SIZE,
                   "all_forward_losses_gradients_parameters_finite": True}


def state_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_one(architecture, context, device):
    seed = SMOKE_SEED + context + (1000 if architecture == "tcn" else 0)
    rng = np.random.default_rng(seed)
    batches = rng.normal(size=(STEPS, BATCH_SIZE, context)).astype(np.float32)
    targets = rng.normal(size=(STEPS, BATCH_SIZE)).astype(np.float32)
    synchronize(device)
    started = time.perf_counter()
    model, diagnostics = train_synthetic(architecture, context, seed, device, batches, targets)
    repeated, repeated_diagnostics = train_synthetic(
        architecture, context, seed, device, batches, targets)
    first_hash, second_hash = state_digest(model), state_digest(repeated)
    exact_state = all(torch.equal(value, repeated.state_dict()[key])
                      for key, value in model.state_dict().items())
    if not exact_state or diagnostics != repeated_diagnostics or first_hash != second_hash:
        raise AssertionError("Fixed-seed same-device synthetic replay was not exact")
    # The frozen audit helper constructs CPU inputs. Explicitly use a CPU copy.
    cpu_model = stage2_models.make_model(architecture).to(dtype=torch.float32)
    cpu_model.load_state_dict({key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()})
    cpu_model.eval()
    causal = stage2_models.causal_prefix_audit(cpu_model, seed=seed + 10000)
    comparison = {"performed": False, "reason": "CPU-only smoke requested"}
    if device.type == "cuda":
        probe = torch.tensor(rng.normal(size=(12, context)), dtype=torch.float32)
        with torch.no_grad():
            cpu_log_scale = cpu_model(probe)
            cuda_log_scale = model(probe.to(device)).cpu()
        comparisons = {}
        for label, cpu_value, cuda_value in (
                ("log_scale", cpu_log_scale, cuda_log_scale),
                ("scale", cpu_log_scale.exp(), cuda_log_scale.exp())):
            assert_finite(cpu_value, "CPU " + label)
            assert_finite(cuda_value, "CUDA " + label)
            passed = torch.allclose(cpu_value, cuda_value,
                                    rtol=CPU_CUDA_RTOL, atol=CPU_CUDA_ATOL)
            comparisons[label] = {"max_abs_error": float((cpu_value-cuda_value).abs().max()),
                                  "passed": bool(passed)}
            if not passed:
                raise AssertionError("Same-weight CPU/CUDA prediction mismatch: " + label)
        comparison = {"performed": True, "same_weights": True, "rtol": CPU_CUDA_RTOL,
                      "atol": CPU_CUDA_ATOL, "outputs": comparisons,
                      "note": "Prediction agreement only; CPU/CUDA training need not be bitwise equal."}
    synchronize(device)
    return {"architecture": architecture, "seed": seed, **diagnostics,
            "same_device_replay_exact": True, "final_parameter_sha256": first_hash,
            "cpu_causal_prefix_audit": causal, "cpu_cuda_same_weight_comparison": comparison,
            "elapsed_seconds_not_benchmark": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=bounded_threads, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; refusing to overwrite")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocated.isdigit() and args.threads > int(allocated):
        parser.error("Requested threads exceed SLURM_CPUS_PER_TASK")

    # Process-local settings only. Set workspace configuration before CUDA use.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    report = {
        "schema": "hpc_synthetic_runtime_smoke_v1", "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_device": args.device,
        "scope": "Synthetic runtime smoke only; not a benchmark or scientific result",
        "project_data_or_checkpoints_read": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "torch_compiled_cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version()},
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "slurm": {field: os.environ[field] for field in SLURM_FIELDS if field in os.environ},
        "runtime": {"threads": torch.get_num_threads(), "interop_threads": 1,
                    "float_dtype": "float32", "deterministic_algorithms": True,
                    "cudnn_benchmark": False, "cudnn_deterministic": True,
                    "matmul_tf32": False, "cudnn_tf32": False,
                    "float32_matmul_precision": "highest",
                    "cublas_workspace_config_process_only": ":4096:8"},
        "frozen_model_module": module_provenance(),
        "nvidia_smi": nvidia_inventory(), "checks": [],
    }
    started = time.perf_counter()
    exit_code = 0
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            report["cuda_device"] = {
                "name": properties.name, "capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "process_device_index": torch.cuda.current_device(),
                "process_visible_device_count": torch.cuda.device_count(),
            }
        for architecture in ("gru", "tcn"):
            for context in (16, 64):
                report["checks"].append(check_one(architecture, context, device))
        report["status"] = "passed"
    except Exception as error:
        # Do not dump a traceback with user/home paths into shared output.
        report["status"] = "failed"
        report["failure_type"] = type(error).__name__
        report["failure_message"] = str(error).splitlines()[0][:500]
        for value in (str(Path.home()), os.environ.get("USER", ""),
                      os.environ.get("LOGNAME", "")):
            if value:
                report["failure_message"] = report["failure_message"].replace(value, "[redacted]")
        exit_code = 1
    report["elapsed_seconds_not_benchmark"] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process creating it meanwhile.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "device": args.device,
                      "completed_checks": len(report["checks"]),
                      "elapsed_seconds_not_benchmark": report["elapsed_seconds_not_benchmark"]}))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
