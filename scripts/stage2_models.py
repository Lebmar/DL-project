"""Matched small neural Gaussian-scale models for the frozen Stage 2 protocol.

Both consume raw observed contexts and predict the next observation's log scale,
with mean fixed to zero. No dynamic hand-crafted summaries or normalization.
The TCN uses only left padding; its structural receptive field is exactly 64.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MODEL_CONFIG = {
    "gru_hidden": 24,
    "tcn_channels": 12,
    "tcn_kernels": [4, 3, 3, 3, 3],
    "tcn_dilations": [1, 2, 4, 8, 16],
    "tcn_activation": "relu",
    "log_scale_bounds": [-3.0, 3.0],
}


def observed_features(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 2 or observations.shape[1] < 1:
        raise ValueError("observed context must have shape [batch, time], time >= 1")
    return torch.stack((observations / 3.0, torch.log1p(observations.abs())), dim=-1)


class GaussianGRU(nn.Module):
    def __init__(self, hidden=24, log_scale_bounds=(-3.0, 3.0)):
        super().__init__()
        self.gru = nn.GRU(2, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.log_scale_bounds = tuple(log_scale_bounds)

    def forward_sequence(self, observations):
        """Output at position i depends only on observations through i."""
        encoded, _ = self.gru(observed_features(observations))
        return self.head(encoded).squeeze(-1).clamp(*self.log_scale_bounds)

    def forward(self, observed_context):
        return self.forward_sequence(observed_context)[:, -1]


class LeftCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation):
        super().__init__()
        self.left_padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class GaussianTCN(nn.Module):
    def __init__(self, channels=12, kernels=(4, 3, 3, 3, 3),
                 dilations=(1, 2, 4, 8, 16), log_scale_bounds=(-3.0, 3.0)):
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("one dilation is required for each kernel")
        modules = []
        for i, (kernel, dilation) in enumerate(zip(kernels, dilations)):
            modules.extend((LeftCausalConv(2 if i == 0 else channels, channels,
                                          kernel, dilation), nn.ReLU()))
        self.encoder = nn.Sequential(*modules)
        self.head = nn.Linear(channels, 1)
        self.log_scale_bounds = tuple(log_scale_bounds)
        self.receptive_field = 1 + sum((k - 1) * d for k, d in zip(kernels, dilations))

    def forward_sequence(self, observations):
        """Left padding preserves time length and excludes all future inputs."""
        features = observed_features(observations).transpose(1, 2)
        encoded = self.encoder(features).transpose(1, 2)
        return self.head(encoded).squeeze(-1).clamp(*self.log_scale_bounds)

    def forward(self, observed_context):
        return self.forward_sequence(observed_context)[:, -1]


def make_model(architecture: str, model_config: dict | None = None) -> nn.Module:
    config = DEFAULT_MODEL_CONFIG if model_config is None else model_config
    bounds = config["log_scale_bounds"]
    if architecture.lower() == "gru":
        return GaussianGRU(config["gru_hidden"], bounds)
    if architecture.lower() == "tcn":
        if config.get("tcn_activation", "relu") != "relu":
            raise ValueError("the specified TCN uses ReLU")
        return GaussianTCN(config["tcn_channels"], tuple(config["tcn_kernels"]),
                           tuple(config["tcn_dilations"]), bounds)
    raise ValueError(f"unknown architecture: {architecture}")


def gaussian_nll(log_scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Zero-mean Gaussian NLL, with the common constant omitted."""
    return (log_scale + 0.5 * target.square() * torch.exp(-2.0 * log_scale)).mean()


def load_checkpoint(path: str | Path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_model(payload["architecture"], payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def causal_prefix_audit(model: nn.Module, seed=91826, length=96, cut=37, batch=4):
    """Structural check only; never reads project train/validation/test data."""
    rng = np.random.default_rng(seed)
    original = torch.tensor(rng.normal(size=(batch, length)), dtype=torch.float32)
    changed = original.clone()
    changed[:, cut:] += torch.tensor(rng.normal(0, 20, size=(batch, length-cut)), dtype=torch.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        before = model.forward_sequence(original)
        after = model.forward_sequence(changed)
        prefix_only = model(original[:, :cut])
    suffix_error = float((before[:, :cut]-after[:, :cut]).abs().max())
    endpoint_error = float((before[:, cut-1]-prefix_only).abs().max())
    model.train(was_training)
    if suffix_error > 1e-6 or endpoint_error > 1e-6:
        raise AssertionError(f"causal prefix audit failed: {suffix_error}, {endpoint_error}")
    return {"suffix_intervention_max_abs_error": suffix_error,
            "prefix_endpoint_max_abs_error": endpoint_error,
            "passed": True, "cut": cut, "length": length,
            "parameters": sum(p.numel() for p in model.parameters()),
            "structural_receptive_field": getattr(model, "receptive_field", None)}

