"""Repair Stage 2 figure layout without recomputing any statistical CSV.

The original analysis script and its statistics remain immutable. This additive
rendering pass records its own provenance and a manually revised report hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from analyze_stage2 import ROOT, OUT, style, PALETTE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert sha(ROOT / "scripts/analyze_stage2.py") == manifest["analysis_source_sha256"]
    for name, value in manifest["artifact_sha256"].items():
        assert sha(OUT / name) == value, name
    csv_hashes = {p.name: sha(p) for p in OUT.glob("*.csv")}
    old_figure_hashes = {name: sha(OUT / name) for name in
                         ["full_grid_interactions.png", "reference_gaussian_curves.png"]}
    protocol = json.loads((ROOT / "configs/stage2_protocol.json").read_text())
    assert sha(ROOT / "configs/stage2_protocol.json") == manifest["protocol_sha256"]
    evaluation = ROOT / "outputs/stage2_evaluation"
    assert sha(evaluation / "reference_curves.csv.gz") == manifest["input_sha256"]["reference_curves.csv.gz"]
    secondary = pd.read_csv(OUT / "secondary_interactions.csv")
    contrast = pd.read_csv(OUT / "secondary_architecture_contrasts.csv")
    style()
    row_keys = [(f, a) for f in protocol["test"]["families"] for a in protocol["test"]["amplitudes"]]
    durations = protocol["test"]["durations"]
    matrices = []
    for phase in ["entry", "recovery64"]:
        for architecture in ["gru", "tcn", "gru-minus-tcn"]:
            selected = contrast if architecture == "gru-minus-tcn" else secondary[secondary.architecture == architecture]
            selected = selected[(selected.phase == phase) & (selected.metric == "interval_score")]
            lookup = selected.set_index(["family", "amplitude", "duration"])["estimate"]
            matrices.append(np.array([[lookup.loc[(f, a, d)] for d in durations] for f, a in row_keys]))
    limit = max(float(np.abs(matrices).max()), 1e-12)
    cmap = LinearSegmentedColormap.from_list("signed_blue_orange", ["#285a8a", "#ffffff", "#b56b2b"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True, sharey=True)
    # Reserve a separate axis; automatic colorbar placement was subsequently
    # invalidated by subplot_adjust in the first rendering pass.
    fig.subplots_adjust(left=.18, right=.85, top=.90, bottom=.14, hspace=.25, wspace=.12)
    for j, (ax, matrix) in enumerate(zip(axes.flat, matrices)):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit))
        for r in range(len(row_keys)):
            for c in range(len(durations)):
                ax.text(c, r, f"{matrix[r,c]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[r,c]) > .65*limit else "#222222")
        ax.set_xticks(range(len(durations)), durations)
        ax.set_yticks(range(len(row_keys)), [f"{f.replace('student_t5','t5').replace('sv_gaussian','SV')} | A={a:g}" for f, a in row_keys])
        ax.set_title(["GRU", "TCN", "GRU − TCN"][j % 3]+" | "+["Entry", "Recovery 64"][j//3])
        if j >= 3:
            ax.set_xlabel("Pulse duration (steps)")
    cax = fig.add_axes([.89, .23, .018, .56])
    fig.colorbar(image, cax=cax, label="Event-minus-control context × window interaction (IS units)")
    fig.suptitle("Complete frozen grid: 3 families × 2 amplitudes × 5 durations", y=.98)
    fig.text(.06, .015, "Five fixed fits, 128 paired paths. Secondary means are descriptive; complete intervals are in CSV.\n"
             "Entry for D≥32 uses the same first 32 observations; equal entry cells are not independent replications.\n"
             "Positive GRU − TCN values indicate a larger interaction, not a ranking of absolute performance.", fontsize=9)
    fig.savefig(OUT / "full_grid_interactions.png")
    plt.close(fig)

    curves = pd.read_csv(evaluation / "reference_curves.csv.gz")
    gaussian = curves[(curves.family == "gaussian") & (curves.window.isin([32, 128]))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ratios = []
    onset = protocol["test"]["onset"]
    for row, architecture in enumerate(["gru", "tcn"]):
        for context in [16, 64]:
            for window in [32, 128]:
                selected = gaussian[(gaussian.architecture == architecture) & (gaussian.context == context)
                    & (gaussian.window == window)].pivot(index="time", columns="world", values=["coverage", "width"])
                x = selected.index.to_numpy()-onset
                kwargs = {"color": PALETTE[context], "linestyle": "-" if window == 32 else "--", "linewidth": 1.2}
                label = f"context {context}, window {window}"
                axes[row, 0].plot(x, selected["coverage"]["event"], label=label, **kwargs)
                ratio = selected["width"]["event"]/selected["width"]["control"]
                ratios.append(ratio.to_numpy())
                axes[row, 1].plot(x, ratio, label=label, **kwargs)
        axes[row, 0].axhline(.9, color="#444444", linewidth=.8, linestyle=":")
        axes[row, 0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[row, 0].set_ylim(0, 1.03)
        axes[row, 1].axhline(1, color="#444444", linewidth=.8, linestyle=":")
        for col in range(2):
            axes[row, col].axvspan(0, 128, color="#dedede", alpha=.4)
            axes[row, col].axvline(320, color="#555555", linestyle=":", linewidth=.8)
            axes[row, col].set_title(architecture.upper()+" | "+["Empirical coverage", "Mean width: event/control"][col])
    common_min, common_max = min(r.min() for r in ratios), max(r.max() for r in ratios)
    for row in range(2):
        axes[row, 1].set_ylim(common_min*.9, common_max*1.04)
    axes[0, 0].legend(fontsize=8, ncol=2, loc="lower right")
    axes[1, 0].set_xlabel("Steps since pulse onset")
    axes[1, 1].set_xlabel("Steps since pulse onset")
    fig.text(.025, .015, "Gaussian A=3, D=128; means over five fixed fits and 128 paths. Shading marks the pulse.\n"
             "Dotted vertical line: exit + 192, the sufficient finite-context rolling-memory erasure bound. No temporal CI shown.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(OUT / "reference_gaussian_curves.png")
    plt.close(fig)
    assert csv_hashes == {p.name: sha(p) for p in OUT.glob("*.csv")}, "Statistical CSV changed"
    revision = {"script": "scripts/render_stage2_grid.py", "source_sha256": sha(Path(__file__)),
        "code_commit": subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "scripts/render_stage2_grid.py"], cwd=ROOT, text=True).strip(),
        "purpose": "layout-only repair; dedicated colorbar axis, coverage percent ticks and common width-ratio scale",
        "original_figure_sha256": old_figure_hashes, "statistical_csv_unchanged": True,
        "original_report_sha256": manifest["report_sha256"],
        "report_revision": "additive manual readout: initialization instability, shared-entry duration identity, absolute performance, supplement link; Markdown header repair"}
    manifest.setdefault("rendering_revisions", []).append(revision)
    manifest["artifact_sha256"] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest["report_sha256"] = sha(ROOT / "research/stage2_results.md")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"rendering_repair_complete": True, "statistical_csv_unchanged": True}))


if __name__ == "__main__":
    main()
