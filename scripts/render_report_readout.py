"""Render report figures/tables without changing completed Stage3 statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/stage3_analysis"
OUT = ROOT / "report"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value):
    return f"{value:+.4f}"


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["artifact_sha256"].items():
        assert sha(SOURCE / name) == expected
    OUT.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(SOURCE / "primary_contrasts.csv")
    panels = pd.read_csv(SOURCE / "primary_panel_contrasts.csv")
    components = pd.read_csv(SOURCE / "individual_seed_components.csv")
    means = pd.read_csv(SOURCE / "component_summary.csv")
    variance = pd.read_csv(SOURCE / "component_variability.csv")
    chosen = components[(components.family == "gaussian") & (components.duration == 128)
                        & (components.amplitude == 3) & (components.metric == "interval_score")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.65))
    colors = {"gru": "#24679e", "tcn": "#bf6138"}
    for ax, phase in zip(axes, ("entry", "recovery64")):
        for panel in (1, 2, 3):
            for arch, offset in (("gru", -.12), ("tcn", .12)):
                for component, marker, shift in (("current_at_archive64", "o", -.03),
                                                  ("archive_at_current16", "x", .03)):
                    part = chosen[(chosen.panel == panel) & (chosen.architecture == arch)
                                  & (chosen.phase == phase) & (chosen.component == component)].sort_values("seed")
                    assert len(part) == 5
                    ax.scatter(panel+offset+shift+np.linspace(-.024, .024, 5), part.estimate,
                               marker=marker, s=17, color=colors[arch], alpha=.8, linewidths=.8)
        ax.axhline(0, color="#999999", lw=.7)
        ax.set_xticks((1, 2, 3), ("1", "2", "3"))
        ax.set_xlabel("Training-data panel")
        ax.set_title("Entry" if phase == "entry" else "Recovery (64 steps)")
        ax.grid(axis="y", alpha=.12)
    axes[0].set_ylabel("Interval-score component")
    handles = [Line2D([], [], color=colors[a], marker="o", ls="", markersize=4, label=a.upper())
               for a in ("gru", "tcn")]
    handles += [Line2D([], [], color="#555555", marker=m, ls="", markersize=4, label=l)
                for m, l in (("o", "Current"), ("x", "Archive"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(.5, .002), columnspacing=1.1)
    fig.tight_layout(rect=(0, .12, 1, 1), pad=.6)
    figure = OUT / "stage3_components.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)
    lines = ["% Presentation only; source CSV statistics are unchanged."]
    for phase, prefix in (("entry", "Entry"), ("recovery64", "Recovery")):
        r = primary[primary.phase == phase].iloc[0]
        for suffix, column in (("Contrast", "estimate"), ("Low", "ci_low"), ("High", "ci_high"),
                               ("CrossLow", "crossed_sensitivity_low"), ("CrossHigh", "crossed_sensitivity_high")):
            lines.append("\\newcommand{\\"+prefix+suffix+"}{"+fmt(r[column])+"}")
    endrow = r" \\"
    lines.extend([r"\newcommand{\PanelReadout}{%", r"\begin{tabular}{lrrr}", r"\toprule",
                  "Phase & Panel 1 & Panel 2 & Panel 3"+endrow, r"\midrule"])
    for phase, label in (("entry", "Entry"), ("recovery64", "Recovery64")):
        d = panels[panels.phase == phase].set_index("panel")
        lines.append(label+" & "+" & ".join(fmt(d.loc[p, "estimate"]) for p in (1, 2, 3))+endrow)
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    lines.extend([r"\newcommand{\ComponentReadout}{%", r"\begin{tabular}{llrrr}", r"\toprule",
                  "Model & Phase & Total & Current & Archive"+endrow, r"\midrule"])
    d = means[(means.family == "gaussian") & (means.duration == 128) & (means.amplitude == 3)
              & (means.metric == "interval_score") & means.phase.isin(("entry", "recovery64"))]
    for arch in ("gru", "tcn"):
        for phase, label in (("entry", "Entry"), ("recovery64", "Recovery64")):
            row = d[(d.architecture == arch) & (d.phase == phase)].set_index("component")
            lines.append(arch.upper()+" & "+label+" & "+" & ".join(fmt(row.loc[c, "estimate"])
                         for c in ("total", "current_at_archive64", "archive_at_current16"))+endrow)
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    d = variance[(variance.family == "gaussian") & (variance.duration == 128) & (variance.amplitude == 3)
                 & (variance.phase == "entry") & (variance.metric == "interval_score")]
    count = int((d.tcn_current_std_greater_than_archive & d.tcn_current_std_greater_than_gru).sum())
    lines.append("\\newcommand{\\VariabilityCount}{"+str(count)+"}")
    lines.append("\\newcommand{\\VariabilityComparisons}{"+str(len(d))+"}")
    readout = OUT / "readout.tex"
    readout.write_text("\n".join(lines)+"\n")
    output = {"operation": "presentation only", "source_sha256": sha(__file__),
              "analysis_manifest_sha256": sha(SOURCE / "manifest.json"),
              "inputs": {name: sha(SOURCE / name) for name in ("primary_contrasts.csv", "primary_panel_contrasts.csv",
                  "individual_seed_components.csv", "component_summary.csv", "component_variability.csv")},
              "artifacts": {p.name: sha(p) for p in (figure, readout)},
              "scope": "Fixed Gaussian D128/A3 reference, all30 fits; two-route counts correlated and descriptive"}
    (OUT / "readout_manifest.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"status": "rendered", "statistics_changed": False}))


if __name__ == "__main__":
    main()
