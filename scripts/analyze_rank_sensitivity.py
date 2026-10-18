"""Analyze prespecified post-hoc rank-rule sensitivity, retaining every fit.

The uncertainty summaries condition on the already-used models and paths. A
crossed panel/shared-seed/path bootstrap is sensitivity, not a new confirmatory
claim. Randomized-rank values integrate interval metrics, not mean widths.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_rank_sensitivity import (COMMON, COMPONENTS, CONTEXTS, METRICS, OUT as IN,
    PART_KEYS, PHASES, PROTOCOL, RANKS, ROOT, RULES, WINDOWS, WORLDS, candidate_frame,
    committed, component_frame, load_protocol, rank_spec, reference_reproduction, sha)

OUT = ROOT / "outputs/rank_sensitivity_analysis"
ARCHITECTURES = ("gru", "tcn")
CONTRASTS = ("gru", "tcn", "gru_minus_tcn")
RULE_CONTRASTS = ("ceiling", "randomized90", "randomized90_minus_ceiling")


def frequencies(indices, size):
    weights = np.zeros((len(indices), size))
    np.add.at(weights, (np.arange(len(indices))[:, None], indices), 1 / indices.shape[1])
    return weights


def bootstrap_weights(panels, seeds, paths, replicates, seed):
    rng = np.random.default_rng(seed)
    path_indices = rng.integers(paths, size=(replicates, paths))
    panel_indices = rng.integers(panels, size=(replicates, panels))
    seed_indices = rng.integers(seeds, size=(replicates, seeds))
    pw = frequencies(path_indices, paths)
    panel_w, seed_w = frequencies(panel_indices, panels), frequencies(seed_indices, seeds)
    crossed = (panel_w[:, :, None, None] * seed_w[:, None, :, None] * pw[:, None, None, :]).reshape(replicates, -1)
    np.testing.assert_allclose(pw.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(crossed.sum(axis=1), 1, atol=1e-14)
    return pw, crossed, (path_indices, panel_indices, seed_indices)


def require_exact_grid(frame, protocol):
    model = protocol["models"]
    expected = pd.MultiIndex.from_product([model["panels"], protocol["test"]["families"], [128], [3.0],
        model["architectures"], model["seeds"], RULES, PHASES, COMPONENTS, range(protocol["test"]["paths"])], names=PART_KEYS)
    assert set(frame.columns) == set(PART_KEYS+METRICS)
    assert not frame.duplicated(PART_KEYS).any()
    actual = pd.MultiIndex.from_frame(frame[PART_KEYS])
    assert len(actual) == len(expected) and set(actual) == set(expected)
    assert np.isfinite(frame[METRICS].to_numpy()).all()


def cube(frame, protocol, family, phase):
    """Axes: panel, shared seed, path, rule, architecture, component, metric."""
    model = protocol["models"]
    keys = ["panel", "seed", "path", "rule", "architecture", "component"]
    index = pd.MultiIndex.from_product([model["panels"], model["seeds"], range(protocol["test"]["paths"]),
        RULES, ARCHITECTURES, COMPONENTS], names=keys)
    select = frame[(frame.family == family) & (frame.phase == phase)]
    assert not select.duplicated(keys).any()
    select = select.set_index(keys)
    assert len(select) == len(index) and set(select.index) == set(index)
    shape = (len(model["panels"]), len(model["seeds"]), protocol["test"]["paths"], 2, 2, len(COMPONENTS), len(METRICS))
    return select.reindex(index)[METRICS].to_numpy().reshape(shape)


def extend_contrasts(values):
    """Input [..., rule2, architecture2, component6, metric5]; exact paired deltas."""
    values = np.concatenate((values, (values[..., 0, :, :]-values[..., 1, :, :])[..., None, :, :]), axis=-3)
    return np.concatenate((values, (values[..., 1, :, :, :]-values[..., 0, :, :, :])[..., None, :, :, :]), axis=-4)


def tiny_self_check():
    rng = np.random.default_rng(618)
    x = rng.normal(size=(3, 5, 7, 2, 2, 6, 5))
    pw, cw, (paths, panels, seeds) = bootstrap_weights(3, 5, 7, 17, 211)
    flat = x.reshape(3*5*7, -1)
    fixed = (pw @ x.mean(axis=(0, 1)).reshape(7, -1)).reshape(17, 2, 2, 6, 5)
    crossed = (cw @ flat).reshape(17, 2, 2, 6, 5)
    slow_cross = np.array([np.mean([x[p, s, n] for p in pp for s in ss for n in nn], axis=0)
        for nn, pp, ss in zip(paths, panels, seeds)])
    slow_fixed = np.array([x[:, :, row].mean(axis=(0, 1, 2)) for row in paths])
    np.testing.assert_allclose(fixed, slow_fixed, rtol=0, atol=1e-14)
    np.testing.assert_allclose(crossed, slow_cross, rtol=0, atol=1e-14)
    contrasts = extend_contrasts(crossed)
    assert contrasts.shape == (17, 3, 3, 6, 5)
    direct = (crossed[:, 1, 0]-crossed[:, 1, 1])-(crossed[:, 0, 0]-crossed[:, 0, 1])
    np.testing.assert_allclose(contrasts[:, 2, 2], direct, atol=1e-14)
    # Asymmetric toy catches sign, axis and shared-seed mistakes.
    deterministic = np.empty((3, 5, 7, 2, 2, 6, 5))
    for p, s, n, r, a, c, m in product(range(3), range(5), range(7), range(2), range(2), range(6), range(5)):
        deterministic[p, s, n, r, a, c, m] = p+3*s+7*n+11*r+13*a+17*c+19*m+23*r*a
    cc = extend_contrasts(deterministic.mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(cc[2, 2], -23, atol=1e-12)
    fake = {"models": {"panels": [1, 2, 3], "seeds": [41, 42], "architectures": list(ARCHITECTURES)},
            "test": {"families": ["gaussian", "student_t5"], "paths": 3}}
    rows = [{**dict(zip(PART_KEYS, row)), **dict.fromkeys(METRICS, 1.0)} for row in product(
        fake["models"]["panels"], fake["test"]["families"], [128], [3.0], ARCHITECTURES,
        fake["models"]["seeds"], RULES, PHASES, COMPONENTS, range(3))]
    frame = pd.DataFrame(rows)
    require_exact_grid(frame.sample(frac=1, random_state=1), fake)
    bad = frame.copy()
    bad.loc[0, "path"] = 99
    rejected = 0
    for item in (frame.iloc[:-1], pd.concat((frame, frame.iloc[:1])), bad):
        try:
            require_exact_grid(item, fake)
        except AssertionError:
            rejected += 1
    assert rejected == 3
    return {"passed": True, "fixed_fit_bootstrap_scalar_loop_error": float(np.max(np.abs(fixed-slow_fixed))),
            "crossed_shared_seed_bootstrap_scalar_loop_error": float(np.max(np.abs(crossed-slow_cross))),
            "paired_rule_by_architecture_change_sign": -23,
            "missing_duplicate_and_offgrid_rejected": True}


def verify_candidates(protocol, evaluation, components, cells):
    """Rebuild every path effect and every cell from auditable per-rank metrics."""
    with np.load(IN / "rank_candidate_phase_metrics.npz", allow_pickle=False) as saved:
        expected_names = {"values", "model_panels", "model_architectures", "model_seeds", "families", "worlds",
            "current_contexts", "archive_contexts", "windows", "rank_choices", "phases", "metrics",
            "lower_ranks", "upper_ranks", "upper_probabilities"}
        assert set(saved.files) == expected_names
        for name, expected in (("families", protocol["test"]["families"]), ("worlds", WORLDS),
            ("current_contexts", CONTEXTS), ("archive_contexts", CONTEXTS), ("windows", WINDOWS),
            ("rank_choices", RANKS), ("phases", PHASES), ("metrics", METRICS)):
            assert saved[name].tolist() == list(expected)
        for name, j in (("lower_ranks", 0), ("upper_ranks", 1), ("upper_probabilities", 2)):
            np.testing.assert_allclose(saved[name], [rank_spec(w)[j] for w in WINDOWS], rtol=0, atol=0)
        keys = list(zip(saved["model_panels"].tolist(), saved["model_architectures"].tolist(), saved["model_seeds"].tolist()))
        assert len(keys) == len(set(keys)) == 30
        assert set(keys) == set(product(protocol["models"]["panels"], ARCHITECTURES, protocol["models"]["seeds"]))
        values = saved["values"]
        assert list(values.shape) == evaluation["candidate_shape"] == [30, 2, 2, 2, 2, 2, 2, 2, 128, 5]
        assert np.isfinite(values).all()
        coverage = values[..., METRICS.index("coverage")]
        assert (coverage >= 0).all() and (coverage <= 1).all()
        for m in METRICS:
            assert (values[..., METRICS.index(m)] >= 0).all()
        rebuilt_parts = component_frame(values, keys, protocol["test"])
        rebuilt_cells = candidate_frame(values, keys, protocol["test"])
    aa = components.set_index(PART_KEYS).sort_index()[METRICS]
    bb = rebuilt_parts.set_index(PART_KEYS).sort_index()[METRICS]
    assert aa.index.equals(bb.index)
    np.testing.assert_allclose(aa, bb, rtol=0, atol=1e-10)
    component_error = float(np.max(np.abs(aa.to_numpy()-bb.to_numpy())))
    ids = COMMON+["rule", "world", "current_context", "archive_context", "window", "phase"]
    aa = cells.set_index(ids).sort_index()[METRICS]
    bb = rebuilt_cells.set_index(ids).sort_index()[METRICS]
    assert not cells.duplicated(ids).any() and aa.index.equals(bb.index)
    np.testing.assert_allclose(aa, bb, rtol=0, atol=1e-10)
    cell_error = float(np.max(np.abs(aa.to_numpy()-bb.to_numpy())))
    stage3 = json.loads((ROOT / "outputs/stage3_evaluation/manifest.json").read_text())
    reference = reference_reproduction(components, cells, stage3)
    return {"component_reconstruction_max_error": component_error, "cell_reconstruction_max_error": cell_error,
            "ceiling_reference_reproduction": reference}


def summarize(components, protocol):
    spec, model = protocol["analysis"], protocol["models"]
    panel_ids, seed_ids = model["panels"], model["seeds"]
    n, b = protocol["test"]["paths"], spec["bootstrap_replicates"]
    pw, cw, _ = bootstrap_weights(len(panel_ids), len(seed_ids), n, b, spec["bootstrap_seed"])
    rows, per_panel, per_fit, leave_one = [], [], [], []
    for family, phase in product(protocol["test"]["families"], PHASES):
        x = cube(components, protocol, family, phase)
        # All columns use the exact same path and training draws, so subtraction is paired.
        fixed = (pw @ x.mean(axis=(0, 1)).reshape(n, -1)).reshape(b, 2, 2, len(COMPONENTS), len(METRICS))
        cross = (cw @ x.reshape(len(panel_ids)*len(seed_ids)*n, -1)).reshape(b, 2, 2, len(COMPONENTS), len(METRICS))
        fixed, cross = extend_contrasts(fixed), extend_contrasts(cross)
        estimate = extend_contrasts(x.mean(axis=(0, 1, 2)))
        fits = extend_contrasts(x.mean(axis=2))
        for ri, rule in enumerate(RULE_CONTRASTS):
            for ai, architecture in enumerate(CONTRASTS):
                for ci, component in enumerate(COMPONENTS):
                    for mi, metric in enumerate(METRICS):
                        reference = (family == "gaussian" and architecture == "gru_minus_tcn"
                                     and component == "total" and metric == "interval_score")
                        level = spec["reference_interval_level"] if reference else spec["descriptive_interval_level"]
                        q = [(1-level)/2, 1-(1-level)/2]
                        low, high = np.quantile(fixed[:, ri, ai, ci, mi], q)
                        cl, ch = np.quantile(cross[:, ri, ai, ci, mi], q)
                        common = dict(family=family, duration=128, amplitude=3.0, phase=phase, rule=rule,
                                      architecture=architecture, component=component, metric=metric)
                        selected = fits[:, :, ri, ai, ci, mi]
                        rows.append({**common, "estimate": estimate[ri, ai, ci, mi], "ci_low": low, "ci_high": high,
                            "crossed_sensitivity_low": cl, "crossed_sensitivity_high": ch, "interval_level": level,
                            "fixed_fits_per_architecture": 15, "paths": n, "fit_min": selected.min(), "fit_max": selected.max(),
                            "scope": "post-hoc paired sensitivity; fixed-fit path intervals; crossed training draws descriptive"})
                        for pi, panel in enumerate(panel_ids):
                            per_panel.append({**common, "panel": panel, "estimate": selected[pi].mean(),
                                "seed_std": selected[pi].std(ddof=1), "seed_min": selected[pi].min(), "seed_max": selected[pi].max()})
                            leave_one.append({**common, "omitted_panel": panel,
                                              "estimate": np.delete(selected, pi, axis=0).mean()})
                            for si, seed in enumerate(seed_ids):
                                per_fit.append({**common, "panel": panel, "seed": seed, "estimate": selected[pi, si]})
    tables = {"summary": pd.DataFrame(rows), "panel_summary": pd.DataFrame(per_panel),
              "individual_fits": pd.DataFrame(per_fit), "leave_one_panel_out": pd.DataFrame(leave_one)}
    tables["reference_architecture_contrasts"] = tables["summary"].query(
        "family == 'gaussian' and architecture == 'gru_minus_tcn' and component == 'total' and metric == 'interval_score'").copy()
    assert len(tables["summary"]) == 1080 and len(tables["reference_architecture_contrasts"]) == 6
    return tables


def plots(tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})
    data = tables["summary"]
    selected = data[(data.architecture == "gru_minus_tcn") & (data.component == "total") & (data.metric == "interval_score")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    labels = ["Ceiling rank", "Randomized rank", "Randomized minus ceiling"]
    for ai, family in enumerate(("gaussian", "student_t5")):
        for bi, phase in enumerate(PHASES):
            ax = axes[ai, bi]
            rows = selected[(selected.family == family) & (selected.phase == phase)].set_index("rule").loc[list(RULE_CONTRASTS)]
            for j, (_, row) in enumerate(rows.iterrows()):
                ax.plot([row.crossed_sensitivity_low, row.crossed_sensitivity_high], [j, j], color="#999999", linewidth=2)
                ax.plot([row.ci_low, row.ci_high], [j, j], color="#255f87", linewidth=5)
                ax.scatter([row.estimate], [j], color="#172f40", s=28, zorder=3)
            ax.axvline(0, color="#777777", linewidth=.8, linestyle=":")
            ax.set_yticks(range(3), labels)
            ax.invert_yaxis()
            ax.set_title(f"{'Gaussian' if family == 'gaussian' else 'Student-t5'} | {phase}")
            ax.set_xlabel("GRU J minus TCN J (interval-score units)")
            ax.grid(axis="x", alpha=.16)
    fig.suptitle("Sensitivity to finite quantile ranks | all 30 frozen models")
    fig.text(.02, .02, "Blue: fixed-fit paired-path intervals; grey: panel/shared-seed/path sensitivity. Gaussian 97.5%; t5 95%.\n"
             "Same previously viewed Stage 3 paths; randomization integrated exactly over interval metrics. No new confirmation.", fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(OUT / "rank_rule_architecture_contrasts.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    component_labels = ["Total J", "Current | archive64", "Archive | current16", "Current | archive16", "Archive | current64", "Third order"]
    for ax, phase in zip(axes, PHASES):
        rows = data[(data.family == "gaussian") & (data.phase == phase) & (data.architecture == "gru_minus_tcn") & (data.metric == "interval_score")]
        for rule, shift, color in (("ceiling", -.12, "#245b83"), ("randomized90", .12, "#ae5f22")):
            view = rows[rows.rule == rule].set_index("component").loc[list(COMPONENTS)]
            yy = np.arange(len(COMPONENTS))+shift
            ax.errorbar(view.estimate, yy, xerr=np.vstack((view.estimate-view.ci_low, view.ci_high-view.estimate)),
                        fmt="o", capsize=2, color=color, label=rule, markersize=4)
        ax.set_yticks(range(len(COMPONENTS)), component_labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=.8, linestyle=":")
        ax.set_title(phase)
        ax.set_xlabel("GRU minus TCN component (interval-score units)")
        ax.grid(axis="x", alpha=.16)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Both decomposition routes under both rank rules | Gaussian D128 A3")
    fig.text(.02, .015, "All fits retained. Fixed-fit paired-path intervals: total 97.5%; other components 95% descriptive.\n"
             "Components depend on decomposition order; correlated intervals do not add and are not causal percentages.", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .94))
    fig.savefig(OUT / "rank_rule_component_contrasts.png")
    plt.close(fig)


def markdown_summary(tables, protocol, checks):
    rows = tables["reference_architecture_contrasts"]
    lines = ["# Finite-rank sensitivity of the frozen Stage 3 study", "",
        "This is post-hoc sensitivity on the same already-viewed paths and all 30 saved models. No new training, model selection, or confirmatory test was performed.", "",
        "The ceiling rule uses ranks 30/32 and 117/128. The randomized90 rule chooses ranks 29/30 with probabilities .30/.70 for W32, and 116/117 with probabilities .90/.10 for W128. Expected interval metrics are computed exactly over those choices; the interval score is never evaluated at an averaged width.", "",
        "| Phase | Rank rule / paired change | GRU−TCN J | Fixed-fit path interval | Crossed sensitivity |",
        "|---|---|---:|---|---|"]
    for r in rows.itertuples():
        lines.append(f"| {r.phase} | {r.rule} | {r.estimate:+.6f} | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | [{r.crossed_sensitivity_low:+.6f}, {r.crossed_sensitivity_high:+.6f}] |")
    lines += ["", "Gaussian reference intervals are 97.5% and correspond to the two fixed phase endpoints within each rule/paired-change series. They do not provide simultaneous protection across rules, components, metrics, or the entire sensitivity exercise. All other intervals are 95% descriptive and unadjusted.", "",
        "The common ideal randomized rank level is 90% under continuous exchangeable scores. No coverage guarantee follows for these dependent nonstationary neural scores. This rule lowers both original finite-rank levels; it is not a unique causal correction or a percentage of the original effect explained by rank rounding. History length, number of observations, and empirical-quantile variability remain different.", "",
        "Fixed-fit intervals average all 15 fitted pairs before resampling the same 128 paths. Crossed sensitivity resamples three panels, five shared seed indices, and common paths, retaining the same seed draw across panels. It cannot establish a reliable training-population ranking from three panels.", "",
        "Coverage deficit is the expectation of each realized candidate interval's positive oracle deficit, and excess width is defined analogously. It is not a positive-part calculation after averaging candidates. Exact rank integration omits Monte Carlo variation from realizing the rank coins; it estimates the random procedure's expected performance.", "",
        "Ceiling reproduction: " + json.dumps(checks["ceiling_reference_reproduction"], ensure_ascii=False), "",
        "Files: summary.csv (all components/metrics/architecture and rule contrasts), panel_summary.csv, individual_fits.csv, leave_one_panel_out.csv, reference_architecture_contrasts.csv, cell_performance_summary.csv, and two figures. Per-path lower/upper candidate metrics remain in ../rank_sensitivity/rank_candidate_phase_metrics.npz for independent audit."]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check(), indent=2))
        return
    for file in (PROTOCOL, Path(__file__), ROOT / "scripts/run_rank_sensitivity.py", ROOT / "research/rank_sensitivity_protocol.md"):
        committed(file)
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite any rank-sensitivity analysis directory")
    protocol = load_protocol()
    manifest_path = IN / "manifest.json"
    evaluation = json.loads(manifest_path.read_text())
    assert evaluation["status"] == "complete" and evaluation["models_evaluated"] == 30
    assert evaluation["protocol_sha256"] == sha(PROTOCOL)
    assert evaluation["source_sha256"] == sha(ROOT / "scripts/run_rank_sensitivity.py")
    assert evaluation["analysis_source_sha256"] == sha(__file__)
    assert evaluation["protocol_document_sha256"] == sha(ROOT / "research/rank_sensitivity_protocol.md")
    for name, value in evaluation["artifacts"].items():
        assert sha(IN / name) == value, name
    started = time.time()
    components = pd.read_csv(IN / "paired_component_effects.csv.gz")
    cells = pd.read_csv(IN / "cell_performance_by_seed.csv")
    require_exact_grid(components, protocol)
    checks = verify_candidates(protocol, evaluation, components, cells)
    tables = summarize(components, protocol)
    ids = ["family", "duration", "amplitude", "architecture", "rule", "world", "current_context", "archive_context", "window", "phase"]
    tables["cell_performance_summary"] = cells.groupby(ids, sort=True)[METRICS].mean().reset_index()
    OUT.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    plots(tables)
    (OUT / "summary.md").write_text(markdown_summary(tables, protocol, checks))
    assert sha(PROTOCOL) == evaluation["protocol_sha256"] and sha(__file__) == evaluation["analysis_source_sha256"]
    manifest = {"status": "complete", "design_status": protocol["design_status"], "new_blind_test": False,
        "source_sha256": sha(__file__), "protocol_sha256": sha(PROTOCOL),
        "evaluation_manifest_sha256": sha(manifest_path), "self_check": tiny_self_check(), "checks": checks,
        "bootstrap": protocol["analysis"], "rows": {name: len(frame) for name, frame in tables.items()},
        "elapsed_seconds": time.time()-started,
        "artifacts": {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(tables["reference_architecture_contrasts"].to_string(index=False), flush=True)
    print(json.dumps({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"], "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
