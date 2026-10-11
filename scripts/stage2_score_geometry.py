"""Exploratory expected interval-score geometry of already viewed Stage 2 paths.

No fitting, new holdout or replacement of the frozen primary analysis occurs.
Integrate the contemporaneous independent innovation under the simulator's
known Gaussian/t5 law, retaining each issued interval and its noisy history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm, t as student

from evaluate_stage2 import ROOT, phases, rolling_half_width


OUT = ROOT / "outputs/stage2_score_geometry"
EVAL = ROOT / "outputs/stage2_evaluation"
ALPHA = .1
DF = 5
CONTEXTS = (16, 64)
WINDOWS = (32, 128)
ARCHITECTURES = ("gru", "tcn")
SEED = 4101


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_score(h, tau, family, alpha=ALPHA, df=DF):
    """E[2h+2/alpha*(|Y|-h)+] for Y=tau times unit-variance noise."""
    h, tau = np.asarray(h, dtype=float), np.asarray(tau, dtype=float)
    if (h < 0).any() or (tau <= 0).any():
        raise ValueError("Nonnegative half-width and positive true scale required")
    if family == "gaussian":
        z = h/tau
        tail_excess_half = tau*norm.pdf(z)-h*norm.sf(z)
    elif family == "student_t5":
        unit = np.sqrt((df-2)/df)
        a = h/(tau*unit)
        tail_excess_half = tau*unit*(df+a*a)/(df-1)*student.pdf(a, df)-h*student.sf(a, df)
    else:
        raise ValueError(family)
    return 2*h+(4/alpha)*tail_excess_half


def derivatives(h, tau, family, alpha=ALPHA, df=DF):
    """R_h, R_hh, and d_s d_q R(s*q) = R_h + h*R_hh."""
    h, tau = np.asarray(h, dtype=float), np.asarray(tau, dtype=float)
    if family == "gaussian":
        x, unit = h/tau, 1.
        density, survival = norm.pdf(x), norm.sf(x)
    elif family == "student_t5":
        unit = np.sqrt((df-2)/df)
        x = h/(tau*unit)
        density, survival = student.pdf(x, df), student.sf(x, df)
    else:
        raise ValueError(family)
    first = 2-(4/alpha)*survival
    second = (4/alpha)*density/(tau*unit)
    return first, second, first+h*second


def rectangle_integral(s0, s1, q0, q1, tau, family):
    """Oriented Gauss-Legendre integral of d_s d_q R over a rectangle."""
    nodes, weights = np.polynomial.legendre.leggauss(48)
    ss = (s1-s0)*nodes/2+(s0+s1)/2
    qs = (q1-q0)*nodes/2+(q0+q1)/2
    kk = derivatives(ss[:, None]*qs[None, :], tau, family)[2]
    return float((s1-s0)*(q1-q0)/4*np.sum(weights[:, None]*weights[None, :]*kk))


def self_check():
    """Independent quadrature, finite differences and integration identities."""
    checks, roots, maxima = [], {}, {"quadrature_absolute_error": 0.,
                                   "first_derivative_absolute_error": 0.,
                                   "second_derivative_absolute_error": 0.,
                                   "mixed_derivative_absolute_error": 0.,
                                   "rectangle_integral_absolute_error": 0.}
    for family in ("gaussian", "student_t5"):
        unit = 1. if family == "gaussian" else np.sqrt((DF-2)/DF)
        for tau in (.4, 1., 3.):
            for z in (0., .05, .3, .8, 1.64, 3., 6.):
                h = tau*z
                pdf = (lambda x: norm.pdf(x/tau)/tau) if family == "gaussian" else (
                    lambda x: student.pdf(x/(tau*unit), DF)/(tau*unit))
                excess = 2*quad(lambda x: (x-h)*pdf(x), h, np.inf,
                                epsabs=1e-11, epsrel=1e-11)[0]
                direct = 2*h+2/ALPHA*excess
                analytic = float(expected_score(h, tau, family))
                err = abs(direct-analytic)
                maxima["quadrature_absolute_error"] = max(maxima["quadrature_absolute_error"], err)
                np.testing.assert_allclose(analytic, direct, rtol=1e-10, atol=1e-10)
                checks.append({"family": family, "tau": tau, "h_over_tau": z,
                               "analytic_expected_score": analytic, "integrated_expected_score": direct,
                               "absolute_error": err})
                if z == 0:
                    continue
                step = 1e-4*tau
                first, second, mixed = derivatives(h, tau, family)
                plus, minus = expected_score(h+step, tau, family), expected_score(h-step, tau, family)
                fd_first = (plus-minus)/(2*step)
                fd_second = (plus-2*analytic+minus)/(step*step)
                s, q = tau*.8, z/.8
                ds, dq = 1e-4*s, 1e-4*q
                f = lambda ss, qq: expected_score(ss*qq, tau, family)
                fd_mixed = (f(s+ds, q+dq)-f(s+ds, q-dq)-f(s-ds, q+dq)+f(s-ds, q-dq))/(4*ds*dq)
                for label, true, approx in (("first", first, fd_first), ("second", second, fd_second),
                                            ("mixed", mixed, fd_mixed)):
                    error = abs(float(true-approx))
                    maxima[f"{label}_derivative_absolute_error"] = max(maxima[f"{label}_derivative_absolute_error"], error)
                    np.testing.assert_allclose(true, approx, rtol=2e-5, atol=2e-4)
        root = brentq(lambda z: derivatives(z, 1., family)[2], 0., 1.5)
        optimal = norm.ppf(1-ALPHA/2) if family == "gaussian" else unit*student.ppf(1-ALPHA/2, DF)
        np.testing.assert_allclose(derivatives(optimal, 1., family)[0], 0, atol=1e-12)
        roots[family] = {"mixed_derivative_zero_h_over_tau": root, "optimal_h_over_tau": float(optimal)}
        for s0, s1, q0, q1, tau in ((.8, 1.3, .7, 1.9, 1.),
                                    (1.3, .8, .7, 1.9, 1.),
                                    (.8, 1.3, 1.9, .7, 1.),
                                    (.15, .35, .2, .5, 1.),
                                    (.8, 1.3, .7, 1.9, 3.)):
            contrast = float(expected_score(s1*q1, tau, family)-expected_score(s0*q1, tau, family)
                             -expected_score(s1*q0, tau, family)+expected_score(s0*q0, tau, family))
            integral = rectangle_integral(s0, s1, q0, q1, tau, family)
            err = abs(contrast-integral)
            maxima["rectangle_integral_absolute_error"] = max(maxima["rectangle_integral_absolute_error"], err)
            np.testing.assert_allclose(contrast, integral, rtol=1e-10, atol=1e-10)
    # Same increasing s/q rectangle has opposite signs in two score regions.
    examples = []
    for family in ("gaussian", "student_t5"):
        for region, s0, s1, q0, q1 in (("very narrow", .15, .35, .2, .5),
                                      ("near nominal", .9, 1.1, 1.4, 1.8)):
            value = rectangle_integral(s0, s1, q0, q1, 1., family)
            assert (value < 0) if region == "very narrow" else (value > 0)
            examples.append({"family": family, "region": region, "s0": s0, "s1": s1,
                             "q0": q0, "q1": q1, "tau": 1., "current_by_window_expected_score": value})
    return {"passed": True, "alpha": ALPHA, "student_df": DF,
            "quadrature_cases": checks, "maximum_errors": maxima,
            "sign_change_and_optimum": roots, "sign_reversal_examples": examples}


def bootstrap_summary(group, rng):
    values = group.sort_values("path")[["realized_interval_score", "conditional_expected_interval_score"]].to_numpy()
    assert group.path.nunique() == len(values) == 128
    sampled = rng.integers(len(values), size=(4000, len(values)))
    boot = values[sampled].mean(axis=1)
    rows = []
    for j, metric in enumerate(("realized_interval_score", "conditional_expected_interval_score")):
        low, high = np.quantile(boot[:, j], [.025, .975])
        rows.append({"metric": metric, "estimate": values[:, j].mean(), "ci_low": low,
                     "ci_high": high, "path_std_ddof1": values[:, j].std(ddof=1),
                     "scope": "exploratory unadjusted paired paths; single fixed seed 4101"})
    difference = values[:, 0]-values[:, 1]
    low, high = np.quantile(boot[:, 0]-boot[:, 1], [.025, .975])
    rows.append({"metric": "realized_minus_conditional_expected", "estimate": difference.mean(),
                 "ci_low": low, "ci_high": high, "path_std_ddof1": difference.std(ddof=1),
                 "scope": "exploratory unadjusted paired paths; single fixed seed 4101"})
    return rows


def render(checks, summaries):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    zz = np.linspace(0, 4, 600)
    for family, label, color in (("gaussian", "Gaussian", "#186ca8"), ("student_t5", "Unit-variance t5", "#b74b38")):
        root = checks["sign_change_and_optimum"][family]
        ax[0].plot(zz, expected_score(zz, 1., family), label=label, color=color)
        ax[0].axvline(root["optimal_h_over_tau"], color=color, ls=":", alpha=.7)
        ax[1].plot(zz, derivatives(zz, 1., family)[2], label=label, color=color)
        ax[1].scatter([root["mixed_derivative_zero_h_over_tau"]], [0], color=color, s=28, zorder=4)
    ax[0].set(xlabel="Half-width / true scale, h / tau", ylabel="Expected interval score / tau",
              title="Convex in half-width; dots mark optimal width")
    ax[0].set_title("Convex in half-width; dotted lines mark optimum")
    ax[1].set(xlabel="Half-width / true scale, h / tau", ylabel="Mixed derivative with respect to s and q",
              title="Score geometry changes interaction sign")
    ax[1].axhline(0, color="black", linewidth=.7)
    for aa in ax:
        aa.legend(frameon=False)
        aa.grid(alpha=.18)
    fig.suptitle("Interval score at alpha = 0.10: standard probability calculation", fontsize=12)
    fig.savefig(OUT / "expected_score_geometry.png", dpi=180)
    plt.close(fig)
    select = summaries[(summaries.phase.isin(["entry", "recovery64"])) & (summaries.architecture != "gru-minus-tcn")]
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.2), constrained_layout=True)
    categories = [(f, a) for f in ("gaussian", "student_t5") for a in ARCHITECTURES]
    for aa, phase in zip(ax, ("entry", "recovery64")):
        for metric, label, color, offset in (("realized_interval_score", "Realized IS", "#186ca8", -.12),
                                            ("conditional_expected_interval_score", "Conditional expected IS", "#b74b38", .12)):
            g = select[(select.phase == phase) & (select.metric == metric)].set_index(["family", "architecture"]).loc[categories]
            x = np.arange(len(categories))+offset
            aa.errorbar(x, g.estimate, yerr=np.vstack((g.estimate-g.ci_low, g.ci_high-g.estimate)),
                        fmt="o", capsize=3, color=color, label=label)
        aa.set_xticks(np.arange(len(categories)), [f"{f.replace('student_t5', 't5')}\n{a.upper()}" for f, a in categories])
        aa.axhline(0, color="black", lw=.7)
        aa.set(title=phase, ylabel="Event-minus-control context × window interaction")
        aa.grid(axis="y", alpha=.18)
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Exploratory: seed 4101, A3 D128; 95% paired-path intervals", fontsize=12)
    fig.savefig(OUT / "realized_expected_interactions.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check:
        print(json.dumps(checks, indent=2))
        return
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                               check=True, capture_output=True).stdout
    assert hashlib.sha256(committed).hexdigest() == sha(__file__), "commit new diagnostic before reference-data access"
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Existing exploratory diagnostic; refusing silent replacement")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    evaluation = json.loads((EVAL / "manifest.json").read_text())
    assert evaluation["status"] == "complete"
    assert sha(ROOT / "scripts/evaluate_stage2.py") == evaluation["sources"]["evaluate_stage2.py"]
    protocol = json.loads((ROOT / "configs/stage2_protocol.json").read_text())
    assert sha(ROOT / "configs/stage2_protocol.json") == evaluation["protocol_sha256"]
    assert protocol["test"]["alpha"] == ALPHA and protocol["test"]["student_t_df"] == DF
    specs_path = ROOT / "outputs/stage2_strong_supplement/validation_calibration_specs.json"
    specs = json.loads(specs_path.read_text())
    assert specs["model_seed"] == SEED and specs["protocol_sha256"] == evaluation["protocol_sha256"]
    assert specs["training_manifest_sha256"] == evaluation["training_manifest_sha256"]
    manifest = {"status": "running", "design_status": "exploratory; specified after viewing frozen Stage 2 outcomes",
                "new_blind_test": False, "fitting_or_selection": "none", "primary_analysis_replaced": False,
                "source_sha256": sha(__file__), "code_commit_before_execution": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
                "evaluation_manifest_sha256": sha(EVAL / "manifest.json"),
                "original_evaluator_sha256": sha(ROOT / "scripts/evaluate_stage2.py"),
                "validation_specs_sha256": sha(specs_path), "protocol_sha256": evaluation["protocol_sha256"],
                "reference_npz_sha256": {}, "model_seed": SEED, "paths": 128,
                "case": {"families": ["gaussian", "student_t5"], "duration": 128, "amplitude": 3},
                "oracle_use": "only post hoc scoring: integrate contemporaneous innovation under known simulator law",
                "bootstrap": {"repetitions": 4000, "seed": 890311, "level": .95,
                              "scope": "unadjusted descriptive path bootstrap conditional on one paired model seed"},
                "self_check": checks, "standard_interval_score_source":
                    "https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf",
                "source_note": "Gneiting-Raftery 2007 equation 43 / sections 6.2-6.3; derived Gaussian/t5 expectations are elementary calculations, not claimed novel"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    performance, effects, fixed_archive = [], [], []
    max_late = 0.
    for family in ("gaussian", "student_t5"):
        filename = f"reference_{family}.npz"
        ref_path = EVAL / filename
        ref_hash = sha(ref_path)
        assert ref_hash == evaluation["artifacts"][filename]
        manifest["reference_npz_sha256"][filename] = ref_hash
        with np.load(ref_path, allow_pickle=False) as ref:
            worlds = {w: (ref[f"y_{w}"], ref[f"true_sigma_{w}"]) for w in ("event", "control")}
            onset, exit_time = int(ref["onset"]), int(ref["exit"])
            assert onset == 384 and exit_time == 512
            slices = phases(onset, exit_time-onset, 1024)
            for architecture in ARCHITECTURES:
                phase_scores = {"event": {}, "control": {}}
                for world, (y, tau) in worlds.items():
                    assert y.shape == tau.shape == (128, 1024)
                    scales = {c: ref[f"{architecture}_c{c}_{world}"] for c in CONTEXTS}
                    quantiles = {}
                    for c in CONTEXTS:
                        warm = specs["calibration_specs"][f"{architecture}_c{c}"]["warmup_scores"]
                        for w in WINDOWS:
                            hh = rolling_half_width(y, scales[c], w, ALPHA, warm)
                            quantiles[c, w] = hh/scales[c]
                    for a in CONTEXTS:
                        for b in CONTEXTS:
                            for w in WINDOWS:
                                h = scales[a]*quantiles[b, w]
                                observed = 2*h+(2/ALPHA)*np.maximum(np.abs(y)-h, 0)
                                expected = expected_score(h, tau, family)
                                kk = derivatives(h, tau, family)[2]
                                for phase, (lo, hi) in slices.items():
                                    vals = np.column_stack((observed[:, lo:hi].mean(axis=1), expected[:, lo:hi].mean(axis=1)))
                                    phase_scores[world][a, b, w, phase] = vals
                                    if a == b:
                                        for p in range(128):
                                            performance.append({"family": family, "architecture": architecture, "seed": SEED,
                                                "context": a, "window": w, "phase": phase, "world": world, "path": p,
                                                "realized_interval_score": vals[p, 0], "conditional_expected_interval_score": vals[p, 1],
                                                "fraction_negative_mixed_derivative": (kk[p, lo:hi] < 0).mean(),
                                                "mean_h_over_true_scale": (h[p, lo:hi]/tau[p, lo:hi]).mean()})
                for phase in slices:
                    d = {key: phase_scores["event"][key]-phase_scores["control"][key] for key in phase_scores["event"]}
                    interaction = (d[64, 64, 128, phase]-d[16, 16, 128, phase]
                                   -d[64, 64, 32, phase]+d[16, 16, 32, phase])
                    if phase == "late":
                        max_late = max(max_late, float(np.abs(interaction).max()))
                    for p in range(128):
                        effects.append({"family": family, "architecture": architecture, "seed": SEED,
                            "phase": phase, "path": p, "realized_interval_score": interaction[p, 0],
                            "conditional_expected_interval_score": interaction[p, 1]})
                    for b in CONTEXTS:
                        value = (d[64, b, 128, phase]-d[16, b, 128, phase]
                                 -d[64, b, 32, phase]+d[16, b, 32, phase])
                        for p in range(128):
                            fixed_archive.append({"family": family, "architecture": architecture, "seed": SEED,
                                "archive_context": b, "phase": phase, "path": p,
                                "realized_interval_score": value[p, 0], "conditional_expected_interval_score": value[p, 1]})
    effects = pd.DataFrame(effects)
    original_path = EVAL / "paired_interactions.csv.gz"
    assert sha(original_path) == evaluation["artifacts"][original_path.name]
    original = pd.read_csv(original_path)
    original = original[(original.seed == SEED) & (original.duration == 128) & (original.amplitude == 3)
                        & original.family.isin(["gaussian", "student_t5"])
                        & original.architecture.isin(ARCHITECTURES)]
    keys = ["family", "architecture", "seed", "phase", "path"]
    assert not original.duplicated(keys).any() and len(original) == len(effects) == 2048
    joined = effects.merge(original[keys+["interval_score"]], on=keys, validate="one_to_one")
    error = float(np.abs(joined.realized_interval_score-joined.interval_score).max())
    np.testing.assert_allclose(joined.realized_interval_score, joined.interval_score, rtol=0, atol=1e-10)
    np.testing.assert_allclose(max_late, 0, rtol=0, atol=1e-12)
    summary, fixed_summary = [], []
    rng = np.random.default_rng(890311)
    contrasts = []
    for (family, phase), g in effects.groupby(["family", "phase"], sort=True):
        pair = g.pivot(index="path", columns="architecture", values=["realized_interval_score", "conditional_expected_interval_score"])
        for p in range(128):
            contrasts.append({"family": family, "architecture": "gru-minus-tcn", "seed": SEED, "phase": phase, "path": p,
                "realized_interval_score": pair.loc[p, ("realized_interval_score", "gru")]-pair.loc[p, ("realized_interval_score", "tcn")],
                "conditional_expected_interval_score": pair.loc[p, ("conditional_expected_interval_score", "gru")]-pair.loc[p, ("conditional_expected_interval_score", "tcn")]})
    all_effects = pd.concat([effects, pd.DataFrame(contrasts)], ignore_index=True)
    for (family, architecture, phase), group in all_effects.groupby(["family", "architecture", "phase"], sort=True):
        summary.extend({"family": family, "architecture": architecture, "seed": SEED, "phase": phase, **row}
                       for row in bootstrap_summary(group, rng))
    fixed = pd.DataFrame(fixed_archive)
    for (family, architecture, archive, phase), group in fixed.groupby(["family", "architecture", "archive_context", "phase"], sort=True):
        fixed_summary.extend({"family": family, "architecture": architecture, "seed": SEED, "archive_context": archive,
                              "phase": phase, **row} for row in bootstrap_summary(group, rng))
    performance = pd.DataFrame(performance)
    summary = pd.DataFrame(summary)
    compression = {"method": "gzip", "mtime": 0}
    performance.to_csv(OUT / "path_performance.csv.gz", index=False, compression=compression)
    all_effects.to_csv(OUT / "path_interactions.csv.gz", index=False, compression=compression)
    fixed.to_csv(OUT / "fixed_archive_path_interactions.csv.gz", index=False, compression=compression)
    summary.to_csv(OUT / "interaction_summary.csv", index=False)
    pd.DataFrame(fixed_summary).to_csv(OUT / "fixed_archive_summary.csv", index=False)
    means = performance.groupby(["family", "architecture", "seed", "context", "window", "phase", "world"], as_index=False).mean(numeric_only=True).drop(columns="path")
    means.to_csv(OUT / "performance_summary.csv", index=False)
    (OUT / "numerical_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    render(checks, summary)
    manifest.update({"status": "complete", "runtime_seconds": time.time()-started,
        "original_interactions_sha256": sha(original_path), "diagonal_realized_max_absolute_error": error,
        "late_realized_and_expected_max_absolute_interaction": max_late,
        "rows": {"path_performance": len(performance), "path_interactions": len(all_effects),
                 "fixed_archive_path_interactions": len(fixed), "interaction_summary": len(summary)},
        "artifacts": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"},
        "variance_note": "Integration removes contemporaneous innovation at each time; noisy histories remain. No guaranteed variance reduction for correlated time sums, interactions or paired-world contrasts is claimed."})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": "complete", "runtime_seconds": manifest["runtime_seconds"], "maximum_checks": checks["maximum_errors"],
                      "reference_diagonal_error": error, "rows": manifest["rows"]}, indent=2))


if __name__ == "__main__":
    main()
