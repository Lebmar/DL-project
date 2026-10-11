"""Causal diagnostic intervention on current and archived GRU scale contexts.

a != b deliberately combines current scale from context a with historical
forecast-time denominators from context b. It is an artificial diagnostic, not
a new deployment method or a claim of guaranteed coverage. No models retrained.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

import run_gru_simulation as g

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "archive_intervention"
METRICS = ("interval_score", "expected_coverage", "width", "mean_log_halfwidth")


def path_metrics(y, sigma, scale, q, start, stop):
    half = scale[:, start:stop] * q[:, start:stop]
    actual = y[:, start:stop]
    width = 2*half
    return {"interval_score": (width+20*np.maximum(np.abs(actual)-half, 0)).mean(axis=1),
            "expected_coverage": (2*ndtr(half/sigma[:, start:stop])-1).mean(axis=1),
            "width": width.mean(axis=1),
            "mean_log_current_scale": np.log(scale[:, start:stop]).mean(axis=1),
            "mean_log_archived_quantile": np.log(q[:, start:stop]).mean(axis=1),
            "mean_log_halfwidth": np.log(half).mean(axis=1)}


def factorial_effects(p):
    original = p[(64,64,128)]-p[(64,64,32)]-p[(16,16,128)]+p[(16,16,32)]
    current16 = p[(64,16,128)]-p[(64,16,32)]-p[(16,16,128)]+p[(16,16,32)]
    current64 = p[(64,64,128)]-p[(64,64,32)]-p[(16,64,128)]+p[(16,64,32)]
    archived16 = p[(16,64,128)]-p[(16,64,32)]-p[(16,16,128)]+p[(16,16,32)]
    archived64 = p[(64,64,128)]-p[(64,64,32)]-p[(64,16,128)]+p[(64,16,32)]
    pair32 = p[(64,64,32)]-p[(64,16,32)]-p[(16,64,32)]+p[(16,16,32)]
    pair128 = p[(64,64,128)]-p[(64,16,128)]-p[(16,64,128)]+p[(16,16,128)]
    assert np.allclose(original, current64+archived16, atol=1e-10)
    assert np.allclose(original, current16+archived64, atol=1e-10)
    assert np.allclose(current64-current16, pair128-pair32, atol=1e-10)
    return {"original_diagonal": original,
            "current_x_window_archive16": current16,
            "current_x_window_archive64": current64,
            "archive_x_window_current16": archived16,
            "archive_x_window_current64": archived64,
            "current_x_archive_window32": pair32,
            "current_x_archive_window128": pair128,
            "three_way": current64-current16}


def bootstrap_effects(paths, n, repetitions=2000):
    rng = np.random.default_rng(730192)
    by_seed, pooled = [], []
    keys = ["scenario", "phase", "metric", "effect"]
    for (seed, *key), p in paths.groupby(["model_seed", *keys]):
        values = p.sort_values("trajectory").adjusted_effect.to_numpy()
        samples = rng.integers(0, n, (repetitions, n))
        b = values[samples].mean(axis=1)
        by_seed.append(dict(zip(keys, key)) | {"model_seed": seed,
                       "adjusted_effect": values.mean(), "paired_path_se": values.std(ddof=1)/np.sqrt(n),
                       "ci95_low": np.quantile(b,.025), "ci95_high": np.quantile(b,.975)})
    for key, p in paths.groupby(keys):
        grid = p.pivot(index="model_seed", columns="trajectory", values="adjusted_effect").to_numpy()
        b = np.empty(repetitions)
        for i in range(repetitions):
            si = rng.integers(0,grid.shape[0],grid.shape[0])
            ti = rng.integers(0,grid.shape[1],grid.shape[1])
            b[i] = grid[np.ix_(si,ti)].mean()
        sm = grid.mean(axis=1)
        pooled.append(dict(zip(keys,key)) | {"adjusted_effect": grid.mean(),
                      "min_seed_effect": sm.min(), "max_seed_effect": sm.max(),
                      "crossed_bootstrap_ci95_low": np.quantile(b,.025),
                      "crossed_bootstrap_ci95_high": np.quantile(b,.975)})
    return pd.DataFrame(by_seed), pd.DataFrame(pooled)


def make_plots(curve, effects):
    c = curve.groupby(["scenario","context","window","relative_time"])[["scale","q"]].mean().reset_index()
    fig, axes = plt.subplots(2,2,figsize=(11,6),sharex=True,sharey="row")
    colors = ["#2669a5", "#bd6b36", "#778047", "#956b9f"]
    for col,scenario in enumerate(["shock_5","shift_128"]):
        d = c[c.scenario.eq(scenario)]
        for context,color in [(16,colors[0]),(64,colors[2])]:
            z = d[d.context.eq(context)&d.window.eq(32)]
            axes[0,col].plot(z.relative_time,z.scale,color=color,label=f"Current GRU context {context}")
        for color,(context,window) in zip(colors,[(16,32),(16,128),(64,32),(64,128)]):
            z = d[d.context.eq(context)&d.window.eq(window)]
            axes[1,col].plot(z.relative_time,z.q,color=color,label=f"Archive context {context}, window {window}")
        duration = 5 if scenario == "shock_5" else 128
        for ax in axes[:,col]:
            ax.axvspan(0,duration,color="gray",alpha=.12)
            ax.grid(alpha=.15)
        axes[0,col].set_title(f"Scale pulse duration {duration}")
        axes[1,col].set_xlabel("Time relative to onset")
    axes[0,0].set_ylabel("Current predicted scale")
    axes[1,0].set_ylabel("Archived-score quantile q")
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,fontsize=9)
    fig.suptitle("Same frozen GRUs and paths: half-width = current scale × archived q")
    fig.tight_layout(rect=[0,.13,1,.95])
    fig.savefig(OUT/"scale_and_archive_quantile.png",dpi=150)
    plt.close(fig)

    wanted = ["original_diagonal","current_x_window_archive64","archive_x_window_current16"]
    labels = ["Original diagonal", "Current × window (archive64)", "Archive × window (current16)"]
    fig,axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    scenarios = ["shock_1","shock_5","shift_32","shift_128","persistent"]
    for col,phase in enumerate(["onset32","recovery64"]):
        d = effects[effects.metric.eq("interval_score")&effects.phase.eq(phase)]
        x = np.arange(len(scenarios))
        for j,(effect,label) in enumerate(zip(wanted,labels)):
            values = [float(d[d.scenario.eq(s)&d.effect.eq(effect)].adjusted_effect.iloc[0])
                      if len(d[d.scenario.eq(s)&d.effect.eq(effect)]) else np.nan for s in scenarios]
            axes[col].bar(x+(j-1)*.24,values,width=.23,color=colors[j],label=label)
        axes[col].set_xticks(x,scenarios,rotation=30,ha="right")
        axes[col].axhline(0,color="black",linewidth=1)
        axes[col].set_title(phase+"; matched-time no-change subtraction")
        axes[col].grid(axis="y",alpha=.15)
        if phase == "recovery64":
            axes[col].text(4, -.015, "No recovery\nobservations", ha="center", va="top", fontsize=8)
    axes[0].set_ylabel("Adjusted interval-score interaction")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=8)
    fig.suptitle("Artificial causal decomposition; components sum to original diagonal")
    fig.tight_layout(rect=[0,.12,1,.95])
    fig.savefig(OUT/"score_interaction_decomposition.png",dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    started = time.time()
    original_path = g.OUT/"trajectory_metrics.csv"
    if not original_path.exists():
        original_path = g.OUT/"trajectory_metrics.csv.gz"
    original = pd.read_csv(original_path)
    original = original[original.calibrator.eq("rolling")]
    original_manifest = json.loads((g.OUT/"run_manifest.json").read_text())
    n = original_manifest["simulation"]["test_paths"]
    length,onset = 768,256
    assert original_manifest["simulation"]["steps"] == length
    assert original_manifest["simulation"]["onset"] == onset
    noise = np.stack([np.random.default_rng(50000+i).normal(size=length) for i in range(n)])
    durations = original_manifest["simulation"]["durations"]
    metrics, effects, log_parts, curve = [], [], [], []
    reproduction_max = 0.0
    checkpoint_hashes = {}
    for seed in g.TRAIN_SEEDS:
        path = g.OUT/f"gru_seed_{seed}.pt"
        checkpoint_hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = g.ScaleGRU()
        model.load_state_dict(torch.load(path,weights_only=True))
        model.eval()
        base_scale = {a:g.infer_targets(model,noise,a) for a in g.CONTEXTS}
        base_q = {(b,w):g.intervals(noise,base_scale[b],w)/base_scale[b]
                  for b in g.CONTEXTS for w in g.WINDOWS}
        for scenario,duration in durations.items():
            sigma = np.ones_like(noise)
            sigma[:,onset:onset+duration] = 3
            y = noise*sigma
            scales = {a:base_scale[a].copy() for a in g.CONTEXTS}
            if duration:
                for a in g.CONTEXTS:
                    lo,hi = onset+1,min(length,onset+duration+a)
                    scales[a][:,lo:hi] = g.infer_targets(model,y,a,lo,hi)
            qs = {(b,w):g.intervals(y,scales[b],w)/scales[b]
                  for b in g.CONTEXTS for w in g.WINDOWS}
            for a in g.CONTEXTS:
                for b in g.CONTEXTS:
                    for w in g.WINDOWS:
                        half = scales[a]*qs[(b,w)]
                        if a == b:
                            rows,_ = g.path_rows(seed,scenario,a,"rolling",w,y,sigma,half,onset,onset+duration)
                            reproduced = pd.DataFrame(rows).set_index(["phase","trajectory"])
                            ref = original[original.model_seed.eq(seed)&original.scenario.eq(scenario)
                                           &original.context.eq(a)&original.window.eq(w)].set_index(["phase","trajectory"])
                            assert reproduced.index.equals(ref.index)
                            for metric in ["expected_coverage","empirical_coverage","width","interval_score"]:
                                err = float(np.max(np.abs(reproduced[metric]-ref[metric])))
                                reproduction_max = max(reproduction_max,err)
            phases = {"onset32": (onset,onset+32)}
            if onset+duration < length:
                phases["recovery64"] = (onset+duration,min(length,onset+duration+64))
            for phase,(lo,hi) in phases.items():
                cells, controls = {}, {}
                for a in g.CONTEXTS:
                    for b in g.CONTEXTS:
                        for w in g.WINDOWS:
                            key = (a,b,w)
                            cells[key] = path_metrics(y,sigma,scales[a],qs[(b,w)],lo,hi)
                            controls[key] = path_metrics(noise,np.ones_like(noise),base_scale[a],base_q[(b,w)],lo,hi)
                            for path_id in range(n):
                                row = {"model_seed":seed,"scenario":scenario,"phase":phase,
                                       "current_context":a,"archive_context":b,"window":w,
                                       "trajectory":path_id,"n_times":hi-lo}
                                for name,values in cells[key].items():
                                    row[name] = values[path_id]
                                    row[name+"_matched_control"] = controls[key][name][path_id]
                                metrics.append(row)
                for metric in METRICS:
                    event_effect = factorial_effects({k:v[metric] for k,v in cells.items()})
                    control_effect = factorial_effects({k:v[metric] for k,v in controls.items()})
                    if metric == "mean_log_halfwidth":
                        assert max(np.abs(event_effect["current_x_window_archive16"]).max(),
                                   np.abs(event_effect["current_x_window_archive64"]).max()) < 1e-12
                    for effect,values in event_effect.items():
                        for path_id in range(n):
                            effects.append({"model_seed":seed,"scenario":scenario,"phase":phase,
                                            "metric":metric,"effect":effect,"trajectory":path_id,
                                            "raw_effect":values[path_id],"matched_control_effect":control_effect[effect][path_id],
                                            "adjusted_effect":values[path_id]-control_effect[effect][path_id]})
                for w in g.WINDOWS:
                    diff_scale = cells[(64,64,w)]["mean_log_current_scale"]-cells[(16,16,w)]["mean_log_current_scale"]
                    diff_q = cells[(64,64,w)]["mean_log_archived_quantile"]-cells[(16,16,w)]["mean_log_archived_quantile"]
                    diff_half = cells[(64,64,w)]["mean_log_halfwidth"]-cells[(16,16,w)]["mean_log_halfwidth"]
                    assert np.allclose(diff_half,diff_scale+diff_q,atol=1e-12)
                    control_scale = controls[(64,64,w)]["mean_log_current_scale"]-controls[(16,16,w)]["mean_log_current_scale"]
                    control_q = controls[(64,64,w)]["mean_log_archived_quantile"]-controls[(16,16,w)]["mean_log_archived_quantile"]
                    for path_id in range(n):
                        log_parts.append({"model_seed":seed,"scenario":scenario,"phase":phase,"window":w,
                                          "trajectory":path_id,"logscale_difference":diff_scale[path_id],
                                          "logq_difference":diff_q[path_id],"loghalf_difference":diff_half[path_id],
                                          "adjusted_logscale_difference":diff_scale[path_id]-control_scale[path_id],
                                          "adjusted_logq_difference":diff_q[path_id]-control_q[path_id],
                                          "adjusted_loghalf_difference":diff_half[path_id]-control_scale[path_id]-control_q[path_id]})
            for b in g.CONTEXTS:
                for w in g.WINDOWS:
                    for t in range(onset-24,onset+257):
                        curve.append({"model_seed":seed,"scenario":scenario,"context":b,"window":w,
                                      "relative_time":t-onset,"scale":scales[b][:,t].mean(),"q":qs[(b,w)][:,t].mean()})
        print(json.dumps({"completed_frozen_model_seed":seed,"elapsed_seconds":time.time()-started}),flush=True)
    assert reproduction_max < 1e-5
    frame = pd.DataFrame(metrics)
    effect_paths = pd.DataFrame(effects)
    logs = pd.DataFrame(log_parts)
    frame.to_csv(OUT/"factorial_metrics_by_path.csv",index=False)
    effect_paths.to_csv(OUT/"interaction_effects_by_path.csv",index=False)
    logs.to_csv(OUT/"log_decomposition_by_path.csv",index=False)
    groups = ["model_seed","scenario","phase","current_context","archive_context","window"]
    names = ["interval_score","expected_coverage","width","mean_log_current_scale","mean_log_archived_quantile","mean_log_halfwidth"]
    frame.groupby(groups)[names].mean().reset_index().to_csv(OUT/"factorial_summary_by_seed.csv",index=False)
    logs.groupby(["model_seed","scenario","phase","window"]).mean(numeric_only=True).drop(columns="trajectory").reset_index().to_csv(OUT/"log_decomposition_summary_by_seed.csv",index=False)
    seed_summary,pooled = bootstrap_effects(effect_paths,n)
    seed_summary.to_csv(OUT/"effects_by_seed_bootstrap.csv",index=False)
    pooled.to_csv(OUT/"effects_crossed_bootstrap.csv",index=False)
    log_effects = []
    for component in ["adjusted_logscale_difference", "adjusted_logq_difference", "adjusted_loghalf_difference"]:
        z = logs[["model_seed", "scenario", "phase", "window", "trajectory", component]].copy()
        z = z.rename(columns={component:"adjusted_effect"})
        z["metric"] = component
        z["effect"] = "ctx64_minus16_window"+z.window.astype(str)
        log_effects.append(z)
    log_seed,log_pooled = bootstrap_effects(pd.concat(log_effects,ignore_index=True),n)
    log_seed.to_csv(OUT/"log_components_by_seed_bootstrap.csv",index=False)
    log_pooled.to_csv(OUT/"log_components_crossed_bootstrap.csv",index=False)
    make_plots(pd.DataFrame(curve),pooled)
    manifest = {"date":"2026-10-11","python":platform.python_version(),"torch":torch.__version__,
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gru_source_sha256":hashlib.sha256((ROOT/"scripts/run_gru_simulation.py").read_bytes()).hexdigest(),
                "checkpoint_sha256":checkpoint_hashes,"training_performed":False,
                "protocol":"same three frozen GRUs, same 128 paired paths, no tuning or model selection",
                "simulation":{"test_paths":n,"steps":length,"onset":onset,"durations":durations,
                              "test_seed_rule":"50000+i","sigma_before":1,"sigma_event":3},
                "current_contexts":[16,64],"archive_contexts":[16,64],"calibration_windows":[32,128],
                "diagonal_reproduction_max_absolute_error":reproduction_max,
                "checks":{"a_equals_b_reproduces_original":reproduction_max<1e-5,
                          "exact_two_path_score_decomposition":True,
                          "fixed_archive_logwidth_current_x_window_zero":True,
                          "loghalf_equals_logscale_plus_logq":True},
                "matched_control":"no_change at identical absolute target dates for each event and phase",
                "uncertainty":"2000 paired path bootstrap per model; crossed seed/path bootstrap pooled; only three training seeds",
                "limitations":["a != b is artificial causal diagnostic, not a new guaranteed-coverage method",
                               "known Gaussian CDF/sigma are evaluation only",
                               "exploratory effects and CIs without multiplicity adjustment",
                               "same prefixes for durations >=32 yield identical onset32 results",
                               "no claim of universal memory mechanism, novelty, or real-market effectiveness"],
                "wall_seconds":time.time()-started}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    lines = ["# Frozen-GRU archive intervention: exploratory diagnostic", "",
             "No training, tuning, or test-based model selection. Three pre-existing models and the same paired paths.", "",
             "Off-diagonal a != b combines current scale from context a with archived, forecast-time denominators from context b. This is an artificial causal intervention, not a proposed deployment algorithm or guaranteed-coverage method.", "",
             f"Diagonal a=b reproduces original metrics with maximum absolute error {reproduction_max:.3g}. Known Gaussian CDF and sigma are used for evaluation only.", "",
             "Effects below subtract no_change at identical target dates. Units are interval-score units. Confidence intervals resample model seeds and paired path ids separately; only three model seeds, exploratory comparisons without multiplicity adjustment.", "",
             "| Scenario / phase | Effect | Mean | Crossed-bootstrap 95% CI | Seed range |",
             "|---|---|---:|---|---|"]
    for scenario,phase in [("persistent","onset32"),("shift_128","recovery64")]:
        for effect in ["original_diagonal","current_x_window_archive64","archive_x_window_current16"]:
            z = pooled[pooled.scenario.eq(scenario)&pooled.phase.eq(phase)&pooled.metric.eq("interval_score")&pooled.effect.eq(effect)].iloc[0]
            lines.append(f"| {scenario} / {phase} | {effect} | {z.adjusted_effect:.5f} | [{z.crossed_bootstrap_ci95_low:.5f}, {z.crossed_bootstrap_ci95_high:.5f}] | [{z.min_seed_effect:.5f}, {z.max_seed_effect:.5f}] |")
    lines += ["", "For each path, original = current_x_window_archive64 + archive_x_window_current16 exactly. An alternative path gives current_x_window_archive16 + archive_x_window_current64; both are reported.", "",
              "log(halfwidth)=log(current_scale)+log(archived_q). At fixed archive context, current_context × window interaction on log width is zero by construction, not an empirical discovery. Interval score is nonlinear, so its components still depend on current scale.", "",
              "The archive component changes sign across model seeds at persistent onset; the early effect cannot be attributed entirely to archived denominators. The recovery attribution differs, illustrating phase dependence. Neither interpretation establishes novelty, market performance, or an improved deployable method.", "",
              "Reproduce: `.venv/bin/python scripts/run_archive_intervention.py`. The original GRU trajectory CSV may be plain or gzip-compressed. Per-path summaries are saved, not giant repeated per-timestep CSVs."]
    (OUT/"readout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    chosen = pooled[pooled.metric.eq("interval_score")&pooled.scenario.eq("persistent")&pooled.phase.eq("onset32")]
    print(json.dumps({"wall_seconds":manifest["wall_seconds"],"reproduction_error":reproduction_max,
                      "persistent_onset32_components":chosen.to_dict("records")},indent=2))


if __name__ == "__main__":
    main()
