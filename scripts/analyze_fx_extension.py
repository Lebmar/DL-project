"""Audit and summarize the frozen exploratory ECB extension using saved streams.

All fits and currencies remain. Date blocks are jointly resampled after averaging
fixed fitted seeds, preserving currency/method dependence without IID inflation.
No training, new market data, event selection, or choice of a winning method.
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
from scipy.stats import norm

from run_fx_extension import (ROOT, PROTOCOL, OUT as IN, CONTEXTS, WINDOWS, METRICS,
    STREAM_KEYS, INTERVAL_KEYS, COMPONENTS, committed_files, load_protocol, sha,
    expected_streams, expected_intervals, decompose_fx)

OUT = ROOT / "outputs/fx_extension_analysis"
METHOD_KEYS = ["predictor", "context", "calibrator", "window"]
PAIRS = ("gru_minus_garch11", "tcn_minus_garch11", "gru_minus_tcn")
CALIBRATORS = (("native",0), ("rolling",32), ("rolling",128), ("aci",128), ("sfogd",0), ("saocp",0))


def method_name(predictor, context, calibrator, window):
    forecaster = predictor + (str(context) if context else "")
    return forecaster+"_"+calibrator+(str(window) if window else "")


def require_indices(streams, intervals, protocol):
    expected_s = expected_streams(protocol)
    expected_i = expected_intervals(expected_s)
    for frame, expected, id_name, keys in ((streams,expected_s,"stream_id",STREAM_KEYS),
                                          (intervals,expected_i,"interval_id",INTERVAL_KEYS)):
        assert set(frame.columns) == set(expected.columns)
        assert len(frame) == len(expected) and not frame.duplicated(keys).any()
        assert not frame[id_name].duplicated().any()
        pd.testing.assert_frame_equal(frame.sort_values(id_name).reset_index(drop=True)[expected.columns],
                                      expected, check_dtype=False)


def block_weights(length, block, replicates, seed):
    """All outcomes share these exact circular date indices and count weights."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(length, size=(replicates, int(np.ceil(length/block))))
    indices = ((starts[:,:,None]+np.arange(block)) % length).reshape(replicates,-1)[:,:length]
    weights = np.zeros((replicates,length), dtype=np.float64)
    np.add.at(weights, (np.arange(replicates)[:,None],indices), 1/length)
    np.testing.assert_allclose(weights.sum(axis=1),1,rtol=0,atol=1e-13)
    return weights, indices


def independent_metrics(y, half):
    outside = np.maximum(np.abs(y)-half,0)
    return np.stack(((np.abs(y) <= half).astype(float), half+half, half+half+20*outside),axis=-1)


def verify_quantiles(qs, intervals, sigmas, returns, assets, forecast_dates, replay_mask, metadata):
    """Independent scalar timing/rank replay for native, rolling and clipped ACI."""
    original_indices = np.flatnonzero(replay_mask)
    validation = forecast_dates <= np.datetime64("2019-12-31")
    max_error, checked = 0., 0
    assert len(metadata) == len(intervals)
    for row in intervals.itertuples():
        yy = returns[assets.index(row.asset), replay_mask]
        scale = sigmas[row.stream_id,replay_mask]
        normalized = np.abs(yy)/scale
        record = metadata[row.interval_id]
        assert record["interval_id"] == row.interval_id and not record["score_truncation"]
        maximum = float(max(np.max(np.abs(returns[assets.index(row.asset),validation])/sigmas[row.stream_id,validation])*np.sqrt(3),1e-12))
        np.testing.assert_allclose(record["max_scale"],maximum,rtol=0,atol=0)
        q = qs[row.interval_id]
        if row.calibrator == "native":
            expected = np.full(len(q),norm.ppf(.95))
        else:
            assert np.isnan(q[:128]).all()
            assert np.isfinite(q[128:]).all() and (q[128:] >= 0).all()
            if row.calibrator not in ("rolling","aci"):
                assert record["author_source"]["commit"] == "95790a9b9e4061166e6d7a8a13282e019b1a8663"
                continue
            expected = np.full(len(q),np.nan)
            adaptive = .1
            for t in range(row.window,len(q)):
                level = adaptive if row.calibrator == "aci" else .1
                rank = min(row.window,max(1,int(np.ceil((row.window+1)*(1-level)))))
                expected[t] = np.sort(normalized[t-row.window:t])[rank-1]
                if row.calibrator == "aci":
                    missed = normalized[t] > expected[t]
                    adaptive = float(np.clip(adaptive+.02*(.1-missed),1/129,.5))
            expected[:128] = np.nan
        np.testing.assert_allclose(q,expected,rtol=0,atol=1e-12,equal_nan=True)
        valid = np.isfinite(expected)
        max_error = max(max_error,float(np.max(np.abs(q[valid]-expected[valid]))))
        checked += 1
    assert len(original_indices) == qs.shape[1]
    return {"independent_native_rolling_aci_streams": checked,"maximum_absolute_q_error": max_error,
            "strong_source_commit_and_scale_specs_checked": True}


def read_and_audit(protocol, manifest):
    streams = pd.read_csv(IN / "forecast_index.csv")
    intervals = pd.read_csv(IN / "interval_index.csv")
    require_indices(streams,intervals,protocol)
    streams,intervals = streams.sort_values("stream_id"),intervals.sort_values("interval_id")
    with np.load(IN / "issued_forecasts.npz",allow_pickle=False) as saved:
        assert set(saved.files) == {"dates","assets","returns_pct","train_std","sigma","stream_ids"}
        dates = saved["dates"].astype("datetime64[D]")
        assets = saved["assets"].tolist()
        returns, std, sigmas = saved["returns_pct"],saved["train_std"],saved["sigma"]
        np.testing.assert_array_equal(saved["stream_ids"],streams.stream_id.to_numpy())
    assert assets == protocol["data"]["assets"]
    assert len(np.unique(dates)) == len(dates) and np.all(dates[1:] > dates[:-1])
    assert str(dates[-1]) == protocol["data"]["rate_last"] and str(dates[0]) >= "2019-01-01"
    assert returns.shape == (4,len(dates)) and sigmas.shape == (96,len(dates)) and std.shape == (4,)
    assert np.isfinite(returns).all() and np.isfinite(sigmas).all() and (sigmas > 0).all()
    assert np.isfinite(std).all() and (std > 0).all()
    np.testing.assert_array_equal(std,[manifest["data"]["train_std_ddof1"][a] for a in assets])
    with np.load(IN / "calibration_streams.npz",allow_pickle=False) as saved:
        assert set(saved.files) == {"dates","q","interval_ids"}
        replay_dates,qs = saved["dates"].astype("datetime64[D]"),saved["q"]
        np.testing.assert_array_equal(saved["interval_ids"],intervals.interval_id.to_numpy())
    replay_mask = dates >= np.datetime64(protocol["data"]["calibration_replay_start"])
    np.testing.assert_array_equal(replay_dates,dates[replay_mask])
    assert qs.shape == (496,len(replay_dates))
    eval_mask = replay_dates >= np.datetime64(protocol["data"]["evaluation_start"])
    evaluation_dates = replay_dates[eval_mask]
    assert len(evaluation_dates) == 1479 and str(evaluation_dates[0]) == "2021-01-04"
    assert np.isfinite(qs[:,eval_mask]).all() and (qs[:,eval_mask] >= 0).all()
    metadata = json.loads((IN / "calibration_metadata.json").read_text())
    checks = verify_quantiles(qs,intervals,sigmas,returns,assets,dates,replay_mask,metadata)
    per_interval = np.empty((len(intervals),len(evaluation_dates),3))
    for row in intervals.itertuples():
        yy = returns[assets.index(row.asset),replay_mask][eval_mask]
        half = sigmas[row.stream_id,replay_mask][eval_mask]*qs[row.interval_id,eval_mask]
        per_interval[row.interval_id] = independent_metrics(yy,half)
    with np.load(IN / "crossed_metrics.npz",allow_pickle=False) as saved:
        assert set(saved.files) == {"values","dates","model_architectures","model_seeds","assets",
                                   "current_contexts","archive_contexts","windows","metrics"}
        cross = saved["values"]
        assert cross.shape == (10,4,2,2,2,1479,3)
        assert np.isfinite(cross).all() and (cross >= 0).all()
        np.testing.assert_array_equal(saved["dates"],evaluation_dates.astype(str))
        for key,value in (("assets",assets),("current_contexts",CONTEXTS),("archive_contexts",CONTEXTS),
                          ("windows",WINDOWS),("metrics",METRICS)):
            assert saved[key].tolist() == list(value)
        keys = list(zip(saved["model_architectures"].tolist(),saved["model_seeds"].tolist()))
        assert keys == list(product(protocol["training"]["architectures"],protocol["training"]["seeds"]))
    index = intervals.set_index(INTERVAL_KEYS)
    maximum,diagonal_error = 0.,0.
    for mi,(architecture,seed) in enumerate(keys):
        for ai,asset in enumerate(assets):
            yy = returns[ai,replay_mask][eval_mask]
            for ci,a in enumerate(CONTEXTS):
                sid = int(index.loc[(architecture,seed,a,asset,"native",0),"stream_id"])
                scale = sigmas[sid,replay_mask][eval_mask]
                for bi,b in enumerate(CONTEXTS):
                    for wi,window in enumerate(WINDOWS):
                        iid = int(index.loc[(architecture,seed,b,asset,"rolling",window),"interval_id"])
                        rebuilt = independent_metrics(yy,scale*qs[iid,eval_mask])
                        np.testing.assert_allclose(cross[mi,ai,ci,bi,wi],rebuilt,rtol=0,atol=1e-10)
                        maximum = max(maximum,float(np.max(np.abs(cross[mi,ai,ci,bi,wi]-rebuilt))))
                        if a == b:
                            np.testing.assert_allclose(rebuilt,per_interval[iid],rtol=0,atol=0)
                            diagonal_error = max(diagonal_error,float(np.max(np.abs(rebuilt-per_interval[iid]))))
    checks.update(crossed_all_cell_metric_reconstruction_error=maximum,
                  diagonal_performance_reconstruction_error=diagonal_error, exact_indices=True)
    return intervals,per_interval,cross,keys,evaluation_dates,std,checks


def aggregate_methods(intervals,per_interval,p):
    """First average five fitted seeds within asset/method; never mix seed counts."""
    method_index = intervals[METHOD_KEYS].drop_duplicates().sort_values(METHOD_KEYS).reset_index(drop=True)
    method_index["method"] = [method_name(*row) for row in method_index[METHOD_KEYS].itertuples(index=False,name=None)]
    assets = p["data"]["assets"]
    values = np.empty((len(method_index),len(assets),per_interval.shape[1],3))
    for mi,row in enumerate(method_index.itertuples()):
        for ai,asset in enumerate(assets):
            selected = intervals[(intervals.predictor == row.predictor) & (intervals.context == row.context) &
                (intervals.calibrator == row.calibrator) & (intervals.window == row.window) & (intervals.asset == asset)]
            expected_seeds = p["training"]["seeds"] if row.predictor in ("gru","tcn") else [-1]
            assert sorted(selected.seed.tolist()) == expected_seeds
            values[mi,ai] = per_interval[selected.interval_id.to_numpy()].mean(axis=0)
    assert len(method_index) == 44
    return method_index,values


def interval_row(values,mean,level):
    low,high = np.quantile(values,[(1-level)/2,1-(1-level)/2])
    return {"estimate":float(mean),"ci_low":float(low),"ci_high":float(high)}


def analyze(intervals,per_interval,cross,model_keys,dates,std,p):
    spec = p["analysis"]
    method_index,method_raw = aggregate_methods(intervals,per_interval,p)
    method_normalized = method_raw.copy()
    method_normalized[:,:,:,1:] /= std[None,:,None,None]
    method_dates = method_normalized.mean(axis=1).transpose(1,0,2)
    methods = method_index.method.tolist()
    cross_normalized = cross.copy()
    cross_normalized[...,1:] /= std[None,:,None,None,None,None,None]
    component_raw = decompose_fx(cross)
    component_normalized = decompose_fx(cross_normalized)
    # Ten models = two architectures, each with five paired seeds; dates remain paired.
    component_dates = component_normalized.reshape(2,5,4,6,len(dates),3).mean(axis=(1,2))
    component_dates = np.concatenate((component_dates,(component_dates[0]-component_dates[1])[None]),axis=0)
    component_dates = component_dates.transpose(2,0,1,3)
    method_rows,pair_rows,component_rows = [],[],[]
    for block in spec["block_lengths"]:
        weights,_ = block_weights(len(dates),block,spec["bootstrap_replicates"],spec["bootstrap_seed"])
        draws = (weights @ method_dates.reshape(len(dates),-1)).reshape(len(weights),len(methods),3)
        component_draws = (weights @ component_dates.reshape(len(dates),-1)).reshape(len(weights),3,6,3)
        for mi,row in method_index.iterrows():
            for metric_index,metric in enumerate(METRICS):
                method_rows.append({**row.to_dict(),"metric":metric,"block_length":block,
                    "interval_level":spec["interval_level"],"dates":len(dates),
                    **interval_row(draws[:,mi,metric_index],method_dates[:,mi,metric_index].mean(),spec["interval_level"])})
        for calibrator,window in CALIBRATORS:
            names = {a:method_name(a,64 if a in ("gru","tcn") else 0,calibrator,window)
                     for a in ("gru","tcn","garch11")}
            for left,right in (("gru","garch11"),("tcn","garch11"),("gru","tcn")):
                a,b = methods.index(names[left]),methods.index(names[right])
                for j,metric in enumerate(METRICS):
                    pair_rows.append({"comparison":left+"_minus_"+right,"calibrator":calibrator,"window":window,
                        "left_method":names[left],"right_method":names[right],"metric":metric,"block_length":block,
                        "interval_level":spec["interval_level"],
                        **interval_row(draws[:,a,j]-draws[:,b,j],
                            (method_dates[:,a,j]-method_dates[:,b,j]).mean(),spec["interval_level"])})
        for ai,architecture in enumerate(("gru","tcn","gru_minus_tcn")):
            for ci,component in enumerate(COMPONENTS):
                for j,metric in enumerate(METRICS):
                    component_rows.append({"architecture":architecture,"component":component,"metric":metric,
                        "block_length":block,"interval_level":spec["interval_level"],
                        "definition":"J_FX actual historical sequence; no matched event/control subtraction",
                        **interval_row(component_draws[:,ai,ci,j],component_dates[:,ai,ci,j].mean(),spec["interval_level"])})
    # Raw per-asset and per-fit tables retain scientific units alongside normalized values.
    individual = intervals.copy()
    means = per_interval.mean(axis=1)
    individual["coverage"],individual["width"],individual["interval_score"] = means.T
    individual["normalized_width"] = means[:,1]/individual.asset.map(dict(zip(p["data"]["assets"],std))).to_numpy()
    individual["normalized_interval_score"] = means[:,2]/individual.asset.map(dict(zip(p["data"]["assets"],std))).to_numpy()
    per_asset = []
    years = pd.DatetimeIndex(dates).year.to_numpy()
    calendar = []
    for mi,row in method_index.iterrows():
        for ai,asset in enumerate(p["data"]["assets"]):
            raw = method_raw[mi,ai].mean(axis=0)
            normalized = method_normalized[mi,ai].mean(axis=0)
            per_asset.append({**row.to_dict(),"asset":asset,"fixed_fits":5 if row.predictor in ("gru","tcn") else 1,
                "coverage":raw[0],"width":raw[1],"interval_score":raw[2],"normalized_width":normalized[1],
                "normalized_interval_score":normalized[2],"dates":len(dates)})
        for year in sorted(set(years)):
            mask = years == year
            label = f"{year} (partial, through {dates[-1]})" if year == 2026 else str(year)
            for ai,asset in enumerate(p["data"]["assets"]):
                raw = method_raw[mi,ai,mask].mean(axis=0)
                normalized = method_normalized[mi,ai,mask].mean(axis=0)
                calendar.append({**row.to_dict(),"asset":asset,"year":year,"period_label":label,"dates":int(mask.sum()),
                    "coverage":raw[0],"width":raw[1],"interval_score":raw[2],
                    "normalized_width":normalized[1],"normalized_interval_score":normalized[2]})
            normalized = method_normalized[mi][:,mask,:].mean(axis=(0,1))
            calendar.append({**row.to_dict(),"asset":"equal_asset_pool","year":year,"period_label":label,
                "dates":int(mask.sum()),"coverage":normalized[0],"width":np.nan,"interval_score":np.nan,
                "normalized_width":normalized[1],"normalized_interval_score":normalized[2]})
    seed_components = []
    for mi,(architecture,seed) in enumerate(model_keys):
        for ai,asset in enumerate(p["data"]["assets"]):
            raw = component_raw[mi,ai].mean(axis=1)
            normalized = component_normalized[mi,ai].mean(axis=1)
            for ci,component in enumerate(COMPONENTS):
                seed_components.append({"architecture":architecture,"seed":seed,"asset":asset,"component":component,
                    "coverage":raw[ci,0],"width":raw[ci,1],"interval_score":raw[ci,2],
                    "normalized_width":normalized[ci,1],"normalized_interval_score":normalized[ci,2]})
    tables = {"method_summary":pd.DataFrame(method_rows),"paired_method_comparisons":pd.DataFrame(pair_rows),
        "mechanism_summary":pd.DataFrame(component_rows),"per_asset_summary":pd.DataFrame(per_asset),
        "individual_fit_asset_summary":individual,"calendar_year_summary":pd.DataFrame(calendar),
        "mechanism_individual_fit_asset_summary":pd.DataFrame(seed_components),"method_index":method_index}
    # Display is fixed in the protocol before any new outcome, independent of ranking.
    main = tables["method_summary"][
        tables["method_summary"].block_length == spec["main_block_length"]]
    core_rows,matrix_rows = [],[]
    for predictor in ("gru","tcn","garch11"):
        context = 64 if predictor in ("gru","tcn") else 0
        for calibrator,window in (("native",0),("rolling",128)):
            name = method_name(predictor,context,calibrator,window)
            lookup = main[main.method == name].set_index("metric")
            row = dict(method=name,predictor=predictor,context=context,calibrator=calibrator,window=window,
                       block_length=spec["main_block_length"],dates=len(dates))
            for metric,label in (("coverage","coverage"),("interval_score","normalized_interval_score"),
                                 ("width","normalized_width")):
                for source,suffix in (("estimate",""),("ci_low","_ci_low"),("ci_high","_ci_high")):
                    row[label+suffix] = float(lookup.loc[metric,source])
            core_rows.append(row)
    for predictor in ("constant","ewma16","ewma64","garch11","gru","tcn"):
        context = 64 if predictor in ("gru","tcn") else 0
        row = dict(forecaster=predictor+(str(context) if context else ""))
        for calibrator,window in CALIBRATORS:
            name = method_name(predictor,context,calibrator,window)
            value = main[(main.method == name)&(main.metric == "interval_score")].iloc[0]
            row[calibrator+(str(window) if window else "")] = value.estimate
        matrix_rows.append(row)
    tables["report_core_table"] = pd.DataFrame(core_rows)
    tables["report_calibration_matrix"] = pd.DataFrame(matrix_rows)
    return tables,method_raw,method_normalized


def figures(tables,p):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"savefig.dpi":160})
    block = p["analysis"]["main_block_length"]
    summary = tables["method_summary"].query("block_length == @block")
    fig,axes = plt.subplots(1,2,figsize=(12,6.5))
    colors = {"gru":"#2b6995","tcn":"#ae642b","garch11":"#477959"}
    labels = ["Native Gaussian","Rolling 32","Rolling 128","Clipped ACI 128","SF-OGD","SAOCP"]
    for ax,metric,title in zip(axes,("coverage","interval_score"),
                               ("Observed coverage | target 90%","Normalized interval score | lower is better")):
        for pi,predictor in enumerate(("gru","tcn","garch11")):
            for ci,(calibrator,window) in enumerate(CALIBRATORS):
                name = method_name(predictor,64 if predictor != "garch11" else 0,calibrator,window)
                row = summary[(summary.method == name)&(summary.metric == metric)].iloc[0]
                yy = ci+(pi-1)*.2
                ax.plot([row.ci_low,row.ci_high],[yy,yy],color=colors[predictor],lw=1.5)
                ax.scatter(row.estimate,yy,s=24,color=colors[predictor],
                           label={"gru":"GRU context64","tcn":"TCN context64","garch11":"GARCH(1,1)"}[predictor] if ci == 0 else None)
        ax.set_yticks(range(6),labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x",alpha=.2)
        if metric == "coverage":
            ax.axvline(.9,color="#777777",ls=":",lw=1)
    axes[0].legend(frameon=False,fontsize=8,loc="best")
    fig.suptitle("ECB FX extension | fixed comparisons across the same calibrators")
    fig.text(.02,.02,"2021-01-04 to 2026-10-09 | four related EUR crosses | all five fits per neural architecture.\n"
             "Exploratory 95% common-date block intervals, block128; fixed fits, unadjusted comparisons. No blind test or coverage guarantee.",fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.94))
    fig.savefig(OUT / "fx_method_comparisons.png")
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,5.6))
    labels = ["Total J_FX","Current | archive64","Archive | current16","Current | archive16","Archive | current64","Third order"]
    for ax,architecture in zip(axes,("gru","tcn")):
        rows = tables["mechanism_summary"].query("block_length == @block and architecture == @architecture and metric == 'interval_score'")
        rows = rows.set_index("component").loc[list(COMPONENTS)]
        for ci,(_,row) in enumerate(rows.iterrows()):
            ax.plot([row.ci_low,row.ci_high],[ci,ci],color=colors[architecture],lw=2)
            ax.scatter(row.estimate,ci,color=colors[architecture],s=30)
        ax.axvline(0,color="#777777",ls=":",lw=1)
        ax.set_yticks(range(6),labels)
        ax.invert_yaxis()
        ax.set_title(architecture.upper()+" | all five fixed fits")
        ax.set_xlabel("Normalized interval-score contrast")
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Current and archived scale pathways on the actual FX sequence")
    fig.text(.02,.02,"J_FX has no event-minus-control subtraction and differs from simulated shock J. Both decomposition orders retained.\n"
             "Exploratory 95% date-block intervals, block128. Correlated component intervals do not add; no causal mediation percentages.",fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94))
    fig.savefig(OUT / "fx_pathway_components.png")
    plt.close(fig)


def summary_markdown(tables,p,checks):
    block = p["analysis"]["main_block_length"]
    rows = tables["paired_method_comparisons"].query("block_length == @block and metric == 'interval_score'")
    text = ["# Exploratory ECB FX extension","",
        "This extends the previously inspected 2021–2026 history with all ten local CPU neural fits, four public EUR-cross currencies, classical scale models, and fixed calibration rules. It is not a new blind test. No paid HPC training was used.","",
        "The model fits use returns through2018, checkpoint selection and calibration scale heuristics use2019, and each asset-specific calibrator replays2020 before continuing into2021. The historical archive is not point-in-time revision data. All model choices were frozen before this run.","",
        "| Same-calibrator comparison | Calibrator | Normalized IS difference | Exploratory95% interval |",
        "|---|---|---:|---|"]
    for row in rows.itertuples():
        label = row.calibrator+(str(row.window) if row.window else "")
        text.append(f"| {row.comparison} | {label} | {row.estimate:+.6f} | [{row.ci_low:+.6f}, {row.ci_high:+.6f}] |")
    text += ["","Negative differences mean lower interval score for the first named predictor; coverage and width must be read alongside this score. Every method and asset remains in the accompanying tables; the figure uses predeclared comparisons, not a selected winning subset.","",
        "Date-block intervals use2,000 common circular resamples with block128 as the main display and20/60/256 as sensitivities. Five fixed neural fits are averaged within each asset before equal asset averaging. Neither seeds nor currencies are counted as independent financial histories. All intervals are exploratory, unadjusted and conditional on these fitted models; arbitrary nonstationarity is not covered.","",
        "J_FX is the actual-history contrast [D(64,64)−D(16,16)], where D(a,b) compares windows128 and32. It does not subtract a matched no-shock world, so it differs from simulated J. Both current/archive decomposition orders and the third-order term are retained, along with each asset and seed. Its sign is not an absolute model ranking or unique causal contribution.","",
        "The Gaussian interval model and heuristic online scale bounds provide no coverage guarantee. Scores are never truncated. Reference rates are informational and are not executable trading prices or realized volatility. Calendar-year tables identify2026 as partial through2026-10-09.","",
        "Audit: "+json.dumps(checks,ensure_ascii=False),"",
        "Outputs: all-method summary, paired same-calibrator comparisons, raw and normalized per-asset/per-fit summaries, calendar-year results, J_FX mechanism tables, two figures, and daily fixed-fit method metrics. Original issued forecasts, quantile streams, full factorial daily metrics and checkpoints are retained in ../fx_extension/."]
    return "\n".join(text)+"\n"


def tiny_self_check():
    rng = np.random.default_rng(674125301)
    values = rng.normal(size=(23,7))
    errors = []
    for block in (1,4,29):
        weights,indices = block_weights(23,block,41,2026101802)
        expected = np.array([values[index].mean(axis=0) for index in indices])
        actual = weights @ values
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14)
        errors.append(float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose((weights @ (values[:,0]-values[:,1])),actual[:,0]-actual[:,1],rtol=0,atol=1e-14)
    fake = {"training":{"architectures":["gru","tcn"],"seeds":list(range(4301,4306))},
            "data":{"assets":["USD","JPY","GBP","CHF"]},
            "classical":{"models":["constant","ewma16","ewma64","garch11"]}}
    streams = expected_streams(fake)
    intervals = expected_intervals(streams)
    require_indices(streams.sample(frac=1,random_state=1),intervals.sample(frac=1,random_state=2),fake)
    rejected = 0
    bad = intervals.copy()
    bad.loc[0,"asset"] = "OFFGRID"
    for item in (intervals.iloc[:-1],pd.concat((intervals,intervals.iloc[:1])),bad):
        try:
            require_indices(streams,item,fake)
        except AssertionError:
            rejected += 1
    assert rejected == 3
    per_interval = np.empty((496,13,3))
    for row in intervals.itertuples():
        # Deliberately different seed counts for neural/classical streams.
        value = (row.seed-4300 if row.seed > 0 else 7)+100*fake["data"]["assets"].index(row.asset)
        per_interval[row.interval_id] = value
    index,means = aggregate_methods(intervals,per_interval,fake)
    for mi,row in index.iterrows():
        for ai in range(4):
            np.testing.assert_array_equal(means[mi,ai],np.full((13,3),(3 if row.predictor in ("gru","tcn") else 7)+100*ai))
    toy = rng.normal(size=(2,4,2,2,2,13,3))
    parts = decompose_fx(toy)
    for m,a,t,j in product(range(2),range(4),range(13),range(3)):
        d = {(c,b):toy[m,a,c,b,1,t,j]-toy[m,a,c,b,0,t,j] for c,b in product(range(2),range(2))}
        expected = [d[1,1]-d[0,0],d[1,1]-d[0,1],d[0,1]-d[0,0],
                    d[1,0]-d[0,0],d[1,1]-d[1,0],d[1,1]-d[0,1]-d[1,0]+d[0,0]]
        np.testing.assert_allclose(parts[m,a,:,t,j],expected,rtol=0,atol=1e-14)
    return {"passed":True,"synthetic_only":True,"scalar_block_bootstrap_max_error":max(errors),
            "shared_date_pairing":True,"missing_duplicate_offgrid_rejected":True,
            "seed_before_asset_averaging":True,"independent_scalar_fx_decomposition":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check(),indent=2))
        return
    sources = committed_files()
    p = load_protocol()
    assert p["execution"]["analysis_output"] == str(OUT.relative_to(ROOT))
    assert p["analysis"]["block_lengths"] == [128,20,60,256]
    assert p["analysis"]["main_block_length"] == 128 and p["analysis"]["interval_level"] == .95
    assert p["analysis"]["bootstrap_replicates"] == 2000 and p["analysis"]["bootstrap_seed"] == 2026101802
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError("Refusing any existing FX analysis directory")
    manifest_path = IN / "manifest.json"
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and manifest["models_completed"] == 10
    assert manifest["protocol_sha256"] == sha(PROTOCOL) and manifest["new_sources"] == sources
    assert manifest["matching_minibatch_prefixes_verified"] and not manifest["test_outcomes_used_for_fitting_or_selection"]
    assert manifest["training_device"] == manifest["prediction_device"] == "cpu" and not manifest["paid_HPC_used"]
    for relative,value in manifest["artifacts"].items():
        assert sha(IN / relative) == value,relative
    started = time.time()
    intervals,values,cross,keys,dates,std,checks = read_and_audit(p,manifest)
    tables,raw,normalized = analyze(intervals,values,cross,keys,dates,std,p)
    OUT.mkdir(parents=True)
    for name,frame in tables.items():
        frame.to_csv(OUT / (name+".csv"),index=False)
    np.savez_compressed(OUT / "daily_method_metrics.npz",dates=dates.astype(str),
        methods=tables["method_index"].method.to_numpy(dtype=str),assets=np.array(p["data"]["assets"]),
        metrics=np.array(METRICS),raw=raw,normalized=normalized,
        note=np.array("Five fixed neural fits averaged within asset; assets retained separately"))
    figures(tables,p)
    (OUT / "summary.md").write_text(summary_markdown(tables,p,checks))
    assert load_protocol() == p and committed_files() == sources and sha(manifest_path) == manifest_hash
    for relative,value in manifest["artifacts"].items():
        assert sha(IN / relative) == value,relative
    result = {"status":"complete","design_status":p["design_status"],"new_blind_test":False,
        "protocol_sha256":sha(PROTOCOL),"new_sources":sources,"evaluation_manifest_sha256":manifest_hash,
        "self_check":tiny_self_check(),"checks":checks,"analysis":p["analysis"],
        "rows":{name:len(frame) for name,frame in tables.items()},"elapsed_seconds":time.time()-started,
        "artifacts":{path.name:sha(path) for path in sorted(OUT.iterdir()) if path.is_file()}}
    (OUT / "manifest.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":"complete","elapsed_seconds":result["elapsed_seconds"],"checks":checks}),flush=True)


if __name__ == "__main__":
    main()
