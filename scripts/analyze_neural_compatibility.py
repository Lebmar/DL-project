"""Analyze a frozen neural mechanism bridge; no fitting or new test generation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import finite_mechanism_math as fm

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"configs/neural_compatibility_protocol.json"
IN=ROOT/"outputs/neural_compatibility_evaluation"
OUT=ROOT/"outputs/neural_compatibility_analysis"
PRIMARY_METHODS=["P","A","D","D_first32","D_last32"]
MOMENTS=["xi_mean","xi_second_moment","eta_mean","eta_second_moment","xiP_eta_mean","xiP_mean","xiP_second_moment"]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss_readout(cell_means):
    """Cells [group,5] or [group,bootstrap,5]; average absolute errors after path means."""
    population=np.abs(cell_means[...,0]-cell_means[...,1]).mean(axis=0)
    redraw=np.abs(cell_means[...,2]-cell_means[...,1]).mean(axis=0)
    return np.stack((population,redraw,population-redraw),axis=-1)


def path_bootstrap(values, replicates, seed):
    rng=np.random.default_rng(seed)
    result=np.empty((replicates,3))
    for start in range(0,replicates,128):
        count=min(128,replicates-start)
        indices=rng.integers(values.shape[1],size=(count,values.shape[1]))
        cell_means=values[:,indices,:].mean(axis=2)
        result[start:start+count]=loss_readout(cell_means)
    return result


def replica_bootstrap(redraw_replicas, fixed_means, replicates, seed):
    rng=np.random.default_rng(seed)
    result=np.empty((replicates,3))
    pmae=np.abs(fixed_means[:,0]-fixed_means[:,1]).mean()
    for start in range(0,replicates,128):
        count=min(128,replicates-start)
        indices=rng.integers(redraw_replicas.shape[1],size=(count,redraw_replicas.shape[1]))
        dstar=redraw_replicas[:,indices].mean(axis=2)
        dmae=np.abs(dstar-fixed_means[:,1,None]).mean(axis=0)
        result[start:start+count]=np.column_stack((np.full(count,pmae),dmae,pmae-dmae))
    return result


def self_check():
    a=np.array([[[1.,2.,2.,2.,2.],[3.,2.,2.,2.,2.]],[[4.,3.,4.,4.,4.],[2.,3.,2.,2.,2.]]])
    # Averaging before abs is materially different from averaging pathwise abs.
    np.testing.assert_array_equal(loss_readout(a.mean(axis=1)),[0.,0.,0.])
    assert np.abs(a[:,:,0]-a[:,:,1]).mean()==1.
    draws=path_bootstrap(a,11,123)
    rng=np.random.default_rng(123)
    indices=rng.integers(2,size=(11,2))
    independent=np.stack([loss_readout(np.stack([a[g,ii].mean(axis=0) for g in range(2)])) for ii in indices])
    np.testing.assert_allclose(draws,independent,rtol=0,atol=1e-14)
    return {"absolute_error_after_path_mean":True,"independent_shared_path_bootstrap":True}


def run():
    checks=self_check()
    assert not OUT.exists(),"Refusing to overwrite results"
    p=json.loads(PROTOCOL.read_text());e=p["evaluation"];a=p["analysis"]
    for f in [Path(__file__).resolve(),PROTOCOL]:
        original=subprocess.check_output(["git","show",f"HEAD:{f.relative_to(ROOT)}"],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==sha(f)
    manifest=json.loads((IN/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    assert manifest["protocol_sha256"]==sha(PROTOCOL)
    assert manifest["analysis_source_sha256"]==sha(__file__)
    assert manifest["source_sha256"]==sha(ROOT/"scripts/evaluate_neural_compatibility.py")
    assert manifest["training_manifest_sha256"]==sha(ROOT/"outputs/neural_compatibility_training/manifest.json")
    for mapping in [manifest["training_artifact_sha256"],manifest["evaluation_dependencies"]]:
        for name,digest in mapping.items():
            assert sha(ROOT/name)==digest,name
    for name,digest in manifest["sources"].items():
        assert sha(ROOT/"scripts"/name)==digest,name
    for name,digest in manifest["outputs"].items():
        artifact=(IN/name).resolve()
        assert artifact.is_relative_to(IN) and sha(artifact)==digest,name
    index=pd.read_csv(IN/"index.csv")
    keys=["panel","architecture","seed","world","context","window"]
    expected={(panel["panel"],arch,seed,world["name"],context,window)
        for panel in p["panels"] for arch in panel["training_config"]["architectures"]
        for seed in panel["training_config"]["model_seeds"] for world in e["worlds"]
        for context in e["contexts"] for window in e["windows"]}
    assert len(index)==len(expected)==192 and not index.duplicated(keys).any()
    assert set(index[keys].itertuples(index=False,name=None))==expected
    assert not index.file.duplicated().any()
    method_names=e["methods"]; metric_names=e["metrics"];phase_names=list(e["phases"])
    cell_rows,moment_rows,approx_rows,primary_rows,primary_values,replica_values=[],[],[],[],[],[]
    sources={str((IN/"manifest.json").relative_to(ROOT)):sha(IN/"manifest.json"),
        str((IN/"index.csv").relative_to(ROOT)):sha(IN/"index.csv"),str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    max_moment_error=0.;max_raw_error=0.;max_replica_error=0.
    for row in index.itertuples(index=False):
        metadata={key:getattr(row,key) for key in keys}
        world=next(x for x in e["worlds"] if x["name"]==row.world)
        family=world["distribution"]
        file=(IN/row.file).resolve();assert file.is_relative_to(IN)
        assert sha(file)==row.sha256
        sources[str(file.relative_to(ROOT))]=sha(file)
        with np.load(file,allow_pickle=False) as saved:
            metrics=saved["metrics"]
            assert metrics.shape==(7,e["paths"],len(e["times"]),5)
            assert saved["methods"].tolist()==method_names
            assert saved["metric_names"].tolist()==metric_names
            assert saved["times"].tolist()==e["times"]
            replicas=saved["D_replicas_phase"]
            assert replicas.shape==(e["redraw_replicas"],4,5)
            assert saved["phase_names"].tolist()==phase_names
            moments=saved["moments"]
            assert moments.shape==(3,e["paths"],len(e["times"]),7)
            assert saved["moment_methods"].tolist()==["P","A","D"]
            assert saved["moment_names"].tolist()==MOMENTS
            first,second=saved["first_order_log_error"],saved["second_order_log_error"]
            delta=saved["delta"]
        assert np.isfinite(metrics).all() and np.isfinite(moments).all() and np.isfinite(replicas).all()
        truth=np.ones(len(e["times"]))
        selected=(np.asarray(e["times"])>=e["onset"])&(np.asarray(e["times"])<e["onset"]+e["duration"])
        truth[selected]=world["amplitude"]
        for raw,std in [(3,1),(4,2)]:
            error=float(np.max(np.abs(metrics[...,raw]-metrics[...,std]*truth)))
            max_raw_error=max(max_raw_error,error);assert error<1e-9
        oracle=float(fm.expected_interval_score(fm.absolute_ppf(.9,family),family))
        assert np.min(metrics[...,2])>=oracle-1e-9
        moment_error=float(np.max(np.abs(moments[...,1]-moments[...,6]-moments[...,3]-2*moments[...,4])))
        max_moment_error=max(max_moment_error,moment_error);assert moment_error<1e-9
        for phase_number,(phase,time_indices) in enumerate(e["phases"].items()):
            phase_metrics=metrics[:,:,time_indices,:].mean(axis=2)
            for mi,method in enumerate(method_names):
                cell_rows.append({**metadata,"phase":phase,"method":method,
                    **dict(zip(metric_names,phase_metrics[mi].mean(axis=0)))})
            for name,r in [("D",replicas[:,phase_number].mean(axis=0)),("D_first32",replicas[:32,phase_number].mean(axis=0)),
                           ("D_last32",replicas[32:,phase_number].mean(axis=0))]:
                diff=float(np.max(np.abs(phase_metrics[method_names.index(name)].mean(axis=0)-r)))
                max_replica_error=max(max_replica_error,diff);assert diff<1e-9
            for mi,method in enumerate(["P","A","D"]):
                mean=moments[mi][:,time_indices,:].mean(axis=(0,1))
                covariance=mean[4]-mean[5]*mean[2]
                moment_rows.append({**metadata,"phase":phase,"method":method,**dict(zip(MOMENTS,mean)),
                    "cross_covariance":covariance,"cross_product_of_means":mean[5]*mean[2],
                    "identity_residual":mean[1]-mean[6]-mean[3]-2*mean[4]})
            exact_p=moments[0][:,time_indices,0]
            coefficient=float(fm.absolute_ppf(.9,family)**2*fm.absolute_pdf(fm.absolute_ppf(.9,family),family)/.1)
            approx_rows.append({**metadata,"phase":phase,
                "first_order_log_mae":float(np.abs(first[:,time_indices]-exact_p).mean()),
                "second_order_log_mae":float(np.abs(second[:,time_indices]-exact_p).mean()),
                "mean_absolute_delta":float(np.abs(delta[:,time_indices]).mean()),
                **{f"{method}_quadratic_excess_risk":float(coefficient*moments[mi][:,time_indices,1].mean()) for mi,method in enumerate(["P","A","D"])},
                **{f"{method}_exact_excess_risk":float(phase_metrics[method_names.index(method),:,2].mean()-oracle) for method in ["P","A","D"]}})
            if row.world==a["primary_world"] and row.context==a["primary_context"] and row.window in a["primary_windows"] and phase in a["primary_phases"]:
                primary_rows.append({**metadata,"phase":phase})
                primary_values.append(phase_metrics[[method_names.index(name) for name in PRIMARY_METHODS],:,2].T)
                replica_values.append(replicas[:,phase_number,2])
    values=np.stack(primary_values);redraw=np.stack(replica_values);groups=pd.DataFrame(primary_rows)
    assert values.shape==(48,64,5) and redraw.shape==(48,64)
    means=values.mean(axis=1)
    result=loss_readout(means)
    distribution=path_bootstrap(values,a["bootstrap_replicates"],a["bootstrap_seed"])
    path_ci=np.quantile(distribution,[.025,.975],axis=0)
    mc_distribution=replica_bootstrap(redraw,means,a["bootstrap_replicates"],a["mc_bootstrap_seed"])
    mc_se=float(mc_distribution[:,2].std(ddof=1))
    first_gain=float(np.abs(means[:,0]-means[:,1]).mean()-np.abs(means[:,3]-means[:,1]).mean())
    last_gain=float(np.abs(means[:,0]-means[:,1]).mean()-np.abs(means[:,4]-means[:,1]).mean())
    per_fit=[]
    for (panel,architecture,seed),selected in groups.groupby(["panel","architecture","seed"]):
        fit=loss_readout(means[selected.index])
        per_fit.append({"panel":int(panel),"architecture":architecture,"seed":int(seed),
            "population_mae":float(fit[0]),"redraw_mae":float(fit[1]),"gain":float(fit[2])})
    fits=pd.DataFrame(per_fit)
    by_arch=fits.groupby("architecture").gain.mean().to_dict()
    by_panel={str(k):float(v) for k,v in fits.groupby("panel").gain.mean().items()}
    relative=float(result[2]/result[0]) if result[0]>1e-12 else None
    mc_fraction=mc_se/max(abs(float(result[2])),1e-12)
    half_fraction=abs(first_gain-last_gain)/max(abs(float(result[2])),1e-12)
    g=a["refinement_gate"]
    gate={"gain_positive":bool(result[2]>0),"path_interval_positive":bool(path_ci[0,2]>0),
        "relative_reduction_sufficient":bool(relative is not None and relative>=g["minimum_relative_MAE_reduction"]),
        "both_architectures_positive":bool(sum(x>0 for x in by_arch.values())>=g["positive_architectures"]),
        "panels_positive":bool(sum(x>0 for x in by_panel.values())>=g["minimum_positive_panels"]),
        "mc_se_small":bool(mc_fraction<=g["maximum_mc_se_fraction_of_gain"]),
        "half_split_stable":bool(half_fraction<=g["maximum_first32_last32_gain_difference_fraction"])}
    gate["passed"]=all(gate.values())
    primary_summary={"population_mae":float(result[0]),"redraw_mae":float(result[1]),"gain":float(result[2]),
        "relative_mae_reduction":relative,"path95_gain":path_ci[:,2].tolist(),"redraw_mc_se_gain":mc_se,
        "redraw_mc95_gain":np.quantile(mc_distribution[:,2],[.025,.975]).tolist(),
        "first32_gain":first_gain,"last32_gain":last_gain,"mc_se_fraction_of_gain":mc_fraction,
        "first_last_difference_fraction":half_fraction,"gain_by_architecture":by_arch,"gain_by_panel":by_panel,
        "refinement_gate":gate,"inference":"Path interval conditional on12savedfits+64redraws; MC interval conditions on originalpaths/fits, notjoint training uncertainty"}
    OUT.mkdir(parents=True)
    for name,rows in [("cell_metrics",cell_rows),("log_moments",moment_rows),("local_approximation",approx_rows),
                      ("primary_groups",primary_rows),("primary_fits",per_fit)]:
        pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False)
    np.savez_compressed(OUT/"primary_arrays.npz",path_metrics=values,redraw_replica_risk=redraw,
        path_bootstrap=distribution,redraw_bootstrap=mc_distribution,methods=np.array(PRIMARY_METHODS))
    (OUT/"primary_summary.json").write_text(json.dumps(primary_summary,indent=2,allow_nan=False)+"\n")
    checks.update(groups=192,primary_cells=48,source_files=len(sources),
        maximum_log_secondmoment_identity_error=max_moment_error,maximum_raw_unit_error=max_raw_error,
        maximum_redraw_aggregation_error=max_replica_error)
    out_manifest={"status":"complete","completed_at":datetime.now().astimezone().isoformat(),
        "protocol_sha256":sha(PROTOCOL),"script_sha256":sha(__file__),"sources":sources,"checks":checks,
        "primary_summary":primary_summary,"outputs":{f.name:sha(f) for f in sorted(OUT.iterdir())}}
    (OUT/"manifest.json").write_text(json.dumps(out_manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps(primary_summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-check",action="store_true");args=parser.parse_args()
    print(json.dumps(self_check())) if args.self_check else run()
