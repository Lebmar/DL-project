"""Post-hoc aggregation sensitivity prompted by exact-risk/cross-path diagnosis.

Does not replace the original frozen phase-mean estimand. No new observations.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/"outputs/long_window_diagnostic"
OUT=ROOT/"outputs/long_window_aggregation"
SOURCE=ROOT/"outputs/neural_compatibility_evaluation"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readout(means):
    """[fit,...,time,P/A/D]; bootstrap axis, if present, remains."""
    error_p=means[...,0]-means[...,1]
    error_d=means[...,2]-means[...,1]
    pooled_p=np.abs(error_p.mean(axis=-1)).mean(axis=0)
    pooled_d=np.abs(error_d.mean(axis=-1)).mean(axis=0)
    endpoint_p=np.abs(error_p).mean(axis=-1).mean(axis=0)
    endpoint_d=np.abs(error_d).mean(axis=-1).mean(axis=0)
    return np.stack((pooled_p-pooled_d,endpoint_p-endpoint_d,
                    endpoint_p-pooled_p,endpoint_d-pooled_d),axis=-1)


def run():
    assert not OUT.exists()
    original=json.loads((ROOT/"configs/neural_compatibility_protocol.json").read_text())
    e=original["evaluation"]
    dm=json.loads((IN/"manifest.json").read_text())
    assert dm["status"]=="complete"
    assert sha(IN/"endpoints.csv")==dm["outputs"]["endpoints.csv"]
    endpoints=pd.read_csv(IN/"endpoints.csv")
    keys=["panel","architecture","seed","world","context","window"]
    rows=[]
    for key,g in endpoints.groupby(keys,sort=True):
        meta=dict(zip(keys,key))
        for phase,indices in e["phases"].items():
            subset=g[g.time.isin(np.array(e["times"])[indices])]
            assert len(subset)==4
            for method in ["P","D","X"]:
                err=subset[method+"_risk"]-subset.A_risk
                pooled=abs(float(err.mean()));endpoint=float(err.abs().mean())
                assert endpoint>=pooled-1e-12
                rows.append({**meta,"phase":phase,"method":method,"phase_mean_absolute_error":pooled,
                    "endpoint_mean_absolute_error":endpoint,"cancellation":endpoint-pooled})
    index=pd.read_csv(SOURCE/"index.csv")
    selected=index[(index.world=="gaussian_up2")&(index.context==64)&(index.window==128)]
    arrays=[]
    sources={str((IN/"manifest.json").relative_to(ROOT)):sha(IN/"manifest.json"),str((IN/"endpoints.csv").relative_to(ROOT)):sha(IN/"endpoints.csv")}
    for row in selected.itertuples(index=False):
        file=SOURCE/row.file;assert sha(file)==row.sha256
        sources[str(file.relative_to(ROOT))]=sha(file)
        with np.load(file,allow_pickle=False) as f:
            # [methods,path,time] -> [path,time,method]; only Gaussian-up entry.
            metrics=f["metrics"]
            arrays.append(np.stack([metrics[k,:,4:8,2] for k in [2,3,4]],axis=-1))
    values=np.stack(arrays)
    assert values.shape==(12,64,4,3)
    point=readout(values.mean(axis=1))
    assert abs(point[0]-point[1]+point[2]-point[3])<1e-12
    rng=np.random.default_rng(2026101810)
    repetitions=8000
    draws=np.empty((repetitions,4))
    for start in range(0,repetitions,128):
        count=min(128,repetitions-start)
        ii=rng.integers(64,size=(count,64))
        draws[start:start+count]=readout(values[:,ii].mean(axis=2))
        # Separately implemented bootstrap check, first batch only.
        if start==0:
            one=values[:,ii[0]].mean(axis=1)
            p=one[:,:,0]-one[:,:,1];d=one[:,:,2]-one[:,:,1]
            independent=np.array([np.mean(np.abs(p.mean(axis=1)))-np.mean(np.abs(d.mean(axis=1))),
                np.mean(np.abs(p))-np.mean(np.abs(d)),np.mean(np.abs(p))-np.mean(np.abs(p.mean(axis=1))),
                np.mean(np.abs(d))-np.mean(np.abs(d.mean(axis=1)))])
            np.testing.assert_allclose(draws[0],independent,atol=1e-14)
    np.testing.assert_allclose(draws[:,0]-draws[:,1]+draws[:,2]-draws[:,3],0,atol=1e-14)
    names=["phase_gain","endpoint_gain","P_cancellation","D_cancellation"]
    summary={name:{"mean":float(point[i]),"posthoc_conditional_path95":np.quantile(draws[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)}
    summary["interpretation"]="Exploratory comparison on previously viewed data; fixed12fits and64redraws; no training-population claim or original-estimand replacement. Bootstrap percentile approximation may be nonregular near zero absolute errors."
    OUT.mkdir()
    pd.DataFrame(rows).to_csv(OUT/"cell_aggregation.csv",index=False)
    np.savez_compressed(OUT/"primary_bootstrap.npz",values=values,draws=draws,names=np.array(names))
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    manifest={"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),
        "sources":sources,"scope":"Posthoc aggregation diagnostic after inspecting exact-risk results; original study untouched",
        "bootstrap_seed":2026101810,"bootstrap_repetitions":repetitions,
        "outputs":{f.name:sha(f) for f in sorted(OUT.iterdir())}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":run()
