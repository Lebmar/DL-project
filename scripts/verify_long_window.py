"""Verify every cross-path bank via full sorting and independent SciPy risks.

No observations/fits are generated; saved original paths are re-paired only.
Does not import diagnose_long_window or its finite-mechanism numerical engine.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from verify_neural_compatibility import independent_distribution, independent_metrics, rank_weights

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT/"outputs/long_window_diagnostic"
SOURCE = ROOT/"outputs/neural_compatibility_evaluation"
OUT = ROOT/"outputs/long_window_verification.json"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(check_only):
    start = time.time()
    assert check_only or not OUT.exists()
    m = json.loads((IN/"manifest.json").read_text())
    assert m["status"] == "complete"
    hashes = 0
    for base, items in [(ROOT,m["inputs"]),(IN,m["outputs"])]:
        for name, digest in items.items():
            assert sha(base/name) == digest
            hashes += 1
    assert sha(ROOT/"scripts/diagnose_long_window.py") == m["source_sha256"]
    assert sha(ROOT/"configs/long_window_diagnostic_protocol.json") == m["protocol_sha256"]
    for name in ["scripts/diagnose_long_window.py","configs/long_window_diagnostic_protocol.json"]:
        frozen = subprocess.check_output(["git","show",f"{m['freeze_commit']}:{name}"],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest() == sha(ROOT/name)
    em = json.loads((SOURCE/"manifest.json").read_text())
    for name, digest in em["outputs"].items():
        assert sha(SOURCE/name) == digest
        hashes += 1
    e = json.loads((ROOT/"configs/neural_compatibility_protocol.json").read_text())["evaluation"]
    index = pd.read_csv(SOURCE/"index.csv")
    cells = pd.read_csv(IN/"cells.csv")
    endpoints = pd.read_csv(IN/"endpoints.csv")
    decompositions = pd.read_csv(IN/"risk_decomposition.csv")
    sensitivity = pd.read_csv(IN/"pairing_sensitivity.csv")
    groups = pd.read_csv(IN/"groups.csv")
    with np.load(IN/"pair_risk_by_phase.npz",allow_pickle=False) as f:
        saved_pairs = f["risk"]
        assert f["phases"].tolist() == list(e["phases"])
    assert len(index)==len(groups)==192 and len(cells)==3072 and len(endpoints)==3072
    assert len(decompositions)==1536 and len(sensitivity)==64
    keys=["panel","architecture","seed","world","context","window"]
    pd.testing.assert_frame_equal(groups[keys],index[keys],check_dtype=False)
    times=np.asarray(e["times"])
    maxima={"pair_risk":0.,"cell_metrics":0.,"endpoint_tail_deficit":0.,"decomposition":0.,"jackknife":0.,"onset_raw_halfwidth":0.}
    worlds, scales_cache = {}, {}
    source_before = {name:sha(ROOT/name) for name in m["inputs"]}
    for gi,row in enumerate(index.itertuples(index=False)):
        select=np.ones(len(cells),dtype=bool);eselect=np.ones(len(endpoints),dtype=bool);dselect=np.ones(len(decompositions),dtype=bool)
        for k in keys:
            select &= cells[k].eq(getattr(row,k));eselect &= endpoints[k].eq(getattr(row,k));dselect &= decompositions[k].eq(getattr(row,k))
        cc,ee,dd=cells[select],endpoints[eselect],decompositions[dselect]
        if row.world not in worlds:
            with np.load(SOURCE/f"test_{row.world}.npz",allow_pickle=False) as f:
                worlds[row.world]=(f["innovations"],f["true_scale"])
        eps,sigma=worlds[row.world]
        if row.scale_file not in scales_cache:
            with np.load(ROOT/row.scale_file,allow_pickle=False) as f:
                scales_cache[row.scale_file]=f["scales"].astype(float)
        scales=scales_cache[row.scale_file]
        delta=np.log(scales/sigma);current=delta[:,times];n=len(eps)
        ages=times[:,None]-np.arange(row.window,0,-1)
        with np.load(SOURCE/row.file,allow_pickle=False) as f:
            original=f["metrics"];moments=f["moments"]
        family=row.distribution;distribution=independent_distribution(family)
        z=distribution.ppf(.95);qp=z*np.exp(moments[0,...,0]-current)
        pair_risk=np.empty((n,n,len(times)))
        sums={name:np.zeros((n,len(times))) for name in ["risk","halfwidth","coverage","eta_mean","eta_second","cdf_deficit","recent_deficit","event_deficit"]}
        diagonal={name:np.zeros((n,len(times))) for name in sums}
        event=(ages>=e["onset"])&(ages<e["onset"]+e["duration"])
        recent=ages >= times[:,None]-min(row.window,row.context)
        expected_tail=2*distribution.sf(qp[...,None]*np.exp(delta[:,ages]))
        for lo in range(0,n,8):
            hi=min(lo+8,n)
            scores=np.abs(eps[None,lo:hi,ages])*np.exp(-delta[:,None,ages])
            residual=(scores>qp[:,None,:,None])-expected_tail[:,None]
            local={"cdf_deficit":residual.mean(axis=-1),"recent_deficit":(residual*recent).mean(axis=-1),"event_deficit":(residual*event).mean(axis=-1)}
            sorted_scores=np.sort(scores,axis=-1)
            for k in ["risk","halfwidth","coverage","eta_mean","eta_second"]:
                local[k]=np.zeros((n,hi-lo,len(times)))
            for rank,weight in rank_weights(row.window,.1):
                q=sorted_scores[...,rank-1]
                half=np.exp(current[:,None])*q
                values=independent_metrics(half,1.,family,.1)
                eta=np.log(q/qp[:,None])
                for name,v in [("risk",values[...,2]),("halfwidth",half),("coverage",values[...,0]),("eta_mean",eta),("eta_second",eta*eta)]:
                    local[name]+=weight*v
            pair_risk[:,lo:hi]=local["risk"]
            for name in sums:
                sums[name]+=local[name].sum(axis=1)
                donor_ids=np.arange(lo,hi)
                diagonal[name][donor_ids]=local[name][donor_ids,np.arange(hi-lo)]
        x={name:(sums[name]-diagonal[name])/(n-1) for name in sums}
        phased=np.stack([pair_risk[:,:,ids].mean(axis=-1) for ids in e["phases"].values()])
        maxima["pair_risk"]=max(maxima["pair_risk"],float(np.max(np.abs(phased-saved_pairs[gi]))))
        np.testing.assert_allclose(phased,saved_pairs[gi],rtol=0,atol=1e-10)
        base_half=original[2,...,1]/2;base_risk=original[2,...,2]
        slope=2/.1*(2*distribution.cdf(base_half)-1-.9)
        for phase,ids in e["phases"].items():
            for method,mi in [("P",2),("A",3),("D",4),("X",None)]:
                if mi is not None:
                    data={"risk":original[mi,...,2],"halfwidth":original[mi,...,1]/2,"coverage":original[mi,...,0],"eta_mean":moments[mi-2,...,2],"eta_second":moments[mi-2,...,3]}
                else:data=x
                tangent=slope*(data["halfwidth"]-base_half)
                convex=data["risk"]-base_risk-tangent
                assert convex.min()>-1e-10
                expected={k:float(v[:,ids].mean()) for k,v in data.items() if k in ["risk","halfwidth","coverage","eta_mean","eta_second"]}
                expected.update(exact_tangent=float(tangent[:,ids].mean()),convex_remainder=float(convex[:,ids].mean()))
                record=cc[(cc.phase==phase)&(cc.method==method)].iloc[0]
                for name,value in expected.items():
                    err=abs(record[name]-value);maxima["cell_metrics"]=max(maxima["cell_metrics"],err)
                    assert err<1e-10,(gi,name)
            for comparator,other_half,other_risk in [("D",original[4,...,1]/2,original[4,...,2]),("X",x["halfwidth"],x["risk"])]:
                width_diff=(original[3,...,1]/2-other_half)[:,ids]
                linear=(slope[:,ids]*width_diff).mean();gap=(original[3,...,2]-other_risk)[:,ids].mean()
                rr=dd[(dd.phase==phase)&(dd.contrast=="A-"+comparator)].iloc[0]
                for name,value in [("risk_gap",gap),("exact_linear",linear),("convex_gap",gap-linear),("linear_product_of_means",slope[:,ids].mean()*width_diff.mean()),("linear_covariance",linear-slope[:,ids].mean()*width_diff.mean())]:
                    error=abs(rr[name]-value);maxima["decomposition"]=max(maxima["decomposition"],error);assert error<1e-10
        for ti,tt in enumerate(times):
            record=ee[ee.time==tt].iloc[0]
            for method,bank in [("A",diagonal),("X",x)]:
                for name in ["cdf_deficit","recent_deficit","event_deficit"]:
                    error=abs(record[f"{method}_{name}"]-bank[name][:,ti].mean())
                    maxima["endpoint_tail_deficit"]=max(maxima["endpoint_tail_deficit"],error);assert error<1e-10
        if (gi+1)%64==0:print(json.dumps({"verified_groups":gi+1}),flush=True)
    for condition,selected in groups.groupby(["world","context","window"]):
        joint=saved_pairs[selected.index].mean(axis=0)
        for pi,phase in enumerate(e["phases"]):
            matrix=joint[pi];n=len(matrix)
            gap=np.diag(matrix).mean()-matrix[~np.eye(n,dtype=bool)].mean()
            loo=[]
            for i in range(n):
                reduced=np.delete(np.delete(matrix,i,axis=0),i,axis=1)
                loo.append(np.diag(reduced).mean()-reduced[~np.eye(n-1,dtype=bool)].mean())
            se=np.sqrt((n-1)*np.var(loo,ddof=0))
            record=sensitivity[(sensitivity.world==condition[0])&(sensitivity.context==condition[1])&(sensitivity.window==condition[2])&(sensitivity.phase==phase)].iloc[0]
            error=max(abs(record.A_minus_X-gap),abs(record.jackknife_se-se))
            maxima["jackknife"]=max(maxima["jackknife"],error);assert error<1e-10
    # Before any shocked observation enters history, Gaussian paired-world raw widths agree.
    onset=endpoints[endpoints.time==e["onset"]]
    pivot=onset.pivot(index=["panel","architecture","seed","context","window"],columns="world",values=["P_half","A_half","D_half","X_half"])
    for method in ["P","A","D","X"]:
        for world,amplitude in [("gaussian_up2",2.),("gaussian_down_half",.5)]:
            error=float(np.max(np.abs(pivot[(method+"_half",world)]*amplitude-pivot[(method+"_half","gaussian_flat")])))
            maxima["onset_raw_halfwidth"]=max(maxima["onset_raw_halfwidth"],error);assert error<1e-10
    for name,digest in source_before.items():assert sha(ROOT/name)==digest
    result={"passed":True,"completed_at":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(__file__),
        "independent_risk_helper_sha256":sha(ROOT/"scripts/verify_neural_compatibility.py"),"manifest_sha256":sha(IN/"manifest.json"),
        "verified_hashes":hashes,"verified_groups":192,"ordered_pair_targets_replayed":192*64*64*16,
        "maximum_errors":maxima,"wall_seconds":time.time()-start,"scope":"Independent full sorting/SciPy risk replay, no new data or fits; descriptive pairing uncertainty only"}
    if not check_only:OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--check-only",action="store_true")
    run(parser.parse_args().check_only)
