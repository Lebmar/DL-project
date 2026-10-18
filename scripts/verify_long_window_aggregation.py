"""Independent count-matrix check of the post-hoc aggregation sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/"outputs/long_window_aggregation"
OUT=ROOT/"outputs/long_window_aggregation_verification.json"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(check_only):
    assert check_only or not OUT.exists()
    m=json.loads((DIR/"manifest.json").read_text())
    assert m["status"]=="complete"
    assert sha(ROOT/"scripts/analyze_long_window_aggregation.py")==m["script_sha256"]
    for base,items in [(ROOT,m["sources"]),(DIR,m["outputs"])]:
        for name,digest in items.items():assert sha(base/name)==digest
    with np.load(DIR/"primary_bootstrap.npz",allow_pickle=False) as f:
        values=f["values"];recorded=f["draws"];names=f["names"].tolist()
    index=pd.read_csv(ROOT/"outputs/neural_compatibility_evaluation/index.csv")
    selected=index[(index.world=="gaussian_up2")&(index.context==64)&(index.window==128)]
    for i,row in enumerate(selected.itertuples(index=False)):
        with np.load(ROOT/"outputs/neural_compatibility_evaluation"/row.file,allow_pickle=False) as f:
            np.testing.assert_array_equal(values[i],np.moveaxis(f["metrics"][[2,3,4],:,4:8,2],0,-1))
    rng=np.random.default_rng(m["bootstrap_seed"]);reps=m["bootstrap_repetitions"]
    independent=np.empty_like(recorded)
    for start in range(0,reps,128):
        count=min(128,reps-start);ii=rng.integers(64,size=(count,64))
        multiplicity=np.zeros((count,64));np.add.at(multiplicity,(np.repeat(np.arange(count),64),ii.ravel()),1.)
        means=np.einsum('bp,fptm->fbtm',multiplicity/64,values,optimize=False)
        ep=means[...,0]-means[...,1];ed=means[...,2]-means[...,1]
        pm=np.abs(ep.mean(axis=2)).mean(axis=0);dm=np.abs(ed.mean(axis=2)).mean(axis=0)
        pe=np.abs(ep).mean(axis=(0,2));de=np.abs(ed).mean(axis=(0,2))
        independent[start:start+count]=np.stack([pm-dm,pe-de,pe-pm,de-dm],axis=-1)
    error=float(np.max(np.abs(independent-recorded)))
    np.testing.assert_allclose(independent,recorded,rtol=0,atol=1e-12)
    np.testing.assert_allclose(recorded[:,0]-recorded[:,1]+recorded[:,2]-recorded[:,3],0,atol=1e-12)
    summary=json.loads((DIR/"summary.json").read_text())
    for i,name in enumerate(names):
        np.testing.assert_allclose(summary[name]["posthoc_conditional_path95"],np.quantile(independent[:,i],[.025,.975]),atol=1e-12)
    cells=pd.read_csv(DIR/"cell_aggregation.csv")
    assert len(cells)==2304
    np.testing.assert_allclose(cells.endpoint_mean_absolute_error-cells.phase_mean_absolute_error,cells.cancellation,atol=1e-12)
    assert cells.cancellation.min()>=-1e-12
    result={"passed":True,"script_sha256":sha(__file__),"manifest_sha256":sha(DIR/"manifest.json"),
        "bootstrap_replicates":reps,"maximum_count_matrix_difference":error,"cell_cancellation_identities":len(cells),
        "scope":"Independent arithmetic replay of exploratory aggregation statistics; does not provide fresh-data confirmation"}
    if not check_only:OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');run(p.parse_args().check_only)
