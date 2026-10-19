"""Exploratory population-matched homogeneous finite-rank risk baseline.

Reuses saved evaluation arrays. This is an offline oracle diagnostic, not an
online forecaster and not a new confirmatory test on unseen observations.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta, norm, t
import finite_mechanism_math as fm

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/population_matched_baseline_protocol.json'
SOURCE=ROOT/'outputs/long_window_validation'
ANALYSIS=ROOT/'outputs/long_window_validation_analysis'
OUT=ROOT/'outputs/population_matched_baseline'
METHODS=['P','A','D','B']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_baseline(half,family,window,nodes):
    """Expected [risk, halfwidth, coverage] under a homogeneous iid surrogate."""
    half=np.asarray(half,float);z=float(fm.absolute_ppf(.9,family))
    result=np.zeros(half.shape+(3,))
    for rank,prob in fm.rank_mixture(window,.1,'randomized'):
        u,weights=fm.beta_quadrature(window,rank,nodes)
        q=fm.absolute_ppf(u,family)/z
        for start in range(0,half.size,256):
            stop=min(start+256,half.size)
            a=half.ravel()[start:stop,None]*q[None]
            result.reshape(-1,3)[start:stop]+=prob*np.stack([
                fm.expected_interval_score(a,family)@weights,
                a@weights,fm.absolute_cdf(a,family)@weights],axis=-1)
    return result


def self_check():
    # Independent probability-space integration, no study observations/models.
    from verify_long_window_validation import risk as independent_risk
    max_error=0.;max_coverage_error=0.
    for family in ('gaussian','t5'):
        dist=norm if family=='gaussian' else t(df=5,scale=np.sqrt(3/5))
        z=float(dist.ppf(.95))
        for window in (32,128):
            for half in (.4,1.,1.6448536269514722,3.,6.):
                exact=0.
                target=(window+1)*.9;k=int(np.floor(target));w=target-k
                for rank,prob in [(k,1-w),(k+1,w)]:
                    val,err=quad(lambda u:independent_risk(half*dist.ppf((1+u)/2)/z,family)*beta.pdf(u,rank,window+1-rank),
                                 0,1,epsabs=2e-8,epsrel=2e-10,limit=250)
                    assert err<1e-6
                    exact+=prob*val
                approximate=finite_baseline(np.array(half),family,window,128)[0]
                max_error=max(max_error,abs(float(exact-approximate)))
            coverage=float(finite_baseline(np.array(z),family,window,128)[2])
            max_coverage_error=max(max_coverage_error,abs(coverage-.9))
    assert max_error<1e-5 and max_coverage_error<1e-12
    toy=np.random.default_rng(88341).normal(size=(3,2,4,4));pooled=toy.mean(0);values=estimands(pooled)
    mae_phase=[];mae_endpoint=[]
    for mi in range(4):
        errors=[[float(pooled[f,t,mi]-pooled[f,t,1]) for t in range(4)] for f in range(2)]
        mae_phase.append(np.mean([abs(np.mean(row)) for row in errors]))
        mae_endpoint.append(np.mean([[abs(x) for x in row] for row in errors]))
    expected=[mae_phase[3]-mae_phase[2],mae_endpoint[3]-mae_endpoint[2],mae_phase[0]-mae_phase[3],mae_endpoint[0]-mae_endpoint[3]]
    np.testing.assert_allclose(values,expected,rtol=0,atol=1e-14)
    return {'passed':True,'independent_beta_integral_max_error':max_error,'ideal_coverage_error':max_coverage_error,'new_study_data_generated':False}


def estimands(risk):
    """[...,fit,time,method] -> [...,4]: BvsD phase/endpoint,PvsB phase/endpoint."""
    e=risk-risk[...,1,None]
    phase=np.abs(e.mean(axis=-2)).mean(axis=-2)
    endpoint=np.abs(e).mean(axis=(-3,-2))
    return np.stack([phase[...,3]-phase[...,2],endpoint[...,3]-endpoint[...,2],
                     phase[...,0]-phase[...,3],endpoint[...,0]-endpoint[...,3]],axis=-1)


def run():
    assert os.environ.get('SLURM_JOB_ID'),'Batch benchmark requires Slurm'
    started=time.time();p=json.loads(PROTOCOL.read_text());assert p['frozen'] and not OUT.exists()
    sources=[PROTOCOL,Path(__file__).resolve(),ROOT/'hpc/population_matched_baseline.sbatch']
    for path in sources:
        frozen=subprocess.check_output(['git','show',f'HEAD:{path.relative_to(ROOT)}'],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==sha(path)
    for name,digest in p['inputs'].items():assert sha(ROOT/name)==digest,name
    check=self_check();m=json.loads((SOURCE/'manifest.json').read_text());assert m['status']=='complete'
    assert sha(SOURCE/'index.csv')==m['outputs']['index.csv']
    index=pd.read_csv(SOURCE/'index.csv');assert len(index)==1920
    fits=sorted(set(index[['panel','architecture','seed']].itertuples(index=False,name=None)))
    worlds=['gaussian_flat','gaussian_up1p5','gaussian_up2','gaussian_up3','t5_up2']
    conditions=[(world,w) for world in worlds for w in (32,128)]
    means=np.empty((16,12,10,16,4));diffs=[];rows=[];minimum=0.;extremes={}
    OUT.mkdir()
    # Per-path B metrics stay on HPC; aggregate means and hashes are public.
    path_data=np.empty((1920,64,16,3))
    for gi,row in enumerate(index.itertuples(index=False)):
        assert sha(SOURCE/row.file)==row.sha256
        with np.load(SOURCE/row.file,allow_pickle=False) as f:metrics=f['metrics']
        key=(row.distribution,row.window)
        lo,hi=extremes.get(key,(float('inf'),0.))
        extremes[key]=(min(lo,float(metrics[0,...,1].min())),max(hi,float(metrics[0,...,1].max())))
        primary=finite_baseline(metrics[0,...,1],row.distribution,row.window,p['quadrature']['main_nodes'])
        coarse=finite_baseline(metrics[0,...,1],row.distribution,row.window,p['quadrature']['check_nodes'])
        discrepancy=float(abs(primary[...,0]-coarse[...,0]).max());diffs.append(discrepancy)
        z=float(fm.absolute_ppf(.9,row.distribution));minimum=min(minimum,float(primary[...,0].min()-fm.expected_interval_score(z,row.distribution)))
        assert minimum>=-1e-10
        path_data[gi]=primary
        pi=row.pool-1;fi=fits.index((row.panel,row.architecture,row.seed));ci=conditions.index((row.world,row.window))
        means[pi,fi,ci]=np.stack([metrics[0,...,0],metrics[1,...,0],metrics[2,...,0],primary[...,0]],axis=-1).mean(axis=0)
        rows.append({'group_id':gi,'pool':row.pool,'panel':row.panel,'architecture':row.architecture,'seed':row.seed,
                     'world':row.world,'window':row.window,'quadrature_risk_discrepancy':discrepancy})
    np.savez_compressed(OUT/'path_metrics.npz',metrics=path_data,metric_names=np.array(['risk','half','coverage']))
    pd.DataFrame(rows).to_csv(OUT/'groups.csv',index=False)
    statnames=['G_B_vs_D_phase','G_B_vs_D_endpoint','G_P_vs_B_phase','G_P_vs_B_endpoint']
    phases={'pre':[0,1,2,3],'entry':[4,5,6,7],'late':[8,9,10,11],'recovery':[12,13,14,15]}
    scopes=[('all','all',list(range(16)),list(range(12)))]+[
        ('fit',str(i),list(range(16)),[i]) for i in range(12)]+[
        ('panel',str(panel),list(range(16)),[i for i,f in enumerate(fits) if f[0]==panel]) for panel in (1,2,3)]+[
        ('pool',str(pool+1),[pool],list(range(12))) for pool in range(16)]
    grid=[]
    for scope,label,pools,fitids in scopes:
        pooled=means[pools][:,fitids].mean(0)
        for ci,(world,window) in enumerate(conditions):
            for phase,ids in phases.items():
                risk=pooled[:,ci][:,ids];stats=estimands(risk)
                err=risk-risk[...,1,None]
                mae_phase=np.abs(err.mean(1)).mean(0);mae_endpoint=np.abs(err).mean((0,1))
                record={'scope':scope,'scope_id':label,'world':world,'window':window,'phase':phase,**dict(zip(statnames,map(float,stats)))}
                for mi,name in enumerate(METHODS):record.update({name+'_phase_MAE':float(mae_phase[mi]),name+'_endpoint_MAE':float(mae_endpoint[mi])})
                grid.append(record)
    pd.DataFrame(grid).to_csv(OUT/'grid.csv',index=False)
    ci=conditions.index(('gaussian_up2',128));primary=means[:,:,ci][:,:,phases['entry']]
    am=json.loads((ANALYSIS/'manifest.json').read_text())
    assert sha(ANALYSIS/'inference.npz')==am['outputs']['inference.npz']
    with np.load(ANALYSIS/'inference.npz',allow_pickle=False) as f:
        counts=f['bootstrap_counts'];old_mc=f['mc_draws'];old_first=f['first32_point'];old_last=f['last32_point']
    assert counts.shape==(12000,16)
    draws=estimands(np.einsum('bp,pftm->bftm',counts/16,primary,optimize=True))
    points=estimands(primary.mean(0));interval=np.quantile(draws,[.025,.975],axis=0).T
    mc=old_mc-points[None,2:];mc_se=mc.std(axis=0,ddof=1);mc_diagnostics=[]
    for i,target in enumerate(['phase','endpoint']):
        denominator=max(abs(float(points[i])),1e-12);split=abs(float(old_first[i]-old_last[i]))
        se_fraction=float(mc_se[i]/denominator);split_fraction=split/denominator
        mc_diagnostics.append({'target':target,'standard_error':float(mc_se[i]),'se_fraction':se_fraction,'split_difference_fraction':split_fraction,'passed':se_fraction<=.1 and split_fraction<=.2})
    extreme_checks=[]
    from verify_long_window_validation import risk as independent_risk
    for (family,window),(lo,hi) in extremes.items():
        dist=norm if family=='gaussian' else t(df=5,scale=np.sqrt(3/5));z=float(dist.ppf(.95))
        for half in (lo,hi):
            integral=0.;target=(window+1)*.9;k=int(np.floor(target));w=target-k
            for rank,prob in [(k,1-w),(k+1,w)]:
                value,error=quad(lambda u:independent_risk(half*dist.ppf((1+u)/2)/z,family)*beta.pdf(u,rank,window+1-rank),0,1,epsabs=2e-8,epsrel=2e-10,limit=250)
                assert error<1e-6
                integral+=prob*value
            difference=abs(float(finite_baseline(np.array(half),family,window,128)[0])-integral)
            extreme_checks.append({'family':family,'window':window,'population_halfwidth':half,'independent_integral_error':difference})
    numerical=max(diffs+[x['independent_integral_error'] for x in extreme_checks])<=p['quadrature']['maximum_absolute_risk_discrepancy']
    np.savez_compressed(OUT/'aggregate.npz',means=means,primary=primary,bootstrap_draws=draws,points=points,mc_draws=mc,
                        conditions=np.array([f'{world}_w{window}' for world,window in conditions]),
                        methods=np.array(METHODS),fit_panel=np.array([f[0] for f in fits]),fit_architecture=np.array([f[1] for f in fits]),fit_seed=np.array([f[2] for f in fits]))
    summary={'status':'complete','scope':p['scope'],'numerical_precision_passed':numerical,'maximum_128_vs_64_risk_discrepancy':max(diffs),
             'self_check':check,'incremental_D_mc_precision':mc_diagnostics,'observed_extreme_halfwidth_checks':extreme_checks,'primary_readout':{name:{'estimate':float(points[i]),'exploratory_95_interval':interval[i].tolist()} for i,name in enumerate(statnames)},
             'interpretation':'B is exact only for the homogeneous surrogate, evaluated by numerical quadrature; P quantile uses oracle latent scales. This is not deployable or evidence of equivalence/confirmation. All new comparisons are post hoc.',
             'groups':1920,'new_training':False,'new_test_paths':False,'wall_seconds':time.time()-started}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest={'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),'freeze_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(__file__),'inputs':p['inputs'],'slurm_job_id':os.environ['SLURM_JOB_ID'],
              'outputs':{f.name:sha(f) for f in OUT.iterdir() if f.is_file()},'wall_seconds':time.time()-started}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    print(json.dumps(self_check(),indent=2)) if args.self_check else run()
