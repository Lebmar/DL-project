"""Precommitted fresh-pool validation on Slurm; fixed models and fixed seeds.

All candidate ranks are scored before mixing. The independent-redraw bank never
enters the model; X excludes the diagonal and uses only the same pool's donors.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import finite_mechanism_math as fm
import evaluate_neural_compatibility as bridge
from evaluate_stage3 import issued_scales, FAST_ATOL, FAST_RTOL
from stage2_models import make_model

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/long_window_validation_protocol.json'
METHODS = ['P','A','D','X','D_first32','D_last32']
METRICS = ['risk','half','coverage','tangent','convex','eta','eta2','cdf_deficit','recent_deficit','event_deficit']
SOURCES = ['scripts/evaluate_long_window_validation.py', 'scripts/analyze_long_window_validation.py',
           'scripts/verify_long_window_validation.py', 'scripts/evaluate_neural_compatibility.py',
           'scripts/train_neural_compatibility.py', 'hpc/long_window_validation.sbatch']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def candidate_metrics(absolute_eps, archive, current, qp, family, ages, onset=256, duration=128, context=64):
    """Broadcast leading axes, historical age last; return ten averaged metrics."""
    scores = np.asarray(absolute_eps)*np.exp(-archive)
    population_half = np.exp(current)*qp
    population_risk = fm.expected_interval_score(population_half, family)
    slope = 20*(fm.absolute_cdf(population_half, family)-.9)
    expected_tail = 1-fm.absolute_cdf(qp[...,None]*np.exp(archive), family)
    residual = (scores > qp[...,None])-expected_tail
    recent = np.arange(scores.shape[-1]) >= max(0,scores.shape[-1]-context)
    event = (ages >= onset)&(ages < onset+duration)
    deficits = [residual.mean(-1),(residual*recent).mean(-1),(residual*event).mean(-1)]
    ranks = fm.rank_mixture(scores.shape[-1], .1, 'randomized')
    scores.partition([k-1 for k,_ in ranks], axis=-1)
    result = np.zeros(scores.shape[:-1]+(len(METRICS),))
    for rank, weight in ranks:
        q = scores[...,rank-1]
        half = np.exp(current)*q
        risk = fm.expected_interval_score(half, family)
        tangent = slope*(half-population_half)
        convex = risk-population_risk-tangent
        assert np.min(convex) >= -1e-10
        eta = np.log(q/qp)
        result += weight*np.stack([risk,half,fm.absolute_cdf(half,family),tangent,convex,eta,eta**2,*deficits],axis=-1)
    assert np.isfinite(result).all()
    return result


def evaluate_banks(eps, redraw, delta, times, window, family, e):
    n, replicas = len(eps), len(redraw)
    ages = times[:,None]-np.arange(window,0,-1)
    archive, current = delta[:,ages], delta[:,times]
    qp = fm.population_mixture_quantile(archive.reshape(-1,window), .9, family).reshape(current.shape)
    residual = np.max(np.abs(fm.mixture_cdf(qp.ravel(),archive.reshape(-1,window),family)-.9))
    assert residual < 2e-12
    half = np.exp(current)*qp
    pop = np.zeros(half.shape+(len(METRICS),))
    pop[...,0],pop[...,1],pop[...,2] = fm.expected_interval_score(half,family),half,fm.absolute_cdf(half,family)
    actual = candidate_metrics(np.abs(eps[:,ages]),archive,current,qp,family,ages,e['onset'],e['duration'])
    dparts = []
    replica_endpoint = np.empty((replicas,len(times),len(METRICS)))
    for start in range(0,replicas,8):
        stop = min(start+8,replicas)
        values = candidate_metrics(np.abs(redraw[start:stop,:,ages]),archive,current,qp,family,ages,e['onset'],e['duration'])
        replica_endpoint[start:stop] = values.mean(axis=1)
        dparts.append(values.sum(axis=0))
    first = sum(dparts[:len(dparts)//2])/(replicas//2)
    last = sum(dparts[len(dparts)//2:])/(replicas//2)
    donor_sum = np.zeros_like(actual)
    pairs = np.empty((n,n,len(times)))
    diagonal = np.empty_like(actual)
    for start in range(0,n,8):
        stop = min(start+8,n)
        donor = candidate_metrics(np.abs(eps[None,start:stop,ages]),archive[:,None],current[:,None],qp[:,None],family,ages,e['onset'],e['duration'])
        pairs[:,start:stop] = donor[...,0]
        donor_sum += donor.sum(axis=1)
        for i in range(start,stop):
            diagonal[i] = donor[i,i-start]
    np.testing.assert_allclose(diagonal,actual,rtol=0,atol=1e-12)
    cross = (donor_sum-diagonal)/(n-1)
    metrics = np.stack([pop,actual,(first+last)/2,cross,first,last])
    z = float(fm.absolute_ppf(.9,family))
    assert metrics[...,0].min() >= float(fm.expected_interval_score(z,family))-1e-10
    assert metrics[...,4].min() >= -1e-10
    slope = 20*(fm.absolute_cdf(half,family)-.9)
    np.testing.assert_allclose(metrics[...,3],slope[None]*(metrics[...,1]-half[None]),atol=1e-11,rtol=1e-11)
    np.testing.assert_allclose(metrics[...,0]-pop[None,...,0],metrics[...,3]+metrics[...,4],atol=1e-11,rtol=1e-11)
    return metrics, replica_endpoint, pairs, slope, float(residual)


def deterministic_controls(eps, sigma, times, window, family):
    """Same scale path for all anchors: donor permutation leaves mean risk fixed."""
    maximum = 0.
    ages = times[:,None]-np.arange(window,0,-1)
    for fixed_scale in (np.ones_like(sigma),sigma):
        delta = np.log(fixed_scale/sigma)
        archive,current = delta[:,ages],delta[:,times]
        qp = fm.population_mixture_quantile(archive[:1].reshape(-1,window),.9,family).reshape(1,len(times))
        # Compute each donor once: every anchor has the identical denominator.
        bank = candidate_metrics(np.abs(eps[:,ages]),archive,current,qp,family,ages)
        x = (bank.sum(axis=0)[None]-bank)/(len(eps)-1)
        error = float(np.max(np.abs((bank[...,0]-x[...,0]).mean(axis=0))))
        assert error < 1e-12
        maximum = max(maximum,error)
    return maximum


def self_check():
    rng = np.random.default_rng(918821)
    eps = rng.normal(size=(4,180)); redraw = rng.normal(size=(16,4,180))
    delta = rng.uniform(-.3,.3,size=eps.shape)
    times = np.array([129,143,161,170]); e={'onset':140,'duration':20}
    error=0.
    for family in ('gaussian','t5'):
        for window in (32,128):
            metrics,reps,pairs,slope,_=evaluate_banks(eps,redraw,delta,times,window,family,e)
            for i in range(4):
                for j in range(4):
                    for ti,tt in enumerate(times):
                        bank=np.sort(np.abs(eps[j,tt-window:tt])*np.exp(-delta[i,tt-window:tt]))
                        direct=sum(weight*fm.expected_interval_score(np.exp(delta[i,tt])*bank[k-1],family) for k,weight in fm.rank_mixture(window,.1,'randomized'))
                        error=max(error,abs(float(direct-pairs[i,j,ti])))
            np.testing.assert_allclose(reps[...,0].mean(0),metrics[2,...,0].mean(0),atol=1e-12)
            sigma=np.ones_like(eps);sigma[:,140:160]=2
            deterministic_controls(eps,sigma,times,window,family)
    assert error<1e-12
    torch.manual_seed(918821)
    causal_error=0.
    for arch in ('gru','tcn'):
        model=make_model(arch)
        a=issued_scales(model,eps[:2],64,129,130)
        changed=eps[:2].copy();changed[:,129:]=1e4
        b=issued_scales(model,changed,64,129,130)
        np.testing.assert_allclose(a,b,atol=FAST_ATOL,rtol=FAST_RTOL)
        causal_error=max(causal_error,float(np.abs(a-b).max()))
    return {'passed':True,'independent_scalar_pair_sort_error':error,'future_perturbation_error':causal_error}


def run(out):
    start=time.time()
    assert os.environ.get('SLURM_JOB_ID'), 'Bulk validation must run in an HPC Slurm job'
    p=json.loads(PROTOCOL.read_text()); e=p['evaluation']
    assert p['frozen'] and not out.exists(), 'Refusing overwrite: use original frozen inputs for any separately recorded retry'
    for path in [PROTOCOL,*[ROOT/s for s in SOURCES]]:
        bridge.committed(path)
    for name,digest in p['inputs'].items():
        assert sha(ROOT/name)==digest,name
    assert e['independent_pools']==16 and e['paths_per_pool']==64 and e['D_replicas']==64
    assert e['length']==640 and e['alpha']==.1 and e['windows']==[32,128] and p['models']['contexts']==[64]
    assert e['phases']=={name:list(range(i*4,i*4+4)) for i,name in enumerate(['pre','entry','late','recovery'])}
    assert len(e['seeds'])==16 and len({v for row in e['seeds'] for k,v in row.items() if k!='pool'})==64
    old=json.loads(bridge.PROTOCOL.read_text())
    training,models,fast_checks,training_hashes=bridge.verified_inputs(old)
    checks=self_check()
    out.mkdir(parents=True); (out/'groups').mkdir();(out/'pools').mkdir()
    m={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(PROTOCOL),
       'freeze_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'sources':{s:sha(ROOT/s) for s in SOURCES},'inputs':p['inputs'],'training_artifact_sha256':training_hashes,
       'models':[fit for fit,_ in models],'validation_fast_inference_checks':fast_checks,
       'environment':{'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,
         'numpy':np.__version__,'torch_threads':torch.get_num_threads(),
         'slurm':{k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE']}},
       'method_names':METHODS,'metric_names':METRICS,'times':e['times'],'completed_pools':0,'completed_groups':0,'checks':checks}
    save_json(out/'manifest.json',m)
    save_json(out/'frozen_protocol.json',p)
    (out/'environment.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
    times=np.asarray(e['times']); index=[]; controls=[]
    maxima={'population_root_residual':0.,'deterministic_pairing_gap':0.,'onset_raw_width_gap':0.,'causal_scale_gap':0.}
    minimum_convex=0.
    for seeds in e['seeds']:
        pool=seeds['pool']; folder=out/'pools'/f'pool{pool:02d}';folder.mkdir();(folder/'scales').mkdir()
        innovations={};redraws={}
        for family in ('gaussian','t5'):
            innovations[family]=bridge.generate_innovations(family,(64,640),seeds[family])
            redraws[family]=bridge.generate_innovations(family,(64,64,640),seeds['redraw_'+family])
            np.savez_compressed(folder/f'redraw_{family}.npz',innovations=redraws[family])
        worlds={}
        for world in e['worlds']:
            name,family=world['name'],world['distribution'];sigma=np.ones((64,640))
            sigma[:,e['onset']:e['onset']+e['duration']]=world['amplitude'];y=sigma*innovations[family]
            np.savez_compressed(folder/f'test_{name}.npz',innovations=innovations[family],true_scale=sigma,observations=y)
            worlds[name]=(family,sigma,y)
            for window in e['windows']:
                error=deterministic_controls(innovations[family],sigma,times,window,family)
                maxima['deterministic_pairing_gap']=max(maxima['deterministic_pairing_gap'],error)
                controls.append({'pool':pool,'world':name,'window':window,'deterministic_A_minus_X_maxabs':error})
        for fit,model in models:
            onset_reference={}
            for name,(family,sigma,y) in worlds.items():
                scales=issued_scales(model,y,64)
                assert np.isfinite(scales).all() and np.all(scales>0)
                key=f"panel{fit['panel']}_{fit['architecture']}_seed{fit['seed']}_{name}"
                scale_file=folder/'scales'/(key+'.npz')
                compact=scales.astype(np.float32);np.testing.assert_array_equal(compact.astype(float),scales)
                np.savez_compressed(scale_file,scales=compact)
                # Each model/pool/world: perturb all current/future values at onset.
                changed=y[:2].copy();changed[:,e['onset']:]=1e4
                causal=issued_scales(model,changed,64,e['onset'],e['onset']+1)
                reference=scales[:2,e['onset']:e['onset']+1]
                np.testing.assert_allclose(causal,reference,rtol=FAST_RTOL,atol=FAST_ATOL)
                maxima['causal_scale_gap']=max(maxima['causal_scale_gap'],float(np.abs(causal-reference).max()))
                delta=np.log(scales/sigma)
                for window in e['windows']:
                    metrics,replicas,pairs,slope,root_error=evaluate_banks(innovations[family],redraws[family],delta,times,window,family,e)
                    maxima['population_root_residual']=max(maxima['population_root_residual'],root_error)
                    minimum_convex=min(minimum_convex,float(metrics[...,4].min()))
                    if family=='gaussian':
                        onset_half=metrics[:4,:,4,1]*sigma[:,e['onset']][None]
                        if name=='gaussian_flat':onset_reference[window]=onset_half
                        else:
                            np.testing.assert_allclose(onset_half,onset_reference[window],rtol=1e-10,atol=1e-10)
                            maxima['onset_raw_width_gap']=max(maxima['onset_raw_width_gap'],float(np.abs(onset_half-onset_reference[window]).max()))
                    output=out/'groups'/f'pool{pool:02d}_{key}_c64_w{window}.npz'
                    payload={'metrics':metrics,'methods':np.array(METHODS),'metric_names':np.array(METRICS),
                      'D_replica_endpoint':replicas,'D_replica_phase':replicas.reshape(64,4,4,len(METRICS)).mean(axis=2),
                      'times':times,'slope_P':slope}
                    if name in ('gaussian_flat','gaussian_up2') and window==128:
                        payload['pair_risk_phase']=pairs.reshape(64,64,4,4).mean(axis=-1).transpose(2,0,1)
                    np.savez_compressed(output,**payload)
                    index.append({'pool':pool,**fit,'world':name,'distribution':family,'context':64,'window':window,
                       'scale_file':scale_file.relative_to(out).as_posix(),'scale_sha256':sha(scale_file),
                       'file':output.relative_to(out).as_posix(),'sha256':sha(output)})
                    m['completed_groups']+=1
        m['completed_pools']+=1;m['checks'].update(maxima);save_json(out/'manifest.json',m)
        print(json.dumps({'completed_pool':pool,'groups':m['completed_groups']}),flush=True)
    assert len(index)==e['neural_groups']==1920
    for name,digest in {**p['inputs'],**training_hashes}.items():assert sha(ROOT/name)==digest,name
    pd.DataFrame(index).to_csv(out/'index.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'negative_controls.csv',index=False)
    m['checks'].update(minimum_convex_remainder=minimum_convex,all_diagonal_replays_passed=True,all_oracle_bounds_passed=True)
    m.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start)
    m['outputs']={f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json'}
    save_json(out/'manifest.json',m)
    print(json.dumps({'status':'complete','pools':16,'groups':1920,'wall_seconds':m['wall_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/long_window_validation')
    args=parser.parse_args();torch.set_num_threads(4)
    print(json.dumps(self_check())) if args.self_check else run(args.output.resolve())
