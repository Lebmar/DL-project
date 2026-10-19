"""Independent Slurm audit of the frozen long-window validation.

Uses SciPy laws and full sorting, never evaluator/analyzer calculation helpers.
A/P are replayed everywhere. D is exhaustive for the primary condition; X is
exhaustive for both conditions with retained pair matrices. Other D/X banks
are checked at four deterministic anchors, with scope recorded in the result.
--self-check uses tiny artificial examples only; production never runs locally.
"""
from __future__ import annotations

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
from scipy.optimize import brentq
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/long_window_validation_protocol.json'
EVAL = ROOT / 'outputs/long_window_validation'
ANALYSIS = ROOT / 'outputs/long_window_validation_analysis'
OUT = ROOT / 'outputs/long_window_validation_verification.json'
METHODS = ['P', 'A', 'D', 'X', 'D_first32', 'D_last32']
METRICS = ['risk', 'half', 'coverage', 'tangent', 'convex', 'eta', 'eta2',
           'cdf_deficit', 'recent_deficit', 'event_deficit']
KEYS = ['pool', 'panel', 'architecture', 'seed', 'world', 'context', 'window']
STATS = ['G_phase', 'G_endpoint', 'C', 'cancellation_P', 'cancellation_D']
SECONDARY_ANCHORS = np.array([0, 21, 42, 63])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def confined(base, name):
    path = (base / name).resolve()
    assert path.is_relative_to(base.resolve()), name
    return path


def check_hashes(base, mapping, complete=False):
    for name, digest in mapping.items():
        assert sha(confined(base, name)) == digest, name
    if complete:
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*')
                  if p.is_file() and p.name != 'manifest.json'}
        assert actual == set(mapping), (actual - set(mapping), set(mapping) - actual)
    return len(mapping)


def close(name, actual, expected, maxima, atol=1e-9, rtol=1e-11):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    maxima[name] = max(maxima.get(name, 0.), error)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=name)


def law(family):
    assert family in ('gaussian', 't5')
    return norm if family == 'gaussian' else t(df=5, scale=np.sqrt(3/5))


def risk(half, family, alpha=.1):
    """Integrate the two symmetric tails using SciPy survival functions."""
    half = np.asarray(half, dtype=float)
    if family == 'gaussian':
        tail_first = norm.pdf(half)
    else:
        scale = np.sqrt(3/5)
        unit = half / scale
        tail_first = scale * (5 + unit**2) / 4 * t.pdf(unit, df=5)
    stop_loss_one_tail = tail_first - half * law(family).sf(half)
    return 2*half + 4/alpha*stop_loss_one_tail


def rank_lottery(window, alpha=.1):
    target = (window+1)*(1-alpha)
    lower = int(np.floor(target))
    upper_probability = target-lower
    assert 1 <= lower <= window
    if upper_probability == 0:
        return [(lower-1, 1.)]
    assert lower+1 <= window
    return [(lower-1, 1-upper_probability), (lower, upper_probability)]


def full_sort_metrics(scores, archive, factor, qp, family, ages, e):
    """Leading axes may represent anchors, replicas or observed donors."""
    dist = law(family)
    base_half = factor*qp
    base_risk = risk(base_half, family, e['alpha'])
    derivative = 2/e['alpha']*(2*dist.cdf(base_half)-1-(1-e['alpha']))
    residual = (scores > qp[..., None]) - 2*dist.sf(qp[..., None]*np.exp(archive))
    recent = ages >= np.asarray(e['times'])[:, None]-min(scores.shape[-1], 64)
    event = (ages >= e['onset']) & (ages < e['onset']+e['duration'])
    deficits = [residual.mean(axis=-1), (residual*recent).mean(axis=-1),
                (residual*event).mean(axis=-1)]
    ordered = np.sort(scores, axis=-1)
    output = np.zeros(ordered.shape[:-1]+(10,))
    for rank, probability in rank_lottery(scores.shape[-1], e['alpha']):
        q = ordered[..., rank]
        half = factor*q
        value = risk(half, family, e['alpha'])
        tangent = derivative*(half-base_half)
        eta = np.log(q/qp)
        candidate = np.stack([value, half, 2*dist.cdf(half)-1, tangent,
                              value-base_risk-tangent, eta, eta**2, *deficits], axis=-1)
        output += probability*candidate
    return output


def independent_stats(pooled):
    """Input [..., fit, endpoint, method]; absolute values follow averaging."""
    ep = pooled[..., 0]-pooled[..., 1]
    ed = pooled[..., 2]-pooled[..., 1]
    pp = np.abs(ep.mean(axis=-1)).mean(axis=-1)
    dp = np.abs(ed.mean(axis=-1)).mean(axis=-1)
    pe = np.abs(ep).mean(axis=(-2, -1))
    de = np.abs(ed).mean(axis=(-2, -1))
    coupling = (pooled[..., 1]-pooled[..., 3]).mean(axis=(-2, -1))
    return np.stack([pp-dp, pe-de, coupling, pe-pp, de-dp], axis=-1)


def multiplicities(indices, population):
    result = np.zeros(indices.shape[:-1]+(population,), dtype=np.uint8)
    flat = result.reshape(-1, population)
    ii = indices.reshape(-1, indices.shape[-1])
    np.add.at(flat, (np.arange(len(flat))[:, None], ii), 1)
    return result


def self_check():
    """Numerical integration, scalar sorting and direct bootstrap tiny toys."""
    maxima = {}
    for family in ('gaussian', 't5'):
        dist = law(family)
        for half in (.2, 1., 1.7, 3.5):
            integrated = 2*half + 40*quad(lambda x: (x-half)*dist.pdf(x), half, np.inf)[0]
            close('risk_vs_quadrature', risk(half, family), integrated, maxima, atol=1e-9)
            h = 1e-5
            numeric = (risk(half+h, family)-risk(half-h, family))/(2*h)
            close('risk_derivative', numeric, 20*(2*dist.cdf(half)-1-.9), maxima, atol=2e-8)
    rng = np.random.default_rng(872681)
    e = {'times': [35, 38], 'onset': 36, 'duration': 3, 'alpha': .1}
    ages = np.array(e['times'])[:, None]-np.arange(32, 0, -1)
    archive = rng.uniform(-.2, .2, size=(3, 2, 32))
    factor = np.exp(rng.uniform(-.2, .2, size=(3, 2)))
    qp = np.ones((3, 2))*1.65
    scores = np.abs(rng.normal(size=(3, 2, 32)))*np.exp(-archive)
    got = full_sort_metrics(scores, archive, factor, qp, 'gaussian', ages, e)
    for i in range(3):
        for j in range(2):
            candidates = sorted(scores[i, j].tolist())
            expected = sum(w*risk(factor[i, j]*candidates[k], 'gaussian') for k,w in rank_lottery(32))
            close('scalar_sorted_risk', got[i,j,0], expected, maxima, atol=1e-12)
    # Unequal donor/anchor dimensions expose accidental axis swaps.
    eps = rng.normal(size=(5,42))
    delta = rng.normal(0,.1,size=(3,42))
    a = delta[:,ages]
    factor = np.exp(delta[:,e['times']])
    bank = np.abs(eps[None,:,ages])*np.exp(-a[:,None])
    for family in ('gaussian','t5'):
        paired = full_sort_metrics(bank,a[:,None],factor[:,None],qp[:,None],family,ages,e)
        for anchor in range(3):
            for donor in range(5):
                scalar = full_sort_metrics(bank[anchor,donor],a[anchor],factor[anchor],qp[anchor],family,ages,e)
                close('unequal_pair_axes',paired[anchor,donor],scalar,maxima,atol=1e-12)
        sigma = np.ones_like(eps)
        sigma[:,36:39] = 2
        maxima['deterministic_pair_control'] = deterministic_control(eps,sigma,32,family,e)
    values = rng.normal(size=(5, 3, 2, 6))
    indices = rng.integers(5, size=(9, 5))
    counts = multiplicities(indices, 5)
    by_counts = np.einsum('bp,pftm->bftm', counts/5, values)
    direct = np.stack([values[ii].mean(0) for ii in indices])
    close('count_matrix', by_counts, direct, maxima, atol=1e-13)
    close('statistics_from_counts', independent_stats(by_counts), independent_stats(direct), maxima, atol=1e-13)
    stats = independent_stats(direct)
    close('cancellation_identity', stats[:, 0], stats[:, 1]-stats[:, 3]+stats[:, 4], maxima, atol=1e-13)
    assert np.all(stats[:,3:] >= -1e-13)
    replica_values = rng.normal(size=(5,3,7,2))
    selections = rng.integers(7,size=(9,5,7))
    weights = multiplicities(selections,7)/7
    replica_pooled = sum((weights[:,pool] @ replica_values[pool].transpose(1,0,2).reshape(7,-1))
                         .reshape(9,3,2) for pool in range(5))/5
    direct_replica = np.stack([np.stack([replica_values[pool][:,selections[draw,pool]].mean(1)
                                        for pool in range(5)]).mean(0) for draw in range(9)])
    close('replica_pool_axes',replica_pooled,direct_replica,maxima,atol=1e-13)
    return {'passed': True, 'maximum_errors': maxima, 'study_data_accessed': False}


def regenerate(family, seed, shape):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) if family == 'gaussian' else rng.standard_t(5, size=shape)*np.sqrt(3/5)


def deterministic_control(eps, sigma, window, family, e):
    """Direct scalar root and sorted donor widths for both common-scale controls."""
    times = np.asarray(e['times'])
    ages = times[:,None]-np.arange(window,0,-1)
    errors = []
    dist = law(family)
    for scale in (np.ones_like(sigma), sigma):
        delta = np.log(scale/sigma)
        archive = delta[:,ages]
        current = np.exp(delta[:,times])
        qp = np.array([brentq(lambda q: (2*dist.cdf(q*np.exp(row))-1).mean()-.9,
                            1e-10, 100, xtol=1e-13) for row in archive[0]])[None]
        data = full_sort_metrics(np.abs(eps[:,ages])*np.exp(-archive), archive, current,
                                 qp, family, ages, e)[...,0]
        # Each donor's widths are identical for every anchor under these controls.
        other = (data.sum(0)[None]-data)/(len(eps)-1)
        errors.append(float(np.abs((data-other).mean(0)).max()))
    assert max(errors) < 1e-11
    return max(errors)


def sampled_scales(model, observations, saved, times, maxima):
    """Assemble past contexts explicitly, independently of evaluator predictors."""
    import torch
    targets = np.unique(np.r_[times, [0, 63, 64, 127, 255, 257, 383, 639]])
    anchors = [0, 63]
    contexts = []
    for anchor in anchors:
        for target in targets:
            context = np.zeros(64)
            past = observations[anchor, max(0,target-64):target]
            if len(past):
                context[-len(past):] = past
            contexts.append(context)
    with torch.no_grad():
        predicted = model(torch.tensor(np.stack(contexts), dtype=torch.float32)).exp().numpy()
    expected = saved[np.ix_(anchors, targets)].ravel()
    close('sampled_scale_replay', predicted, expected, maxima, atol=2e-6, rtol=1e-6)
    changed = observations.copy()
    changed[:,256:] = 1e4
    before = observations[anchors,192:256]
    after = changed[anchors,192:256]
    with torch.no_grad():
        a = model(torch.tensor(before, dtype=torch.float32)).exp().numpy()
        b = model(torch.tensor(after, dtype=torch.float32)).exp().numpy()
    close('future_perturbation', a, b, maxima, atol=0, rtol=0)
    close('onset_scale_replay', a, saved[anchors,256], maxima, atol=2e-6, rtol=1e-6)
    return len(contexts)


def audit_group(row, data, redraw, saved, scales, e, maxima, counts):
    eps, sigma = data['innovations'], data['true_scale']
    times = np.asarray(e['times'])
    ages = times[:,None]-np.arange(row.window,0,-1)
    delta = np.log(scales/sigma)
    archive, factor = delta[:,ages], np.exp(delta[:,times])
    metrics = saved['metrics']
    assert metrics.shape == (6,64,16,10)
    assert saved['methods'].tolist() == METHODS and saved['metric_names'].tolist() == METRICS
    np.testing.assert_array_equal(saved['times'], times)
    assert np.isfinite(metrics).all()
    assert (metrics[...,1] > 0).all()
    assert metrics[...,2].min() >= -1e-12 and metrics[...,2].max() <= 1+1e-12
    assert metrics[...,4].min() >= -1e-10
    assert (metrics[...,6]-metrics[...,5]**2).min() >= -1e-10
    dist = law(row.distribution)
    oracle = risk(dist.ppf(.95), row.distribution)
    assert metrics[...,0].min() >= oracle-1e-10
    base_half = metrics[0,...,1]
    qp = base_half/factor
    residual = (2*dist.cdf(qp[...,None]*np.exp(archive))-1).mean(-1)-.9
    close('population_root', residual, np.zeros_like(residual), maxima, atol=3e-12, rtol=0)
    population = np.zeros_like(metrics[0])
    population[...,0] = risk(base_half,row.distribution)
    population[...,1] = base_half
    population[...,2] = 2*dist.cdf(base_half)-1
    close('population_metrics', metrics[0], population, maxima)
    derivative = 20*(population[...,2]-.9)
    close('population_slope', saved['slope_P'], derivative, maxima)
    close('tangent_identity', metrics[...,3], derivative[None]*(metrics[...,1]-base_half[None]), maxima)
    close('convex_identity', metrics[...,0]-population[None,...,0], metrics[...,3]+metrics[...,4], maxima)
    actual = full_sort_metrics(np.abs(eps[:,ages])*np.exp(-archive),archive,factor,qp,row.distribution,ages,e)
    close('actual_full_sort', metrics[1], actual, maxima)
    counts['A_P_path_endpoints'] += 64*16
    primary = row.world == 'gaussian_up2' and row.context == 64 and row.window == 128
    anchors = np.arange(64) if primary else SECONDARY_ANCHORS
    total = np.zeros((len(anchors),16,10))
    split = np.zeros_like(total)
    replica_check = np.empty((64,16,10)) if primary else None
    for lo in range(0,64,8):
        hi = lo+8
        # Explicit transpose exposes [anchor, replica, endpoint, historical age].
        innovation = redraw[lo:hi][:,anchors][:,:,ages].transpose(1,0,2,3)
        scores = np.abs(innovation)*np.exp(-archive[anchors,None])
        value = full_sort_metrics(scores,archive[anchors,None],factor[anchors,None],
                                  qp[anchors,None],row.distribution,ages,e)
        total += value.sum(1)
        if lo < 32:
            split += value.sum(1)
        if primary:
            replica_check[lo:hi] = value.mean(0)
    close('D_full_sort', metrics[2,anchors], total/64, maxima)
    close('D_first_full_sort', metrics[4,anchors], split/32, maxima)
    close('D_last_full_sort', metrics[5,anchors], (total-split)/32, maxima)
    counts['D_anchor_replica_endpoints'] += len(anchors)*64*16
    counts['D_exhaustive_groups'] += int(primary)
    rep = saved['D_replica_endpoint']
    phase = saved['D_replica_phase']
    assert rep.shape == (64,16,10) and phase.shape == (64,4,10)
    assert np.isfinite(rep).all() and np.isfinite(phase).all()
    if primary:
        close('primary_D_replica_endpoint_full_sort', rep, replica_check, maxima)
    for method, section in [(2,slice(None)),(4,slice(0,32)),(5,slice(32,64))]:
        close('D_replica_mean', rep[section].mean(0), metrics[method].mean(0), maxima)
    close('D_replica_phase', phase, np.stack([rep[:,ids].mean(1) for ids in e['phases'].values()],axis=1), maxima)
    close('D_split_identity', metrics[2], (metrics[4]+metrics[5])/2, maxima)
    full_pairs = row.world in ('gaussian_flat','gaussian_up2') and row.window == 128
    anchors = np.arange(64) if full_pairs else SECONDARY_ANCHORS
    total = np.zeros((len(anchors),16,10))
    pair = np.empty((len(anchors),64,16))
    diagonal = np.empty_like(total)
    for lo in range(0,64,8):
        hi = lo+8
        scores = np.abs(eps[None,lo:hi,ages])*np.exp(-archive[anchors,None])
        value = full_sort_metrics(scores,archive[anchors,None],factor[anchors,None],
                                  qp[anchors,None],row.distribution,ages,e)
        pair[:,lo:hi] = value[...,0]
        total += value.sum(1)
        for ai, anchor in enumerate(anchors):
            if lo <= anchor < hi:
                diagonal[ai] = value[ai,anchor-lo]
    close('same_path_diagonal', diagonal, metrics[1,anchors], maxima)
    close('X_full_sort', (total-diagonal)/63, metrics[3,anchors], maxima)
    counts['X_anchor_donor_endpoints'] += len(anchors)*64*16
    counts['X_exhaustive_groups'] += int(full_pairs)
    assert ('pair_risk_phase' in saved) == full_pairs
    if full_pairs:
        phased = np.stack([pair[:,:,ids].mean(-1) for ids in e['phases'].values()])
        close('pair_phase_matrices', saved['pair_risk_phase'], phased, maxima)
    return metrics.mean(axis=1), rep[...,0]


def run(check_only=False):
    started = time.time()
    assert os.environ.get('SLURM_JOB_ID'), 'Bulk replay audits must run in an HPC Slurm job'
    assert check_only or not OUT.exists(), 'Verification record exists; --check-only is read-only'
    p = read_json(PROTOCOL)
    e = p['evaluation']
    em, am = read_json(EVAL/'manifest.json'), read_json(ANALYSIS/'manifest.json')
    assert p['frozen'] and em['status'] == am['status'] == 'complete'
    assert em['protocol_sha256'] == am['protocol_sha256'] == sha(PROTOCOL)
    assert em['completed_groups'] == 1920 and em['completed_pools'] == 16
    assert e['paths_per_pool'] == e['D_replicas'] == 64 and e['length'] == 640
    assert e['independent_pools'] == 16 and e['alpha'] == .1
    assert em['inputs'] == p['inputs']
    assert read_json(EVAL/'frozen_protocol.json') == p
    # The verifier itself, analyzer and evaluator must have been frozen before data.
    assert 'scripts/verify_long_window_validation.py' in em['sources']
    required = {'scripts/evaluate_long_window_validation.py','scripts/analyze_long_window_validation.py',
                'scripts/verify_long_window_validation.py'}
    assert required <= set(em['sources'])
    freeze_files = {**em['sources'], PROTOCOL.relative_to(ROOT).as_posix():sha(PROTOCOL)}
    for name,digest in freeze_files.items():
        at_freeze = subprocess.check_output(['git','show',f"{em['freeze_commit']}:{name}"],cwd=ROOT)
        assert hashlib.sha256(at_freeze).hexdigest() == digest == sha(ROOT/name), name
    current_verifier = subprocess.check_output(['git','show','HEAD:scripts/verify_long_window_validation.py'],cwd=ROOT)
    assert hashlib.sha256(current_verifier).hexdigest() == sha(__file__)
    verified_hashes = 0
    for base,mapping,complete in [(ROOT,p['inputs'],False),(ROOT,em['sources'],False),
                                  (ROOT,em['training_artifact_sha256'],False),
                                  (EVAL,em['outputs'],True),(ROOT,am['sources'],False),
                                  (ANALYSIS,am['outputs'],True)]:
        verified_hashes += check_hashes(base,mapping,complete)
    maxima = {}
    counts = {'A_P_path_endpoints':0,'D_anchor_replica_endpoints':0,'D_exhaustive_groups':0,
              'X_anchor_donor_endpoints':0,'X_exhaustive_groups':0,'sampled_issued_scales':0,
              'seed_arrays':0,'deterministic_control_endpoints':0}
    # Load only the frozen model definitions, never prediction/evaluation helpers.
    import torch
    from stage2_models import load_checkpoint
    torch.set_num_threads(4)
    training = read_json(ROOT/'outputs/neural_compatibility_training/manifest.json')
    assert training['status'] == 'complete' and training['completed_models'] == 12
    models, expected_fits = {}, []
    for panel in training['panels']:
        assert sha(ROOT/panel['manifest']) == panel['manifest_sha256']
        fitted = read_json(ROOT/panel['manifest'])
        verified_hashes += check_hashes(ROOT,panel['artifact_sha256'])
        for fit in fitted['models']:
            key = (panel['panel'],fit['architecture'],fit['model_seed'])
            assert sha(ROOT/fit['checkpoint']) == fit['checkpoint_sha256']
            model,payload = load_checkpoint(ROOT/fit['checkpoint'])
            assert payload['protocol_sha256'] == fitted['protocol_sha256']
            assert payload['architecture'] == key[1] and payload['model_seed'] == key[2]
            models[key] = model
            expected_fits.append(key)
    expected_fits.sort()
    assert len(models) == 12
    index = pd.read_csv(EVAL/'index.csv')
    expected = {(seeds['pool'],*fit,world['name'],64,window) for seeds in e['seeds']
                for fit in expected_fits for world in e['worlds'] for window in e['windows']}
    assert len(index) == len(expected) == 1920 and not index.duplicated(KEYS).any()
    assert set(index[KEYS].itertuples(index=False,name=None)) == expected
    assert len(set(index.file)) == 1920 and len(set(index.scale_file)) == 960
    assert {(m['panel'],m['architecture'],m['seed']) for m in em['models']} == set(expected_fits)
    seeds_flat = [seed for record in e['seeds'] for name,seed in record.items() if name != 'pool']
    assert len(set(seeds_flat)) == len(seeds_flat) == 64
    controls = pd.read_csv(EVAL/'negative_controls.csv')
    assert len(controls) == 16*5*2 and not controls.duplicated(['pool','world','window']).any()
    group_means, replica_risks = {}, {}
    onset_reference = {}
    for seedrow in e['seeds']:
        pool = seedrow['pool']
        folder = EVAL/'pools'/f'pool{pool:02d}'
        redraw, originals = {}, {}
        for family in ('gaussian','t5'):
            originals[family] = regenerate(family,seedrow[family],(64,640))
            with np.load(folder/f'redraw_{family}.npz',allow_pickle=False) as f:
                redraw[family] = f['innovations']
            np.testing.assert_array_equal(redraw[family],regenerate(family,seedrow['redraw_'+family],(64,64,640)))
            counts['seed_arrays'] += 2
        worlds = {}
        for world in e['worlds']:
            name, family = world['name'],world['distribution']
            with np.load(folder/f'test_{name}.npz',allow_pickle=False) as f:
                data = {k:f[k] for k in f.files}
            assert data['innovations'].shape == data['true_scale'].shape == data['observations'].shape == (64,640)
            np.testing.assert_array_equal(data['innovations'],originals[family])
            sigma = np.ones((64,640))
            sigma[:,e['onset']:e['onset']+e['duration']] = world['amplitude']
            np.testing.assert_array_equal(data['true_scale'],sigma)
            np.testing.assert_array_equal(data['observations'],sigma*originals[family])
            worlds[name] = data
            for window in e['windows']:
                error = deterministic_control(originals[family],sigma,window,family,e)
                maxima['deterministic_control_gap'] = max(maxima.get('deterministic_control_gap',0.),error)
                reported = controls[(controls.pool==pool)&(controls.world==name)&(controls.window==window)]
                assert len(reported) == 1 and 0 <= reported.iloc[0].deterministic_A_minus_X_maxabs < 1e-11
                counts['deterministic_control_endpoints'] += 2*len(e['times'])
        scale_cache = {}
        selected = index[index.pool == pool].sort_values(['panel','architecture','seed','world','window'])
        for row in selected.itertuples(index=False):
            key = tuple(getattr(row,k) for k in KEYS)
            assert row.distribution == next(w['distribution'] for w in e['worlds'] if w['name']==row.world)
            assert row.sha256 == em['outputs'][row.file]
            assert row.scale_sha256 == em['outputs'][row.scale_file]
            if row.scale_file not in scale_cache:
                with np.load(confined(EVAL,row.scale_file),allow_pickle=False) as f:
                    scales = f['scales'].astype(float)
                assert scales.shape == (64,640) and np.isfinite(scales).all() and np.all(scales>0)
                scale_cache[row.scale_file] = scales
                counts['sampled_issued_scales'] += sampled_scales(models[(row.panel,row.architecture,row.seed)],
                    worlds[row.world]['observations'],scales,np.array(e['times']),maxima)
            scales = scale_cache[row.scale_file]
            with np.load(confined(EVAL,row.file),allow_pickle=False) as f:
                saved = {k:f[k] for k in f.files}
            means,reps = audit_group(row,worlds[row.world],redraw[row.distribution],saved,scales,e,maxima,counts)
            group_means[key] = means
            if row.world=='gaussian_up2' and row.window==128:
                replica_risks[key] = reps
            if row.distribution == 'gaussian':
                onset = e['times'].index(e['onset'])
                raw = saved['metrics'][:4,:,onset,1]*worlds[row.world]['true_scale'][:,e['onset']][None]
                matching = (pool,row.panel,row.architecture,row.seed,row.window)
                if row.world == 'gaussian_flat':
                    onset_reference[matching] = raw
                else:
                    close('onset_raw_halfwidth',raw,onset_reference[matching],maxima,atol=1e-10)
        print(json.dumps({'verified_pool':pool,'groups':len(group_means)}),flush=True)
    inference_result = audit_analysis(p,am,expected_fits,group_means,replica_risks,maxima)
    # Detect source/provenance changes during a long replay audit.
    verified_hashes += check_hashes(ROOT,{**p['inputs'],**em['sources'],**em['training_artifact_sha256']})
    result = {'status':'complete','passed':True,'completed_at':datetime.now(timezone.utc).isoformat(),
              'script_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),
              'evaluation_manifest_sha256':sha(EVAL/'manifest.json'),
              'analysis_manifest_sha256':sha(ANALYSIS/'manifest.json'),
              'verified_hashes':verified_hashes,'verified_groups':len(group_means),
              'counts':counts,'maximum_errors':maxima,'independent_inference':inference_result,
              'scope':{'A_P':'All 1920 groups, all 64 paths and 16 endpoints',
                       'D':'All 64 replicas/paths/endpoints in 192 primary groups; all replicas at anchors 0,21,42,63 in 1728 secondary groups',
                       'X':'All 64 donors/anchors/endpoints in 384 flat/up2 W128 groups; all donors at anchors 0,21,42,63 in 1536 secondary groups',
                       'scales':'Independent reset-context predictions for paths 0 and 63 at all evaluation endpoints plus fixed boundary targets, every pool/fit/world',
                       'inference':'All pool/replica bootstrap draws and support gates; fixed existing fits only'},
              'notes':['Oracle lower bound applies to the risk metric; coverage/width and signed diagnostics have other units.',
                       'MC precision fractions divide by max(abs(full G point estimate), 1e-12).',
                       'Seed regeneration reproduces and compares existing frozen arrays; no new study pools or fits are created.',
                       'Secondary D/X full-sort replay is a declared deterministic subset; saved aggregate identities are checked everywhere.'],
              'self_check':self_check(),'new_study_data_generated':False,
              'slurm_job_id':os.environ['SLURM_JOB_ID'],'wall_seconds':time.time()-started}
    if not check_only:
        with OUT.open('x') as stream:
            stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


def audit_tables(endpoint, slopes, fits, conditions, e, maxima):
    """Reconstruct every descriptive table cell and require its complete grid."""
    scopes = {('all','all'):(list(range(16)),list(range(12)))}
    scopes.update({('panel',str(p)):(list(range(16)),[i for i,f in enumerate(fits) if f[0]==p]) for p in (1,2,3)})
    scopes.update({('fit',str(i)):(list(range(16)),[i]) for i in range(12)})
    scopes.update({('pool',str(i+1)):([i],list(range(12))) for i in range(16)})
    cached = {}
    for key,(pool_ids,fit_ids) in scopes.items():
        selected = endpoint[pool_ids][:,fit_ids]
        cached[key] = (selected.mean(0),slopes[pool_ids][:,fit_ids].mean(0))
    phases = e['phases']
    entry_targets = {f'entry_endpoint_{i+1}':[ti] for i,ti in enumerate(phases['entry'])}
    entry_targets['entry_post_onset3'] = phases['entry'][1:]
    contrasts = {'A-P':(1,0),'D-P':(2,0),'X-P':(3,0),'A-D':(1,2),'A-X':(1,3)}
    table_counts = {}
    for filename in ('phase_grid.csv','phase_metrics.csv','entry_targets.csv','decompositions.csv','event_minus_flat.csv'):
        table = pd.read_csv(ANALYSIS/filename,dtype=str,keep_default_na=False)
        seen = set()
        for row in table.itertuples(index=False):
            scope = (row.scope,row.scope_id)
            assert scope in scopes
            condition = (row.world,int(row.context),int(row.window))
            assert condition in conditions
            ci = conditions.index(condition)
            means,slope = cached[scope]
            if row.scope == 'panel':
                assert int(row.panel) == int(row.scope_id)
            if row.scope == 'fit':
                fit = fits[int(row.scope_id)]
                assert (int(row.panel),row.architecture,int(row.seed)) == fit
            if row.scope == 'pool':
                assert int(row.pool) == int(row.scope_id)
            ids = entry_targets[row.target] if filename=='entry_targets.csv' else phases[row.phase]
            source = means[:,ci]
            pooled = source[:,:,ids,0].transpose(0,2,1)
            statistics = independent_stats(pooled)
            # Sum distinct fit/time cells directly, independent of table builder axes.
            phase_metrics = sum(source[fi,:,ti] for fi in range(len(source)) for ti in ids)/(len(source)*len(ids))
            expectation = {}
            if filename in ('phase_grid.csv','entry_targets.csv'):
                expectation.update(dict(zip(STATS,statistics)))
                expectation['P_minus_A_signed'] = float((pooled[...,0]-pooled[...,1]).mean())
                expectation['D_minus_A_signed'] = float((pooled[...,2]-pooled[...,1]).mean())
                if filename=='phase_grid.csv':
                    expectation.update({f'D_minus_X_{name}':phase_metrics[2,k]-phase_metrics[3,k]
                                        for k,name in enumerate(METRICS)})
                    rowkey = (*scope,*condition,row.phase)
                else:
                    expectation['A_minus_X_signed'] = float((pooled[...,1]-pooled[...,3]).mean())
                    assert row.times == ';'.join(str(e['times'][ti]) for ti in ids)
                    rowkey = (*scope,*condition,row.target)
            elif filename=='phase_metrics.csv':
                assert row.method in METHODS
                expectation.update(dict(zip(METRICS,phase_metrics[METHODS.index(row.method)])))
                rowkey = (*scope,*condition,row.phase,row.method)
            elif filename=='decompositions.csv':
                left,right = contrasts[row.contrast]
                diff = phase_metrics[left]-phase_metrics[right]
                slope_average = float(slope[:,ci][:,ids].mean())
                product = slope_average*diff[1]
                expectation.update(risk_gap=diff[0],exact_tangent=diff[3],convex_gap=diff[4],
                                   slope_mean=slope_average,halfwidth_gap=diff[1],mean_product=diff[3],
                                   product_of_means=product,covariance=diff[3]-product)
                rowkey = (*scope,*condition,row.phase,row.contrast)
            else:
                assert condition[0] in ('gaussian_up1p5','gaussian_up2','gaussian_up3')
                flat = means[:,conditions.index(('gaussian_flat',condition[1],condition[2]))]
                c_event = float((source[:,1,ids,0]-source[:,3,ids,0]).mean())
                c_flat = float((flat[:,1,ids,0]-flat[:,3,ids,0]).mean())
                expectation.update(event_A_minus_X=c_event,flat_A_minus_X=c_flat,event_minus_flat_A_minus_X=c_event-c_flat)
                rowkey = (*scope,*condition,row.phase)
            assert rowkey not in seen, (filename,rowkey)
            seen.add(rowkey)
            for field,value in expectation.items():
                close('secondary_tables',float(getattr(row,field)),value,maxima,atol=1e-10)
        if filename=='phase_metrics.csv':
            expected = {(*s,*c,phase,method) for s in scopes for c in conditions for phase in phases for method in METHODS}
        elif filename=='decompositions.csv':
            expected = {(*s,*c,phase,contrast) for s in scopes for c in conditions for phase in phases for contrast in contrasts}
        elif filename=='entry_targets.csv':
            expected = {(*s,*c,target) for s in scopes for c in conditions for target in entry_targets}
        else:
            cs = [c for c in conditions if c[0] in ('gaussian_up1p5','gaussian_up2','gaussian_up3')] if filename=='event_minus_flat.csv' else conditions
            expected = {(*s,*c,phase) for s in scopes for c in cs for phase in phases}
        assert seen == expected, filename
        table_counts[filename] = len(seen)
    return table_counts


def audit_analysis(p,manifest,fits,means,replicas,maxima):
    e = p['evaluation']
    inference,precision = p['primary']['inference'],p['mc_precision']
    conditions = [(w['name'],64,window) for w in e['worlds'] for window in e['windows']]
    endpoint = np.empty((16,12,10,6,16,10))
    slopes = np.empty((16,12,10,16))
    primary = np.empty((16,12,4,6))
    redraw = np.empty((16,12,64,4))
    entry = e['phases']['entry']
    for pi in range(16):
        for fi,fit in enumerate(fits):
            for ci,condition in enumerate(conditions):
                key = (pi+1,*fit,*condition)
                endpoint[pi,fi,ci] = means[key]
                slopes[pi,fi,ci] = 20*(means[key][0,:,2]-.9)
                if condition == ('gaussian_up2',64,128):
                    primary[pi,fi] = means[key][:,entry,0].T
                    redraw[pi,fi] = replicas[key][:,entry]
    assert manifest['input_manifest_sha256'] == sha(EVAL/'manifest.json')
    assert manifest['input_outputs'] == read_json(EVAL/'manifest.json')['outputs']
    with np.load(ANALYSIS/'endpoints.npz',allow_pickle=False) as f:
        assert f['methods'].tolist() == METHODS and f['metric_names'].tolist() == METRICS
        assert list(zip(f['fit_panel'].tolist(),f['fit_architecture'].tolist(),f['fit_seed'].tolist())) == fits
        assert list(zip(f['condition_world'].tolist(),f['condition_context'].tolist(),f['condition_window'].tolist())) == conditions
        np.testing.assert_array_equal(f['times'],e['times'])
        close('analysis_endpoints',f['means'],endpoint,maxima,atol=1e-12)
        close('analysis_slopes',f['slope_P_mean'],slopes,maxima,atol=1e-12)
    with np.load(ANALYSIS/'inference.npz',allow_pickle=False) as f:
        saved = {k:f[k] for k in f.files}
    assert saved['statistic_names'].tolist() == STATS and saved['methods'].tolist() == METHODS
    assert list(zip(saved['fit_panel'].tolist(),saved['fit_architecture'].tolist(),saved['fit_seed'].tolist())) == fits
    assert saved['architecture_names'].tolist() == ['gru','tcn']
    np.testing.assert_array_equal(saved['times'],np.asarray(e['times'])[entry])
    close('primary_inputs',saved['primary_risk'],primary,maxima,atol=1e-12)
    close('primary_replica_inputs',saved['primary_D_replica_risk'],redraw,maxima,atol=1e-12)
    point = independent_stats(primary.mean(0))
    close('primary_point',saved['point'],point,maxima,atol=1e-12)
    architecture_points = {}
    for ai,architecture in enumerate(('gru','tcn')):
        selected = [i for i,fit in enumerate(fits) if fit[1]==architecture]
        architecture_points[architecture] = independent_stats(primary[:,selected].mean(0))
        close('architecture_point',saved['architecture_points'][ai],architecture_points[architecture],maxima,atol=1e-12)
    leave_one_out = np.stack([independent_stats((primary.sum(0)-primary[i])/15) for i in range(16)])
    close('leave_one_pool_out',saved['leave_one_pool_out'],leave_one_out,maxima,atol=1e-12)
    loo_table = pd.read_csv(ANALYSIS/'leave_one_pool_out.csv')
    np.testing.assert_array_equal(loo_table.omitted_pool,np.arange(1,17))
    close('leave_one_pool_out_table',loo_table[STATS].to_numpy(),leave_one_out,maxima,atol=1e-12)
    split_points = []
    for field,method in [('first32_point',4),('last32_point',5)]:
        local = primary.mean(0)
        local[...,2] = local[...,method]
        split_points.append(independent_stats(local))
        close('split_points',saved[field],split_points[-1],maxima,atol=1e-12)
    assert saved['bootstrap_counts'].shape == (12000,16) and saved['bootstrap_draws'].shape == (12000,5)
    assert saved['mc_counts'].shape == (4000,16,64) and saved['mc_draws'].shape == (4000,2)
    bootstrap = np.empty((inference['replicates'],5))
    rng = np.random.default_rng(inference['seed'])
    for lo in range(0,len(bootstrap),128):
        hi = min(lo+128,len(bootstrap))
        counts = multiplicities(rng.integers(16,size=(hi-lo,16),dtype=np.int64),16)
        np.testing.assert_array_equal(saved['bootstrap_counts'][lo:hi],counts)
        pooled = (counts.astype(float) @ primary.reshape(16,-1)/16).reshape(hi-lo,12,4,6)
        bootstrap[lo:hi] = independent_stats(pooled)
    close('bootstrap_draws',saved['bootstrap_draws'],bootstrap,maxima,atol=1e-12)
    close('bootstrap_cancellation_identity',bootstrap[:,0],bootstrap[:,1]-bootstrap[:,3]+bootstrap[:,4],maxima,atol=1e-12)
    mc = np.empty((precision['bootstrap_replicates'],2))
    rng = np.random.default_rng(precision['seed'])
    fixed = primary.mean(0)
    for lo in range(0,len(mc),128):
        hi = min(lo+128,len(mc))
        counts = multiplicities(rng.integers(64,size=(hi-lo,16,64),dtype=np.int64),64)
        np.testing.assert_array_equal(saved['mc_counts'][lo:hi],counts)
        dmean = np.zeros((hi-lo,12,4))
        # Separate matrix multiplication for each independent pool; shared fit weights.
        for pool in range(16):
            values = redraw[pool].transpose(1,0,2).reshape(64,-1)
            dmean += (counts[:,pool].astype(float) @ values).reshape(hi-lo,12,4)/(16*64)
        combined = np.broadcast_to(fixed,(hi-lo,*fixed.shape)).copy()
        combined[...,2] = dmean
        mc[lo:hi] = independent_stats(combined)[:,:2]
    close('replica_bootstrap_draws',saved['mc_draws'],mc,maxima,atol=1e-12)
    pool_c = (primary[...,1]-primary[...,3]).mean(axis=(1,2))
    close('pool_C',saved['pool_C'],pool_c,maxima,atol=1e-12)
    ci = np.quantile(bootstrap[:,:3],inference['quantiles'],axis=0).T
    se_c = float(np.sqrt(np.sum((pool_c-pool_c.mean())**2)/(16*15)))
    critical = float(t.ppf(inference['quantiles'][1],15))
    student = np.array([pool_c.mean()-critical*se_c,pool_c.mean()+critical*se_c])
    mc_se = np.sqrt(np.sum((mc-mc.mean(0))**2,axis=0)/(len(mc)-1))
    summary = read_json(ANALYSIS/'summary.json')
    assert summary['status']=='complete' and summary['protocol_sha256']==sha(PROTOCOL)
    assert summary['point'].keys() == dict(zip(STATS,point)).keys()
    for name,value in zip(STATS,point):
        close('summary_point',summary['point'][name],value,maxima,atol=1e-12)
    expected_hypotheses = {}
    for i,hypothesis in enumerate(p['primary']['hypotheses']):
        name = hypothesis['name']
        row = summary['primary'][name]
        negative = hypothesis['predicted_direction']=='negative'
        direction = -1 if negative else 1
        interval_pass = bool(ci[i,1]<0 if negative else ci[i,0]>0)
        architecture_pass = bool(all(direction*v[i]>0 for v in architecture_points.values()))
        assert row['statistic']==STATS[i] and row['predicted_direction']==hypothesis['predicted_direction']
        assert row['interval_direction_pass']==interval_pass
        assert row['architecture_direction_pass']==architecture_pass
        close('hypothesis_estimate',row['estimate'],point[i],maxima,atol=1e-12)
        close('hypothesis_interval',row['bootstrap_interval'],ci[i],maxima,atol=1e-12)
        assert set(row['architecture_estimates'])=={'gru','tcn'}
        for arch,v in architecture_points.items():
            close('hypothesis_architecture',row['architecture_estimates'][arch],v[i],maxima,atol=1e-12)
        if i < 2:
            denominator = max(abs(float(point[i])),precision['denominator_floor'])
            difference = abs(float(split_points[0][i]-split_points[1][i]))
            se_fraction,diff_fraction = float(mc_se[i]/denominator),difference/denominator
            se_pass = se_fraction<=precision['maximum_se_fraction']
            split_pass = diff_fraction<=precision['maximum_first32_last32_difference_fraction']
            precision_pass = se_pass and split_pass
            expected_precision = {'standard_error':float(mc_se[i]),'denominator':denominator,
                'se_fraction':se_fraction,'first32_estimate':float(split_points[0][i]),
                'last32_estimate':float(split_points[1][i]),'first32_last32_difference':difference,
                'difference_fraction':diff_fraction,'se_pass':se_pass,'split_pass':split_pass,'passed':precision_pass}
            assert set(row['mc_precision']) == set(expected_precision)
            for field,value in expected_precision.items():
                if isinstance(value,bool):
                    assert row['mc_precision'][field] == value
                else:
                    close('mc_precision_summary',row['mc_precision'][field],value,maxima,atol=1e-10)
            supported = interval_pass and architecture_pass and precision_pass
            status = 'precision-inconclusive' if not precision_pass else 'supported' if supported else 'unsupported'
        else:
            expected_student = {'df':15,'confidence':inference['individual_two_sided_confidence'],
                'standard_error':se_c,'critical_value':critical,'interval':student.tolist(),'upper_negative':bool(student[1]<0)}
            assert set(row['student_t']) == set(expected_student)
            for field,value in expected_student.items():
                if isinstance(value,bool):
                    assert row['student_t'][field]==value
                else:
                    close('student_t_summary',row['student_t'][field],value,maxima,atol=1e-12)
            supported = interval_pass and architecture_pass and student[1]<0
            status = 'supported' if supported else 'unsupported'
        assert row['supported']==bool(supported) and row['status']==status
        expected_hypotheses[name] = {'estimate':float(point[i]),'supported':bool(supported),'status':status}
    table_counts = audit_tables(endpoint,slopes,fits,conditions,e,maxima)
    assert summary['secondary_table_rows'] == table_counts
    interpretations = summary['interpretation']
    assert interpretations['aggregation_specific_reversal_supported'] == (expected_hypotheses['H1_phase_failure']['supported'] and expected_hypotheses['H2_endpoint_improvement']['supported'])
    assert interpretations['pairing_gap_supported']==expected_hypotheses['H3_same_pool_pairing']['supported']
    assert interpretations['stop_and_decide']==p['stop_and_decide']
    assert interpretations['qualification']==inference['qualification']
    assert interpretations['fit_uncertainty']==inference['fit_uncertainty']
    return {'hypotheses':expected_hypotheses,'pool_bootstrap_replicates':len(bootstrap),
            'replica_bootstrap_replicates':len(mc),'student_t_interval':student.tolist(),
            'secondary_table_rows':table_counts,'leave_one_pool_out_points':len(leave_one_out)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-check',action='store_true')
    mode.add_argument('--check-only',action='store_true',help='Replay all audits without writing or replacing verification JSON')
    args = parser.parse_args()
    print(json.dumps(self_check(),indent=2)) if args.self_check else run(args.check_only)
