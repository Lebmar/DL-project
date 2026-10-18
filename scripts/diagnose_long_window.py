"""Exploratory diagnosis on already-viewed neural trajectories; no new data/fits.

Exact convex risk geometry and exhaustive cross-path innovation donors distinguish
the observed A-D gap from effects of sharing one finite innovation sample.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, t

import finite_mechanism_math as fm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/long_window_diagnostic_protocol.json"
SOURCE = ROOT / "outputs/neural_compatibility_evaluation"
OUT = ROOT / "outputs/long_window_diagnostic"
KEYS = ["panel", "architecture", "seed", "world", "context", "window"]
METHODS = ["P", "A", "D", "X"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def risk_geometry(half, family, alpha=.1):
    """Exact first/second derivatives of expected IS with respect to halfwidth."""
    slope = 2/alpha*(fm.absolute_cdf(half, family)-(1-alpha))
    curvature = 2/alpha*fm.absolute_pdf(half, family)
    return slope, curvature


def off_diagonal(matrix):
    n = matrix.shape[-1]
    return (matrix.sum(axis=(-2, -1))-np.trace(matrix, axis1=-2, axis2=-1))/(n*(n-1))


def jackknife_gap(matrix):
    """A-X with deletion of the same independent path in both donor/anchor axes."""
    n = matrix.shape[-1]
    diag = np.diag(matrix)
    total = matrix.sum()-diag.sum()
    estimate = diag.mean()-total/(n*(n-1))
    remaining = total-(matrix.sum(axis=0)-diag)-(matrix.sum(axis=1)-diag)
    loo = (diag.sum()-diag)/(n-1)-remaining/((n-1)*(n-2))
    se = np.sqrt((n-1)/n*np.square(loo-loo.mean()).sum())
    return float(estimate), float(se), loo


def donor_bank(innovations, delta, current, qp, times, window, family, context, onset, duration):
    """Return ordered-pair diagnostics [anchor, donor, time], incl. original diagonal."""
    n = len(innovations)
    ages = times[:, None]-np.arange(window, 0, -1)
    archive = delta[:, ages]
    original_eps = np.abs(innovations[:, ages])
    z = float(fm.absolute_ppf(.9, family))
    shape = (n, n, len(times))
    pair = {key: np.zeros(shape) for key in ["risk", "half", "coverage", "eta", "eta2", "cdf_deficit", "recent_deficit", "event_deficit"]}
    expected_tail = 1-fm.absolute_cdf(qp[..., None]*np.exp(archive), family)
    recent = np.arange(window)[None, :] >= max(0, window-context)
    event = (ages >= onset) & (ages < onset+duration)
    for start in range(0, n, 8):
        stop = min(n, start+8)
        # One observed donor chronology is reused at every overlapping endpoint.
        scores = original_eps[None, start:stop]*np.exp(-archive[:, None])
        residual = (scores > qp[:, None, :, None])-expected_tail[:, None]
        pair["cdf_deficit"][:, start:stop] = residual.mean(axis=-1)
        pair["recent_deficit"][:, start:stop] = (residual*recent).mean(axis=-1)
        pair["event_deficit"][:, start:stop] = (residual*event).mean(axis=-1)
        scores.partition([k-1 for k, _ in fm.rank_mixture(window, .1, "randomized")], axis=-1)
        for rank, weight in fm.rank_mixture(window, .1, "randomized"):
            q = scores[..., rank-1]
            half = np.exp(current[:, None])*q
            eta = np.log(q/qp[:, None])
            pair["risk"][:, start:stop] += weight*fm.expected_interval_score(half, family)
            pair["half"][:, start:stop] += weight*half
            pair["coverage"][:, start:stop] += weight*fm.absolute_cdf(half, family)
            pair["eta"][:, start:stop] += weight*eta
            pair["eta2"][:, start:stop] += weight*eta*eta
    return pair


def self_check():
    max_derivative = max_integral = 0.
    for family in ["gaussian", "t5"]:
        distribution = norm if family == "gaussian" else t(5, scale=np.sqrt(3/5))
        for base in [.4, 1., 2., 3.]:
            slope, curve = risk_geometry(base, family)
            h = 1e-5
            finite = (fm.expected_interval_score(base+h, family)-fm.expected_interval_score(base-h, family))/(2*h)
            max_derivative = max(max_derivative, abs(float(finite-slope)))
            np.testing.assert_allclose(curve, 40*distribution.pdf(base), rtol=1e-12)
            for other in [.25, .8, 1.8, 3.5]:
                bregman = fm.expected_interval_score(other, family)-fm.expected_interval_score(base, family)-slope*(other-base)
                integral = (other-base)**2*quad(lambda u: (1-u)*40*distribution.pdf(base+u*(other-base)), 0, 1)[0]
                assert bregman >= -1e-12
                max_integral = max(max_integral, abs(float(bregman-integral)))
    assert max_derivative < 1e-7 and max_integral < 1e-10
    rng = np.random.default_rng(2026101809)
    matrix = rng.normal(size=(7, 7))
    estimate, se, loo = jackknife_gap(matrix)
    direct = []
    for i in range(7):
        keep = np.delete(np.arange(7), i)
        sub = matrix[np.ix_(keep, keep)]
        direct.append(np.diag(sub).mean()-off_diagonal(sub))
    np.testing.assert_allclose(loo, direct, atol=1e-15)
    eps = rng.normal(size=(4, 180))
    delta = np.broadcast_to(np.linspace(-.2, .2, 180), eps.shape)
    times = np.array([129, 143, 161, 170])
    archive = delta[:, times[:, None]-np.arange(32, 0, -1)]
    qp = fm.population_mixture_quantile(archive.reshape(-1, 32)).reshape(4, 4)
    pairs = donor_bank(eps, delta, delta[:, times], qp, times, 32, "gaussian", 16, 140, 20)
    risk = pairs["risk"].mean(axis=-1)
    np.testing.assert_allclose(np.diag(risk).mean(), off_diagonal(risk), atol=1e-14)
    # Independent scalar sorting checks axes and candidate-rank mixing.
    for p in range(4):
        for donor in range(4):
            for ti, tt in enumerate(times):
                bank = np.sort(np.abs(eps[donor, tt-32:tt])*np.exp(-delta[p, tt-32:tt]))
                expected = sum(w*fm.expected_interval_score(np.exp(delta[p, tt])*bank[k-1]) for k, w in fm.rank_mixture(32, .1, "randomized"))
                np.testing.assert_allclose(pairs["risk"][p, donor, ti], expected, atol=1e-13)
    return {"maximum_derivative_error": max_derivative, "maximum_exact_convex_integral_error": max_integral,
            "jackknife_both_axes": True, "common_deterministic_scale_zero_pairing_gap": True,
            "independent_scalar_pair_sort": True}


def run():
    started = time.time()
    checks = self_check()
    p = json.loads(PROTOCOL.read_text())
    assert p["frozen"] and not OUT.exists()
    for file in [PROTOCOL, Path(__file__).resolve()]:
        original = subprocess.check_output(["git", "show", f"HEAD:{file.relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == sha(file)
    for name, digest in p["inputs"].items():
        assert sha(ROOT/name) == digest, name
    em = json.loads((SOURCE/"manifest.json").read_text())
    assert em["status"] == "complete"
    for name, digest in em["outputs"].items():
        assert sha(SOURCE/name) == digest, name
    original = json.loads((ROOT/"configs/neural_compatibility_protocol.json").read_text())
    e = original["evaluation"]
    times = np.array(e["times"])
    index = pd.read_csv(SOURCE/"index.csv")
    assert len(index) == 192 and not index.duplicated(KEYS).any()
    phase_rows, endpoint_rows, decomp_rows, jackknife_rows, pair_risks, group_rows = [], [], [], [], [], []
    world_cache, scale_cache = {}, {}
    max_diagonal = max_identity = 0.
    minimum_convex = float("inf")
    OUT.mkdir(parents=True)
    for group_id, row in enumerate(index.itertuples(index=False)):
        meta = {k:getattr(row, k) for k in KEYS}
        family = row.distribution
        if row.world not in world_cache:
            with np.load(SOURCE/f"test_{row.world}.npz", allow_pickle=False) as f:
                world_cache[row.world] = (f["innovations"], f["true_scale"])
        innovations, sigma = world_cache[row.world]
        if row.scale_file not in scale_cache:
            with np.load(ROOT/row.scale_file, allow_pickle=False) as f:
                scale_cache[row.scale_file] = f["scales"].astype(float)
        delta = np.log(scale_cache[row.scale_file]/sigma)
        with np.load(SOURCE/row.file, allow_pickle=False) as f:
            metrics = f["metrics"]
            moments = f["moments"]
        n = len(delta)
        current = delta[:, times]
        z = float(fm.absolute_ppf(.9, family))
        xi_p = moments[0, ..., 0]
        qp = z*np.exp(xi_p-current)
        pairs = donor_bank(innovations, delta, current, qp, times, row.window, family, row.context, e["onset"], e["duration"])
        diag = {key:value[np.arange(n), np.arange(n)] for key, value in pairs.items()}
        x = {key:(value.sum(axis=1)-diag[key])/(n-1) for key, value in pairs.items()}
        for key, reference in [("risk", metrics[3, ..., 2]), ("half", metrics[3, ..., 1]/2), ("eta", moments[1, ..., 2]), ("eta2", moments[1, ..., 3])]:
            error = float(np.max(np.abs(diag[key]-reference)))
            max_diagonal = max(max_diagonal, error)
            np.testing.assert_allclose(diag[key], reference, rtol=1e-11, atol=1e-11)
        risks = np.stack([metrics[mi, ..., 2] for mi in [2,3,4]]+[x["risk"]])
        halves = np.stack([metrics[mi, ..., 1]/2 for mi in [2,3,4]]+[x["half"]])
        coverages = np.stack([metrics[mi, ..., 0] for mi in [2,3,4]]+[x["coverage"]])
        eta = np.stack([moments[mi, ..., 2] for mi in [0,1,2]]+[x["eta"]])
        eta2 = np.stack([moments[mi, ..., 3] for mi in [0,1,2]]+[x["eta2"]])
        slope, curve = risk_geometry(halves[0], family)
        tangent = slope[None]*(halves-halves[0])
        convex = risks-risks[0]-tangent
        minimum_convex = min(minimum_convex, float(convex.min()))
        assert convex.min() > -1e-9
        # Log-width Taylor is a separate descriptive approximation, not the exact convex split.
        log_first = slope*halves[0]
        log_second = log_first+curve*halves[0]**2
        log_linear = log_first[None]*eta
        log_quadratic = .5*log_second[None]*eta2
        log_remainder = risks-risks[0]-log_linear-log_quadratic
        phase_pair = np.stack([pairs["risk"][:, :, ids].mean(axis=-1) for ids in e["phases"].values()])
        pair_risks.append(phase_pair)
        group_rows.append({"group_id":group_id, **meta})
        for phase, ids in e["phases"].items():
            for mi, name in enumerate(METHODS):
                phase_rows.append({**meta, "phase":phase, "method":name,
                    "risk":float(risks[mi][:, ids].mean()), "halfwidth":float(halves[mi][:, ids].mean()),
                    "coverage":float(coverages[mi][:, ids].mean()), "eta_mean":float(eta[mi][:, ids].mean()),
                    "eta_second":float(eta2[mi][:, ids].mean()), "exact_tangent":float(tangent[mi][:, ids].mean()),
                    "convex_remainder":float(convex[mi][:, ids].mean()),
                    "log_linear":float(log_linear[mi][:, ids].mean()), "log_quadratic":float(log_quadratic[mi][:, ids].mean()),
                    "log_remainder":float(log_remainder[mi][:, ids].mean())})
            for comparator, ci in [("D",2),("X",3)]:
                gap = (risks[1]-risks[ci])[:, ids]
                linear = (tangent[1]-tangent[ci])[:, ids]
                rem = (convex[1]-convex[ci])[:, ids]
                diff_half = (halves[1]-halves[ci])[:, ids]
                local_slope = slope[:, ids]
                product = local_slope.mean()*diff_half.mean()
                err = float(np.max(np.abs(gap-linear-rem)))
                max_identity = max(max_identity, err)
                assert err < 1e-10
                decomp_rows.append({**meta,"phase":phase,"contrast":"A-"+comparator,
                    "risk_gap":float(gap.mean()),"exact_linear":float(linear.mean()),"convex_gap":float(rem.mean()),
                    "slope_mean":float(local_slope.mean()),"halfwidth_gap":float(diff_half.mean()),
                    "linear_product_of_means":float(product),"linear_covariance":float(linear.mean()-product),
                    "P_underwidth_fraction":float((halves[0][:,ids]<z).mean()),
                    "log_linear":float((log_linear[1]-log_linear[ci])[:,ids].mean()),
                    "log_quadratic":float((log_quadratic[1]-log_quadratic[ci])[:,ids].mean()),
                    "log_remainder":float((log_remainder[1]-log_remainder[ci])[:,ids].mean())})
        ages = times[:, None]-np.arange(row.window, 0, -1)
        archive = delta[:, ages]
        qscale = qp[..., None]*np.exp(archive)
        log_density = (qscale*fm.absolute_pdf(qscale, family)).mean(axis=-1)
        for ti, tt in enumerate(times):
            record = {**meta,"time":int(tt),"offset_from_onset":int(tt-e["onset"]),
                "event_fraction_in_archive":float(((ages[ti]>=e["onset"])&(ages[ti]<e["onset"]+e["duration"])).mean())}
            for mi, name in enumerate(METHODS):
                record.update({f"{name}_risk":float(risks[mi,:,ti].mean()),f"{name}_half":float(halves[mi,:,ti].mean()),
                    f"{name}_eta":float(eta[mi,:,ti].mean()),f"{name}_tangent":float(tangent[mi,:,ti].mean()),
                    f"{name}_convex":float(convex[mi,:,ti].mean())})
            for name, bank in [("A",diag),("X",x)]:
                for key in ["cdf_deficit","recent_deficit","event_deficit"]:
                    record[f"{name}_{key}"] = float(bank[key][:,ti].mean())
                record[f"{name}_cdf_linear_eta"] = float((bank["cdf_deficit"][:,ti]/log_density[:,ti]).mean())
            endpoint_rows.append(record)
        if (group_id+1)%32 == 0:
            print(json.dumps({"complete_groups":group_id+1}), flush=True)
    pair_risks = np.stack(pair_risks)
    groups = pd.DataFrame(group_rows)
    # All 12 fits use the same path pool. Delete a path jointly across every fit.
    for condition, selected in groups.groupby(["world","context","window"], sort=True):
        joint = pair_risks[selected.index].mean(axis=0)
        for pi, phase in enumerate(e["phases"]):
            gap, se, loo = jackknife_gap(joint[pi])
            jackknife_rows.append({**dict(zip(["world","context","window"],condition)),"phase":phase,
                "A_minus_X":gap,"jackknife_se":se,"descriptive_low":gap-1.959963984540054*se,
                "descriptive_high":gap+1.959963984540054*se})
    for name, records in [("cells",phase_rows),("risk_decomposition",decomp_rows),("endpoints",endpoint_rows),
                          ("pairing_sensitivity",jackknife_rows),("groups",group_rows)]:
        pd.DataFrame(records).to_csv(OUT/(name+".csv"), index=False)
    np.savez_compressed(OUT/"pair_risk_by_phase.npz", risk=pair_risks, phases=np.array(list(e["phases"])))
    checks.update(maximum_diagonal_replay_error=max_diagonal, maximum_exact_decomposition_error=max_identity,
                  minimum_convex_remainder=minimum_convex, groups=192, saved_pair_risk_shape=list(pair_risks.shape))
    for name, digest in p["inputs"].items():
        assert sha(ROOT/name) == digest
    manifest = {"status":"complete","scope":"exploratory on previously viewed paths; no fresh innovations or training",
        "completed_at":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(PROTOCOL),"source_sha256":sha(__file__),
        "freeze_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "inputs":p["inputs"],"checks":checks,"wall_seconds":time.time()-started,
        "outputs":{f.name:sha(f) for f in sorted(OUT.iterdir())}}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    print(json.dumps(manifest["checks"],indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(self_check(),indent=2)) if args.self_check else run()
