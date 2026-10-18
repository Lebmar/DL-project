"""Frozen exogenous finite-sample mechanism study and endogenous boundary toy."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtri, stdtrit

import finite_mechanism_math as fm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/finite_mechanism_protocol.json"
OUT = ROOT / "outputs/finite_mechanism"
METRICS = ["coverage", "width", "interval_score"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_metrics(half, distribution, alpha=.1):
    return np.stack((fm.absolute_cdf(half, distribution), 2*half,
                     fm.expected_interval_score(half, distribution, alpha)), axis=-1)


def template_values(template, frequency, amplitude, times, period=256):
    phase = (frequency*np.asarray(times)) % period
    base = np.sin(2*np.pi*phase/period) if template == "sine" else np.where(phase < period//2, -1., 1.)
    return amplitude*base


def scored_windows(window_abs, archive_delta, current_delta, distribution, rules, alpha=.1):
    """Archives end strictly before each current time; rank coins integrated after scoring."""
    scores = window_abs*np.exp(-archive_delta)
    n = scores.shape[-1]
    mixtures = {rule: fm.rank_mixture(n, alpha, rule) for rule in rules}
    ranks = sorted({rank for terms in mixtures.values() for rank, _ in terms})
    scores.partition([k-1 for k in ranks], axis=-1)
    rank_metrics = {k: point_metrics(np.exp(current_delta)*scores[..., k-1], distribution, alpha) for k in ranks}
    return {rule: sum(probability*rank_metrics[k] for k, probability in terms) for rule, terms in mixtures.items()}


def resample_intervals(values, repetitions, seed, levels=(.95,.99375)):
    """values[case,path,metric]; preserve all phase/model/condition dependence within path."""
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, values.shape[0], values.shape[2]))
    for start in range(0, repetitions, 128):
        count = min(128, repetitions-start)
        indices = rng.integers(values.shape[1], size=(count, values.shape[1]))
        samples[start:start+count] = values[:, indices, :].mean(axis=2).transpose(1,0,2)
    return {str(level): np.quantile(samples, [(1-level)/2, 1-(1-level)/2], axis=0) for level in levels}


def converged_iid(current, archive, distribution, n, rule, p, quantile_cache):
    numerics = p["numerics"]
    previous = None
    for nodes in [numerics["quadrature_initial_nodes"], numerics["quadrature_check_nodes"], *numerics["quadrature_refinement_nodes"]]:
        diagnostics = {}
        result = fm.iid_mixture_metrics(current, archive, n=n, distribution=distribution, alpha=p["alpha"],
            rule=rule, quadrature_nodes=nodes, quantile_cache=quantile_cache, diagnostics=diagnostics)
        assert diagnostics["root_cdf_max_residual"] < numerics["root_tolerance"]
        if previous is not None:
            error = max(float(np.max(np.abs(result[k]-previous[k]))) for k in METRICS)
            if error <= numerics["maximum_phase_metric_difference"]:
                return result, {"quadrature_nodes": nodes, "maximum_phase_metric_change": error,
                                "root_cdf_max_residual": diagnostics["root_cdf_max_residual"]}
        previous = result
    raise RuntimeError(f"Quadrature failed convergence for {distribution}, n={n}, rule={rule}: {error}")


def self_check():
    z = np.array([[.1, -.5, 1., -2., .7, .9, 1.5, .3, 2.5, .8, -.4, 1.1]])
    times = np.array([10, 11])
    ages = np.arange(10, 0, -1)
    windows = np.abs(z[:, times[:,None]-ages])
    delta = .1*np.cos(np.arange(12))
    result = scored_windows(windows, delta[times[:,None]-ages], delta[times], "gaussian", ["ceiling"])["ceiling"]
    expected = []
    for tt in times:
        q = np.sort(np.abs(z[0,tt-10:tt])*np.exp(-delta[tt-10:tt]))[-1]
        expected.append(point_metrics(np.exp(delta[tt])*q, "gaussian"))
    np.testing.assert_allclose(result[0], expected, rtol=0, atol=1e-12)
    altered = z.copy(); altered[:,10:] = 1e6
    changed = scored_windows(np.abs(altered[:,times[:,None]-ages]), delta[times[:,None]-ages], delta[times], "gaussian", ["ceiling"])["ceiling"]
    np.testing.assert_allclose(result[:,0], changed[:,0], rtol=0, atol=0)
    for template in ["sine", "square"]:
        eval_times = np.arange(128,384,4)
        one = template_values(template,1,.2,eval_times)
        seventeen = template_values(template,17,.2,eval_times)
        np.testing.assert_array_equal(np.sort(one),np.sort(seventeen))
    return {"independent_sort": True, "strict_past_timing": True, "actual_phase_native_multiset": True}


def run():
    started = time.time()
    checks = self_check()
    checks["numerical_engine"] = fm.self_check()
    p = json.loads(PROTOCOL.read_text())
    assert not OUT.exists(), "Refusing to overwrite results"
    source_paths = [PROTOCOL,Path(__file__).resolve(),ROOT/"scripts/finite_mechanism_math.py"]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT).as_posix()
        assert hashlib.sha256(subprocess.check_output(["git","show",f"HEAD:{relative}"],cwd=ROOT)).hexdigest() == sha(path)
        source_hashes[relative] = sha(path)
    OUT.mkdir(parents=True)
    manifest = {"status":"running", "freeze_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "sources":source_hashes,"checks":checks,"protocol":p,"completed_groups":[],
        "started_at":datetime.now().astimezone().isoformat(),
        "versions":{"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__}}
    def save_manifest():
        (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    save_manifest()
    times = np.arange(**dict(start=p["evaluation_times"]["start"], stop=p["evaluation_times"]["stop_exclusive"], step=p["evaluation_times"]["stride"]))
    length = p["evaluation_times"]["stop_exclusive"]
    rng = np.random.default_rng(p["seeds"]["innovations"])
    u = rng.uniform(size=(p["paths"],length))
    independent_u = np.random.default_rng(p["seeds"]["independent_redraw"]).uniform(size=u.shape)
    innovations = {"gaussian":ndtri(u), "t5":np.sqrt(3/5)*stdtrit(5,u)}
    independent = {"gaussian":ndtri(independent_u), "t5":np.sqrt(3/5)*stdtrit(5,independent_u)}
    rows, phase_rows, numerical_rows, records, path_values = [], [], [], [], []
    boundary_records, boundary_values, control_records, control_values = [], [], [], []
    primary_arrays, primary_records = [], []
    maximum_constant_error = 0.
    oracle_cache = {}
    for distribution in p["distributions"]:
        z = float(fm.absolute_ppf(.9,distribution))
        oracle_risk = float(fm.expected_interval_score(z,distribution))
        coefficient = z*z*float(fm.absolute_pdf(z,distribution))/p["alpha"]
        second_coefficient = -.5*(1+z*float(fm.absolute_pdf_derivative(z,distribution))/float(fm.absolute_pdf(z,distribution)))
        for n in p["iid_bank_counts"]:
            cache = {}
            for rule in p["rules"]:
                oracle, diagnostics = converged_iid(np.zeros(1),np.zeros((1,1)),distribution,n,rule,p,cache)
                oracle_cache[(distribution,n,rule)] = oracle
                exact_coverage = sum(prob*k/(n+1) for k,prob in fm.rank_mixture(n,p["alpha"],rule))
                np.testing.assert_allclose(oracle["coverage"],exact_coverage,rtol=0,atol=1e-10)
                numerical_rows.append({"distribution":distribution,"template":"oracle","amplitude":0,"span":1,"frequency":0,"n":n,"rule":rule,**diagnostics})
        for span in p["history_spans"]:
            age_indices = times[:,None]-np.arange(span,0,-1)
            windows = np.abs(innovations[distribution][:,age_indices])
            zero = scored_windows(windows,np.zeros((len(times),span)),np.zeros(len(times)),distribution,p["rules"])
            constant = scored_windows(windows,np.full((len(times),span),.3),np.full(len(times),.3),distribution,p["rules"])
            for rule in p["rules"]:
                difference = float(np.max(np.abs(zero[rule]-constant[rule])))
                assert difference < 1e-11
                maximum_constant_error=max(maximum_constant_error,difference)
                control_records.append({"distribution":distribution,"span":span,"rule":rule,
                    "expected_coverage":float(oracle_cache[(distribution,span,rule)]["coverage"][0]),
                    "expected_interval_score":float(oracle_cache[(distribution,span,rule)]["interval_score"][0])})
                control_values.append(zero[rule].mean(axis=1))
            for template in p["templates"]:
                for amplitude in p["amplitudes"]:
                    pair_paths, pair_predictions, native_by_frequency = {}, {}, []
                    for frequency in p["frequencies"]:
                        case={"distribution":distribution,"template":template,"amplitude":amplitude,"span":span,"frequency":frequency}
                        current=template_values(template,frequency,amplitude,times,p["period"])
                        archive=template_values(template,frequency,amplitude,age_indices,p["period"])
                        native=point_metrics(z*np.exp(current),distribution)
                        native_by_frequency.append(native.mean(axis=0))
                        mu, var = archive.mean(axis=1), archive.var(axis=1)
                        first=current-mu
                        second=first+second_coefficient*var
                        population=fm.population_metrics(current,archive,distribution,p["alpha"])
                        residual=np.max(np.abs(fm.mixture_cdf(population["quantile_mean"],archive,distribution)-.9))
                        assert residual < p["numerics"]["root_tolerance"]
                        assert population["interval_score"].min() >= oracle_risk-1e-10
                        for source, values in [("native",native),("population",np.stack([population[m] for m in METRICS],axis=-1)),
                            ("first_order_population",point_metrics(z*np.exp(first),distribution)),
                            ("second_order_population",point_metrics(z*np.exp(second),distribution))]:
                            rows.append({**case,"source":source,"n":0,"rule":"population",**dict(zip(METRICS,values.mean(axis=0)))})
                        for phase_index,tt in enumerate(times):
                            phase_rows.append({**case,"time":int(tt),"current_delta":current[phase_index],"archive_mean":mu[phase_index],
                                "archive_variance":var[phase_index],"first_order_log_error":first[phase_index],"second_order_log_error":second[phase_index],
                                "exact_population_log_error":population["log_relative_halfwidth_mean"][phase_index]})
                        cache={}
                        for n in p["iid_bank_counts"]:
                            for rule in p["rules"]:
                                iid,diag=converged_iid(current,archive,distribution,n,rule,p,cache)
                                assert iid["interval_score"].min() >= oracle_risk-1e-10
                                rows.append({**case,"source":"iid_mixture","n":n,"rule":rule,**{m:float(iid[m].mean()) for m in METRICS}})
                                numerical_rows.append({**case,"n":n,"rule":rule,**diag})
                        observed=scored_windows(windows,archive,current,distribution,p["rules"])
                        for rule,values in observed.items():
                            mean_path=values.mean(axis=1)
                            records.append({**case,"rule":rule})
                            path_values.append(mean_path)
                            rows.append({**case,"source":"contiguous","n":span,"rule":rule,**dict(zip(METRICS,mean_path.mean(axis=0)))})
                            prediction=float(oracle_cache[(distribution,span,rule)]["interval_score"][0]+coefficient*np.mean(first**2))
                            rows.append({**case,"source":"first_order_plus_noise","n":span,"rule":rule,"interval_score":prediction})
                            if rule==p["primary"]["rule"]:
                                pair_paths[frequency]=mean_path
                                pair_predictions[frequency]=prediction
                        print(json.dumps({"completed_case":case}),flush=True)
                    np.testing.assert_allclose(native_by_frequency[0],native_by_frequency[1],rtol=0,atol=1e-12)
                    if amplitude==p["primary"]["amplitude"]:
                        primary_records.append({"distribution":distribution,"template":template,"amplitude":amplitude,"span":span,
                            "predicted_is_contrast":pair_predictions[17]-pair_predictions[1]})
                        primary_arrays.append(pair_paths[17]-pair_paths[1])
            # Endogenous lag-1 stress: same issued scales, independent archived innovations.
            delta=np.zeros_like(innovations[distribution])
            delta[:,1:]=p["endogenous_boundary"]["amplitude"]*np.sign(np.abs(innovations[distribution][:,:-1])-float(fm.absolute_ppf(.5,distribution)))
            actual=scored_windows(windows,delta[:,age_indices],delta[:,times],distribution,["randomized"])["randomized"].mean(axis=1)
            decoupled=scored_windows(np.abs(independent[distribution][:,age_indices]),delta[:,age_indices],delta[:,times],distribution,["randomized"])["randomized"].mean(axis=1)
            boundary_records.append({"distribution":distribution,"span":span,"contrast":"actual_minus_independent_redraw",
                **{f"actual_{m}":float(actual[:,mi].mean()) for mi,m in enumerate(METRICS)},
                **{f"redraw_{m}":float(decoupled[:,mi].mean()) for mi,m in enumerate(METRICS)}})
            boundary_values.append(actual-decoupled)
            manifest["completed_groups"].append({"distribution":distribution,"span":span})
            save_manifest()
    primary=np.stack(primary_arrays)
    assert primary.shape==(8,p["paths"],3)
    intervals=resample_intervals(primary,p["primary"]["bootstrap_replicates"],p["seeds"]["bootstrap"])
    gate=p["primary"]["neural_bridge_gate"]
    for i,record in enumerate(primary_records):
        for mi,m in enumerate(METRICS):
            record[f"observed_{m}_contrast"]=float(primary[i,:,mi].mean())
            for level,values in intervals.items():
                record[f"{m}_low_{level}"]=float(values[0,i,mi])
                record[f"{m}_high_{level}"]=float(values[1,i,mi])
        predicted,actual=record["predicted_is_contrast"],record["observed_interval_score_contrast"]
        record["indeterminate_prediction"]=abs(predicted)<gate["predicted_absolute_contrast_indeterminate_below"]
        record["correct_direction"]=bool(not record["indeterminate_prediction"] and predicted*actual>0)
        low,high=record["interval_score_low_0.99375"],record["interval_score_high_0.99375"]
        record["supported_direction"]=bool(record["correct_direction"] and ((predicted>0 and low>0) or (predicted<0 and high<0)))
        record["relative_contrast_error"]=float(abs(predicted-actual)/max(abs(actual),gate["relative_error_denominator_floor"]))
    frame=pd.DataFrame(primary_records)
    by_distribution=frame.groupby("distribution").supported_direction.sum().astype(int).to_dict()
    by_span={str(k):int(v) for k,v in frame.groupby("span").supported_direction.sum().items()}
    verdict={"correct_directions":int(frame.correct_direction.sum()),"supported_directions":int(frame.supported_direction.sum()),
        "supported_by_distribution":by_distribution,"supported_by_span":by_span,"median_relative_contrast_error_all8":float(frame.relative_contrast_error.median())}
    verdict["passed"]=bool(verdict["correct_directions"]>=gate["minimum_correct_directions"] and
        verdict["supported_directions"]>=gate["minimum_correct_directions_with_simultaneous_interval_excluding_zero"] and
        min(by_distribution.values())>=gate["minimum_supported_directions_per_distribution"] and
        min(by_span.values())>=gate["minimum_supported_directions_per_span"] and
        verdict["median_relative_contrast_error_all8"]<=gate["maximum_median_relative_contrast_error"])
    for record_set,values in [(boundary_records,np.stack(boundary_values)),(control_records,np.stack(control_values))]:
        ci=resample_intervals(values,p["primary"]["bootstrap_replicates"],p["seeds"]["bootstrap"],levels=(.95,))["0.95"]
        for i,record in enumerate(record_set):
            for mi,m in enumerate(METRICS):
                record[m]=float(values[i,:,mi].mean());record[m+"_low95"]=float(ci[0,i,mi]);record[m+"_high95"]=float(ci[1,i,mi])
    for name,data in [("summary",rows),("population_phases",phase_rows),("quadrature_checks",numerical_rows),
        ("primary_contrasts",primary_records),("endogenous_boundary",boundary_records),("oracle_controls",control_records),("path_index",records)]:
        pd.DataFrame(data).to_csv(OUT/(name+".csv"),index=False)
    np.savez_compressed(OUT/"path_metrics.npz",values=np.stack(path_values),primary_contrasts=primary,
        endogenous_contrasts=np.stack(boundary_values),oracle_controls=np.stack(control_values),metrics=np.array(METRICS))
    manifest.update(status="complete",wall_seconds=time.time()-started,neural_bridge_gate=verdict,
        numerical_cases=len(numerical_rows),maximum_quadrature_metric_change=max(x["maximum_phase_metric_change"] for x in numerical_rows),
        maximum_constant_error=maximum_constant_error,outputs={f.name:sha(f) for f in sorted(OUT.iterdir()) if f.name!="manifest.json"})
    save_manifest()
    print(json.dumps({"status":"complete","wall_seconds":manifest["wall_seconds"],"gate":verdict},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    print(json.dumps(self_check())) if args.self_check else run()
