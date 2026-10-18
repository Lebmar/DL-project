# Finite-rank sensitivity protocol — 18 October 2026

This bounded diagnostic addresses a stated limitation of the completed Stage 3 study. It reuses all 30 saved neural fits and the already-viewed Gaussian/t5 D128 A3 paths. It is **post-hoc sensitivity**, not a new independent replication or confirmatory test. No training, model selection, checkpoint update, or choice based on the new outcome is allowed. The original Stage 3 results and frozen sources remain unchanged.

## Question and fixed scope

The original rolling intervals use the integer rank `ceil((W+1)*.9)`: k30 for W32 and k117 for W128. Under continuous exchangeable scores, these correspond to ideal coverage 30/33 and 117/129. Thus the window contrast changes finite-rank level as well as history length and sample count. Event-minus-control pairing does not prove that this rank effect vanishes.

We ask how the saved current/archive interactions change if the quantile rule has common **ideal rank level .90**. Use all three training panels, the same five shared seeds per panel, both architectures, and both noise families. Duration128 and amplitude3 are fixed; entry `[384,416)` and recovery64 `[512,576)` are the only phases. All128 original paired paths per family remain, with original validation warmups and frozen scale forecasts. Existing path hashes must reproduce exactly. Both decomposition routes and all five metrics remain visible.

## Exact expected randomized-rank metrics

For W32, choose rank29 with probability .30 or rank30 with probability .70. For W128, choose rank116 with probability .90 or rank117 with probability .10. These weights are fixed by `(W+1)*.9`, not data or outcomes. The same original k30/k117 rule is evaluated as a reproduction control.

For each interval, compute the metric at each of the two possible widths, and average those metric values with the specified probabilities. In particular, expected interval score is `(1-p)*IS(h_lower)+p*IS(h_upper)`. Evaluating `IS((1-p)*h_lower+p*h_upper)` would define a different deterministic interval and is incorrect for this experiment. No rank Monte Carlo is needed; linear expectations also commute with the phase means and factorial differences. Positive-part oracle coverage deficit and excess width are averaged **after** their candidate-level positive-part operation, consistent with expected performance of randomized intervals.

This integrates random-rank uncertainty and estimates expected metrics of the random procedure, rather than the variability of an actually realized sequence of rank coins. Consequently bootstrap intervals condition on this integration. If one implemented realized intervals, coin coupling could affect joint finite-sample variability; this study makes no claim about that variance.

Ideal coverage .90 follows from the mean of the two uniform-rank coverage probabilities only under continuity and exchangeability. The neural scores here are dependent and nonstationary, so this is not a coverage theorem for the simulation or market data. The alternative rule lowers both original ideal levels; differences are not a unique causal 'rank correction' or a percentage of effects explained. Number of scores, sample-quantile variability, and history span still differ between windows.

## Outcomes and uncertainty

For each rank rule, compute the original event-minus-control D and J definitions, all four route components, and third-order interaction. Retain both architectures separately, GRU-minus-TCN, and paired `randomized90-minus-ceiling` changes. Main readout is Gaussian interval-score J architecture contrast for entry and recovery64; t5 and every metric/component are retained as descriptive support.

Use 4000 common bootstrap draws with seed 2026101801. Fixed-fit/path intervals average the 15 fixed fitted pairs, then resample 128 paths shared across all rules, architectures and components. The crossed sensitivity additionally resamples three panels and five seed indices; the same seed draw is broadcast across sampled panels, preserving shared initialization/sampling identities. Rule and architecture differences are formed inside each common draw. Save all panel/individual-fit and leave-one-panel-out estimates.

Reference intervals are 97.5%, a two-phase approximation within an individual rule or paired-change series. They do not protect jointly across all rank rules, metrics, components or the overall post-hoc exercise. Other intervals are 95% descriptive/unadjusted. No selected sign, significance, percent attenuation, or model winner defines success. The deliverable is an honest account of whether and how the previously reported result changes. Three panels cannot support a reliable training-population claim.

## Audits, saved artifacts, and execution

Before execution, commit this document, the JSON protocol and both new scripts. Both scripts refuse a pre-existing output directory. Verify all original protocol/source/manifest/checkpoint hashes; no frozen source or saved primary result is modified.

Pure synthetic checks cover adjacent-order-statistic indexing, current/future exclusion, exact ideal rank coverage, nonlinear-metric expectation, both decompositions, and independent scalar bootstrap loops. During evaluation, upper candidate quantiles must match the old quantile routine bitwise. The original ceiling rule must reproduce every retained Stage3 per-path component and every crossed cell mean for all five metrics within 1e-10. Both candidate ranks must pass the same finite-memory negative controls. The analysis rejects missing, duplicate and off-grid rows.

Save per-path phase metrics for **both candidate ranks, all eight cells, both worlds, all 30 models and both families** in an indexed compressed NumPy file. Also save per-path decompositions under both rules, all-cell model means, negative controls and reference reproduction checks. Analysis reconstructs all components and cell means from candidate metrics before inference, then saves all statistics, two figures and a readable summary. A separate reviewer can audit the expected-metric weighting and decomposition without rerunning neural prediction.

Commands after the freeze:

```sh
.venv/bin/python scripts/run_rank_sensitivity.py --self-check
.venv/bin/python scripts/analyze_rank_sensitivity.py --self-check
.venv/bin/python scripts/run_rank_sensitivity.py
.venv/bin/python scripts/analyze_rank_sensitivity.py
```

No HPC is required: 30 existing small fits are only re-evaluated, not trained. Estimated local wall time 10–30 minutes, subject to CPU availability. Stop if reproduction or input checks fail; repair transparently before claiming results. Do not select another rank rule, filter fits, or increase paths after seeing sensitivity outcomes.
