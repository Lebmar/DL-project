# Frozen exploratory ECB extension

Specified on 18 October 2026, before training or viewing any result from this extension. The executable configuration is configs/fx_extension_protocol.json. This is a bounded extension of the existing financial pilot, not a fresh blind test: the 2021–2026 history was already inspected. Original CPU simulations, their selected checkpoints, and the original financial pilot remain unchanged.

## Data, chronology and scope

Use only the existing ECB ZIP with SHA-256 3d4f0b3877df2d995df6c6be36638b8923b283fac473e588df4d70798c32c501. No download, new assets, refreshed data or private data are permitted. The four series are USD, JPY, GBP and CHF in currency units per EUR, spanning7,111 publication dates from1999-01-04 through2026-10-09. The target is the next publication-date percentage log return, 100 log(rate_t/rate_previous). There is no weekend zero imputation.

Each currency's standard deviation is fitted only on returns through2018-12-31, with ddof1. Divide by this number without demeaning. Neural target indices start after64 historical returns; older observations may be used as inputs across later split boundaries.

- Gradient training: target dates through2018-12-31.
- Checkpoint selection: all2019 target dates, equally weighted across assets and contexts.
- Online-calibrator scale heuristic: each stream's own2019 scores, after selecting the model. This is retrospective2019 preparation, not a claim that the selected checkpoint was deployed during2019.
- Calibration replay: starts with the first2020 observation, independently for each currency/model/seed/context. The first128 replay observations are excluded from any reporting. Continue the same state across the2021 boundary.
- Exploratory evaluation:1,479 common publication dates,2021-01-04 through2026-10-09. Never use these targets for gradient training, checkpoint selection, calibrator tuning, asset choice or selecting a seed.

The archive contains the whole history when read, but fitting receives only the declared training/selection indices. This is not a point-in-time revision-vintage dataset. The series share a common EUR denominator and dates; currencies and model seeds do not create independent market histories. Reference rates are informational, not executable trading prices, intraday realized volatility or evidence of trading returns.

## Models and fixed training

Use the frozen stage2_models GaussianGRU and GaussianTCN without modifications:2,041 and1,897 parameters, zero center, original two features observed_return/3 and log1p(abs(observed_return)), and log-scale clipping at[-3,3]. The input returns here are divided by the currency's training standard deviation. Context16 and64 share all learned weights within each fit; TCN reset-window inference respects the selected context, even though its structural receptive field is64.

Train both architectures at each paired seed 4301–4305, for ten fits total. Pool all four standardized currencies. Each batch samples assets uniformly among the four currencies and target indices uniformly with replacement from eligible training dates. Individual batches need not contain equal numbers from every currency. Every epoch has 32 batches of 128, with exactly 16 batches per context. Independent NumPy samplers reset to the same seed for each architecture, making their context/asset/target minibatch prefixes identical. Save a digest of every epoch's actual sampling sequence and verify the paired prefixes.

Adam learning rate.002, gradient clipping5, maximum60 epochs, minimum20 epochs before stopping, patience8 and minimum improvement1e-4 are fixed. Validation uses every2019 target for every asset and both contexts. The saved checkpoint is the absolute minimum mean validation NLL, including small improvements below the patience threshold. The minimum epoch rule controls stopping, not whether an earlier checkpoint can be selected. Stopping times may differ, so comparisons concern complete training procedures. Keep all ten fitted models; numerical failures cause investigation rather than unrecorded exclusion.

Training and final inference both run on the local CPU with four Torch threads, float32, deterministic algorithms, and highest float32 matmul precision. This resource choice was fixed before any new FX result to avoid uncertain HPC charges. The earlier GPU runtime smoke is a separate environment check and is neither an experiment nor a prerequisite here. No paid HPC/GPU scientific training is authorized by this protocol.

## Classical scales and calibration

Fit the frozen stage2_baselines constant scale, EWMA16, EWMA64 and Gaussian GARCH(1,1) separately for each currency, using its raw percentage returns through2018. GARCH retains the frozen four-start optimization, train-only fit selection and convergence/boundary diagnostics. Generate each classical scale sequence over the entire observed history once and then slice by date; do not reset variance at2019,2020 or2021. The training-based initial variance is used for all earlier replay; only later historical predictions are evaluated.

Every scale stream receives native Gaussian, rolling32, rolling128 and clipped ACI128 intervals. The native half-width is the90% Gaussian multiplier times the issued scale. Rolling uses rank ceil((W+1)*.9), with scores abs(return_i)/issued_scale_i for strictly earlier dates. ACI starts at alpha.1 and updates by.02*(.1−miss) after observing the current return, clipped to[1/129,.5]. This is the explicitly clipped practical procedure, not an unrestricted theoretical ACI guarantee.

The pinned author-source SF-OGD and SAOCP methods are additionally applied to every neural context64 stream, at all five seeds, and all classical streams. Their source and license are preserved; use the existing read-only external-score adapter. SAOCP lifetime is8. For each individual stream, maximum expected score is sqrt(3) times its own maximum2019 normalized score. It is a heuristic, not a hard future bound; actual scores are never clipped. Counts exceeding the heuristic are saved as end-of-run diagnostics and never used as contemporaneous inputs.

All methods replay only their own currency's chronological2020 observations. No shared synthetic-path warmup is broadcast across assets. Rolling/ACI first update once their own window is available; author methods update from the first replay observation. A common128-observation reporting burn-in makes this difference explicit, and the entire2020 year remains unscored. This yields96 scale streams and496 interval configurations, including56 streams for each strong calibrator.

## Eight-cell pathway analysis on actual history

For each neural fit, currency and date, retain the full eight combinations a,b in{16,64}, W in{32,128}:

h_t(a,b,W) = sigma_t(a) Q_W({abs(return_i)/sigma_i(b): t−W <= i < t}).

Historical denominators remain their originally issued values. Crossed cells diagnose current and archived scale pathways; the diagonal a=b must reproduce independent rolling calibration.

Let D(a,b) denote the daily or mean interval-metric difference between W128 andW32. Define J_FX = D(64,64)−D(16,16). Preserve both exact decompositions:

- [D(64,64)−D(16,64)] + [D(16,64)−D(16,16)].
- [D(64,16)−D(16,16)] + [D(64,64)−D(64,16)].

Also retain the third-order difference between the two current components. **J_FX has no event-minus-control subtraction.** Real data offer no matched no-shock counterfactual, so this differs from simulated shock J. Signs are descriptive computational contrasts, not causal mediation percentages, effective-memory estimates or absolute model rankings. No post-hoc event selection, regime labeling or real-market recovery guarantee is introduced.

## Evaluation, dependence and predeclared displays

Report coverage, width and interval score2h+20 max(abs(return)−h,0). Raw width/IS retain percentage-log-return units. For pooled comparisons, divide each asset's width/IS by that asset's train-only standard deviation; coverage is unchanged. Average all five fixed neural fits within currency/method first, then average the four currencies equally. Retain per-asset, individual-fit and calendar-year outputs;2026 is explicitly partial through October9.

Use2,000 circular date-block resamples with seed2026101802. Block128 is the main display, with20,60 and256 retained as sensitivities. Each block-length calculation resets a NumPy generator to this fixed seed. Every comparison within that calculation uses exactly the same date indices for all currencies, fitted models, methods and pathway cells. This preserves their pairing. All95% intervals are exploratory, unadjusted and conditional on these saved fits. They are not training-population intervals, IID currency/seed intervals or guarantees under arbitrary nonstationarity. A256-date block leaves only about six blocks in the sample and is a coarse sensitivity.

For each identical calibrator among native, rolling32, rolling128, ACI128, SF-OGD and SAOCP, predeclare context64 GRU−GARCH, TCN−GARCH and GRU−TCN comparisons. Keep all other methods descriptively; no winning seed, context, calibrator or block length defines success.

Before viewing any new result, fix the report displays:

1. Main-text small table: context64 GRU/TCN and GARCH, each with native and rolling128, showing coverage and normalized IS. The saved table also provides normalized width and the corresponding main-block intervals.
2. Appendix matrix: constant/EWMA16/EWMA64/GARCH/GRU64/TCN64 by the six calibrators, showing normalized IS.
3. The text must say whether the complete results support the same interpretation; all coverage, width, asset, year and seed results remain available. These displays are not selected by result ranking.

## Saved artifacts and independent audit

The runner writes a new outputs/fx_extension directory and refuses any existing directory. It saves:

- Protocol copy, model checkpoints, all training histories, best epochs, paired sampling digests and model/source hashes.
- Classical parameters and fitting diagnostics, source ZIP hash, exact date/count summaries, training standard deviations, versions and CPU runtime settings.
- forecast_index.csv and issued_forecasts.npz:96 indexed scales from2019 onward, four actual-return streams, dates and units.
- interval_index.csv and calibration_streams.npz:496 indexed native/online q streams from2020 onward, with explicit initial invalid reporting rows.
- calibration_metadata.json: separate2019 scale heuristics, clipping counts and untruncated-score exceedances.
- crossed_metrics.npz: all ten models, four currencies, eight cells and1,479 dates, preserving coverage/width/IS.

The analysis independently reconstructs every crossed daily metric from issued scales and q. It separately reproduces every native/rolling/ACI q sequence with explicit past-window order statistics and scalar ACI timing, verifies the pinned source/scale specs for strong methods, and checks the exact stream/asset/seed/date grids. Missing, duplicate or off-grid identities are rejected before averaging. All arrays and checksums must match the completed runner manifest.

Analysis outputs include method_summary.csv, paired_method_comparisons.csv, mechanism_summary.csv, per_asset_summary.csv, individual_fit_asset_summary.csv, calendar_year_summary.csv, mechanism_individual_fit_asset_summary.csv, report_core_table.csv, report_calibration_matrix.csv, daily_method_metrics.npz, two PNG figures and summary.md. Main pooled summary width/IS columns are normalized; raw units appear explicitly in per-asset/per-fit tables. The report renderer must read these saved outputs, without fitting, selecting or recalculating a different statistical procedure.

Synthetic checks use distinct non-scientific seeds and no ECB data. They cover current/future suffix exclusion for GRU/TCN and all calibrators, separate asset warmups, scalar classical recursions, independent sorted rolling quantiles and scalar ACI, paired/balanced minibatches, both factorial routes, joint date-block bootstrap against scalar loops, and malformed-grid rejection.

## Freeze and execution

Commit the JSON protocol, this document and both scripts before any actual training/evaluation. Both runner and analysis verify that the four files match HEAD, plus the frozen model/baseline/author-adapter dependency hashes. Verify again before completing results; preserve all original artifacts. A failure must be diagnosed transparently rather than silently replacing outputs or selecting a changed scientific rule after looking at results.

Local commands after root approval of the concrete frozen files:

    .venv/bin/python scripts/run_fx_extension.py --self-check
    .venv/bin/python scripts/analyze_fx_extension.py --self-check
    .venv/bin/python scripts/run_fx_extension.py
    .venv/bin/python scripts/analyze_fx_extension.py

No new coverage theorem, trading advantage, publication claim or grade guarantee follows from this experiment. Informative negative results and unstable comparisons are retained.
