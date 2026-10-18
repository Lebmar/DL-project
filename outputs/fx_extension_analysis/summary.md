# Exploratory ECB FX extension

This extends the previously inspected 2021–2026 history with all ten local CPU neural fits, four public EUR-cross currencies, classical scale models, and fixed calibration rules. It is not a new blind test. No paid HPC training was used.

The model fits use returns through2018, checkpoint selection and calibration scale heuristics use2019, and each asset-specific calibrator replays2020 before continuing into2021. The historical archive is not point-in-time revision data. All model choices were frozen before this run.

| Same-calibrator comparison | Calibrator | Normalized IS difference | Exploratory95% interval |
|---|---|---:|---|
| gru_minus_garch11 | native | -0.062336 | [-0.121200, -0.000688] |
| tcn_minus_garch11 | native | -0.074239 | [-0.122580, -0.027416] |
| gru_minus_tcn | native | +0.011903 | [-0.006860, +0.035074] |
| gru_minus_garch11 | rolling32 | +0.086101 | [+0.055592, +0.120613] |
| tcn_minus_garch11 | rolling32 | +0.054217 | [+0.030977, +0.079065] |
| gru_minus_tcn | rolling32 | +0.031884 | [+0.001104, +0.069660] |
| gru_minus_garch11 | rolling128 | +0.027990 | [+0.001323, +0.062733] |
| tcn_minus_garch11 | rolling128 | +0.012393 | [-0.009673, +0.034295] |
| gru_minus_tcn | rolling128 | +0.015597 | [-0.000694, +0.037594] |
| gru_minus_garch11 | aci128 | +0.034907 | [+0.010655, +0.057520] |
| tcn_minus_garch11 | aci128 | +0.011645 | [-0.028428, +0.040926] |
| gru_minus_tcn | aci128 | +0.023262 | [+0.003105, +0.050889] |
| gru_minus_garch11 | sfogd | +0.062621 | [+0.038376, +0.088936] |
| tcn_minus_garch11 | sfogd | +0.026742 | [+0.005364, +0.049549] |
| gru_minus_tcn | sfogd | +0.035879 | [+0.012863, +0.063478] |
| gru_minus_garch11 | saocp | +0.067676 | [+0.033319, +0.112414] |
| tcn_minus_garch11 | saocp | +0.022759 | [+0.007244, +0.039124] |
| gru_minus_tcn | saocp | +0.044917 | [+0.019095, +0.079248] |

Negative differences mean lower interval score for the first named predictor; coverage and width must be read alongside this score. Every method and asset remains in the accompanying tables; the figure uses predeclared comparisons, not a selected winning subset.

Date-block intervals use2,000 common circular resamples with block128 as the main display and20/60/256 as sensitivities. Five fixed neural fits are averaged within each asset before equal asset averaging. Neither seeds nor currencies are counted as independent financial histories. All intervals are exploratory, unadjusted and conditional on these fitted models; arbitrary nonstationarity is not covered.

J_FX is the actual-history contrast [D(64,64)−D(16,16)], where D(a,b) compares windows128 and32. It does not subtract a matched no-shock world, so it differs from simulated J. Both current/archive decomposition orders and the third-order term are retained, along with each asset and seed. Its sign is not an absolute model ranking or unique causal contribution.

The Gaussian interval model and heuristic online scale bounds provide no coverage guarantee. Scores are never truncated. Reference rates are informational and are not executable trading prices or realized volatility. Calendar-year tables identify2026 as partial through2026-10-09.

Audit: {"independent_native_rolling_aci_streams": 384, "maximum_absolute_q_error": 0.0, "strong_source_commit_and_scale_specs_checked": true, "crossed_all_cell_metric_reconstruction_error": 0.0, "diagonal_performance_reconstruction_error": 0.0, "exact_indices": true}

Outputs: all-method summary, paired same-calibrator comparisons, raw and normalized per-asset/per-fit summaries, calendar-year results, J_FX mechanism tables, two figures, and daily fixed-fit method metrics. Original issued forecasts, quantile streams, full factorial daily metrics and checkpoints are retained in ../fx_extension/.
