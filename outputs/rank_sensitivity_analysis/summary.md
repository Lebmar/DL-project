# Finite-rank sensitivity of the frozen Stage 3 study

This is post-hoc sensitivity on the same already-viewed paths and all 30 saved models. No new training, model selection, or confirmatory test was performed.

The ceiling rule uses ranks 30/32 and 117/128. The randomized90 rule chooses ranks 29/30 with probabilities .30/.70 for W32, and 116/117 with probabilities .90/.10 for W128. Expected interval metrics are computed exactly over those choices; the interval score is never evaluated at an averaged width.

| Phase | Rank rule / paired change | GRU−TCN J | Fixed-fit path interval | Crossed sensitivity |
|---|---|---:|---|---|
| entry | ceiling | -0.198573 | [-0.248766, -0.149846] | [-0.426968, +0.016007] |
| entry | randomized90 | -0.194599 | [-0.239047, -0.151159] | [-0.413756, +0.012122] |
| entry | randomized90_minus_ceiling | +0.003974 | [-0.012415, +0.021049] | [-0.026229, +0.032452] |
| recovery64 | ceiling | +0.006947 | [-0.004943, +0.019845] | [-0.054969, +0.067110] |
| recovery64 | randomized90 | +0.004268 | [-0.006173, +0.015531] | [-0.051851, +0.057971] |
| recovery64 | randomized90_minus_ceiling | -0.002679 | [-0.010257, +0.004832] | [-0.014684, +0.010667] |

Gaussian reference intervals are 97.5% and correspond to the two fixed phase endpoints within each rule/paired-change series. They do not provide simultaneous protection across rules, components, metrics, or the entire sensitivity exercise. All other intervals are 95% descriptive and unadjusted.

The common ideal randomized rank level is 90% under continuous exchangeable scores. No coverage guarantee follows for these dependent nonstationary neural scores. This rule lowers both original finite-rank levels; it is not a unique causal correction or a percentage of the original effect explained by rank rounding. History length, number of observations, and empirical-quantile variability remain different.

Fixed-fit intervals average all 15 fitted pairs before resampling the same 128 paths. Crossed sensitivity resamples three panels, five shared seed indices, and common paths, retaining the same seed draw across panels. It cannot establish a reliable training-population ranking from three panels.

Coverage deficit is the expectation of each realized candidate interval's positive oracle deficit, and excess width is defined analogously. It is not a positive-part calculation after averaging candidates. Exact rank integration omits Monte Carlo variation from realizing the rank coins; it estimates the random procedure's expected performance.

Ceiling reproduction: {"passed": true, "component_path_rows": 92160, "cell_rows": 1920, "component_path_max_abs_error": 0.0, "all_cell_metric_max_abs_error": 0.0, "reference_hashes": {"paired_component_effects.csv.gz": "f46eb20576216226cb6facbd7b599b12014c239a719dca8043aaf5fb25b7039d", "crossed_performance_by_seed.csv": "4d817d468e7358523e4416d8d5b573b780e2be820c9d1dd6fcb99d1d90fbeee1"}, "absolute_tolerance": 1e-10}

Files: summary.csv (all components/metrics/architecture and rule contrasts), panel_summary.csv, individual_fits.csv, leave_one_panel_out.csv, reference_architecture_contrasts.csv, cell_performance_summary.csv, and two figures. Per-path lower/upper candidate metrics remain in ../rank_sensitivity/rank_candidate_phase_metrics.npz for independent audit.
