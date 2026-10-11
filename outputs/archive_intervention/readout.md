# Frozen-GRU archive intervention: exploratory diagnostic

No training, tuning, or test-based model selection. Three pre-existing models and the same paired paths.

Off-diagonal a != b combines current scale from context a with archived, forecast-time denominators from context b. This is an artificial causal intervention, not a proposed deployment algorithm or guaranteed-coverage method.

Diagonal a=b reproduces original metrics with maximum absolute error 2.84e-14. Known Gaussian CDF and sigma are used for evaluation only.

Effects below subtract no_change at identical target dates. Units are interval-score units. Confidence intervals resample model seeds and paired path ids separately; only three model seeds, exploratory comparisons without multiplicity adjustment.

| Scenario / phase | Effect | Mean | Crossed-bootstrap 95% CI | Seed range |
|---|---|---:|---|---|
| persistent / onset32 | original_diagonal | -0.20454 | [-0.24639, -0.16496] | [-0.21455, -0.19516] |
| persistent / onset32 | current_x_window_archive64 | -0.14442 | [-0.26058, -0.05840] | [-0.26796, -0.05771] |
| persistent / onset32 | archive_x_window_current16 | -0.06011 | [-0.14560, 0.04381] | [-0.14619, 0.05340] |
| shift_128 / recovery64 | original_diagonal | -0.05705 | [-0.07352, -0.03979] | [-0.06843, -0.04967] |
| shift_128 / recovery64 | current_x_window_archive64 | 0.02892 | [-0.00603, 0.08425] | [-0.00351, 0.08768] |
| shift_128 / recovery64 | archive_x_window_current16 | -0.08596 | [-0.13344, -0.04794] | [-0.13735, -0.04954] |

For each path, original = current_x_window_archive64 + archive_x_window_current16 exactly. An alternative path gives current_x_window_archive16 + archive_x_window_current64; both are reported.

log(halfwidth)=log(current_scale)+log(archived_q). At fixed archive context, current_context × window interaction on log width is zero by construction, not an empirical discovery. Interval score is nonlinear, so its components still depend on current scale.

The archive component changes sign across model seeds at persistent onset; the early effect cannot be attributed entirely to archived denominators. The recovery attribution differs, illustrating phase dependence. Neither interpretation establishes novelty, market performance, or an improved deployable method.

Reproduce: `.venv/bin/python scripts/run_archive_intervention.py`. The original GRU trajectory CSV may be plain or gzip-compressed. Per-path summaries are saved, not giant repeated per-timestep CSVs.
