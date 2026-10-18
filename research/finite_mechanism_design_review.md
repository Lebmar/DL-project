# Independent design audit: finite-sample temporal compatibility

Date: 2026-10-18. Design review before the new finite-sample outcomes. This note proposes tests and acceptance criteria; it does not register a completed experiment or establish novelty. The final executable protocol, including its exact seeds and thresholds, must be frozen separately before outcome generation. Reports and frozen prior studies remain untouched.

## Decision

The proposed study is worth running, provided the primary question is **whether temporal ordering predicts a finite-window risk difference at matched native quality**, not whether every calibrated interval beats its native counterpart. Finite quantile noise can erase a small absolute improvement while leaving a reliable difference between two temporal orderings. Longer windows may amplify rather than attenuate some frequencies.

Start with a small exogenous study. Treat the endogenous experiment as an explicit boundary test. A successful result justifies a limited fresh neural-model bridge; it does not yet justify a new adaptive calibration algorithm or a publishability claim.

## Objects that must remain separate

For a fixed phase and error trajectory, distinguish:

1. The exact population-mixture root, with history span `L` and no finite bank.
2. An IID mixture bank with independently drawn ages and innovations, with sample count `n` independent of `L`.
3. One innovation at every actual historical age, where a contiguous bank has `n=L` and non-identical score distributions.

Repeatedly sampling one already-observed score at an age is not the IID mixture control. The Beta law for the transformed order statistic applies only to object 2. Its quantile expectation is not object 1. For object 3, conditioning on an exogenous error trajectory leaves independent, non-identically distributed scores; it does not make them IID.

Use `L in {32,128}`, `n in {32,128}` for the IID bank, and `n=L` for the operational bridge. Include both ceiling ranks and adjacent-rank mixtures with mean rank `0.9(n+1)`. Integrate the interval metrics of the two ranks, not the metric of their averaged halfwidth. Calling this rule “matched 90%” refers to its ideal IID exchangeable rank target, not guaranteed coverage in a changing-score process.

## Primary and secondary contrasts

The existing 256-phase sine orderings with multipliers 1 and 17 are exact permutations. For each amplitude and innovation distribution, they therefore have identical native marginal NLL, interval score, coverage, and width. The primary effect is

`D = average calibrated risk(slow) - average calibrated risk(fast)`

at the same `L`, `n`, and rank rule. Keep the zero native contrast as a verified invariant. Do not replace this with a post hoc amplitude pair chosen to exhibit a reversal.

The root design subsequently fixed Gaussian and variance-standardized t5, amplitudes `{0.05,0.2,0.5}`, and both spans. Its primary family is the eight contrasts from amplitude `0.2`, contiguous `n=L`, randomized ranks, two distributions, two spans, and sine/square templates. It uses **fast minus slow**, the negative of the illustrative `D` convention above; the protocol and implementation must use this convention consistently. The small amplitude checks local accuracy; the large amplitude is a deliberate approximation stress. Ceiling ranks and other response templates remain sensitivity tests. Step templates need fully specified phase weights and observation horizons before evaluation; otherwise their pooled risk can be changed by arbitrary amounts of stationary history.

The planned 64-phase schedule `128,132,...,380` is a fixed diagnostic phase grid, not all 256 phases. Verify native-metric equality and exact multiset equality on that actual schedule. Define the square wave by integer phase (for example, positive on phases 0 through 127 and negative on 128 through 255), not the floating-point sign of a sine at its zeros.

If the primary sign is taken from the population calculation, record that sign before finite-sample draws. Do not require all secondary cells to have the same sign: span and frequency interact by construction. Preserve both the sign and magnitude when an approximate theory fails.

## Risk and uncertainty

For every simulated halfwidth, evaluate the expected interval score conditional on that halfwidth by integrating the independent current innovation. This removes unnecessary target noise. Also retain exact conditional coverage and width. A causal endogenous scale is measurable before the current innovation, so this conditional-risk evaluation remains valid even when the exogenous mixture identity fails.

Prefer the Beta integral as an independent numerical benchmark for IID mixture banks when accurate quadrature is feasible. Otherwise simulate independent IID mixture banks. Exact integration of the same mixture model is a ground-truth calculation, not successful prediction of a real process.

For pseudo-Monte Carlo, one independent innovation replication, after averaging its entire fixed phase cycle, is the uncertainty unit. Pair all models, templates, ranks, and windows within each replication where possible. For randomized quasi-Monte Carlo, use independent scrambles as the uncertainty units; points within a Sobol net are not IID replicates. Record the number of scrambles, points per scramble, and seed. Do not bootstrap phases as if they were independent samples. Use a predeclared fixed simulation size, with no outcome-triggered doubling to obtain a significant sign.

The eight primary contrasts use two-sided 99.375% intervals for an approximate Bonferroni familywise 95% assessment; ordinary 95% intervals can also be displayed descriptively. The root's proposed 2,048 independent innovation paths are the sampling units. With 4,000 bootstrap replicates, a 99.375% endpoint is only about 12.5 order statistics from the tail. Prefer freezing 16,000 or 20,000 replicates if inexpensive, or explicitly retain that Monte Carlo limitation without outcome-triggered reruns. Mark all remaining intervals as descriptive, and report their denominator/count explicitly. A numerical quadrature tolerance is distinct from a statistical confidence interval.

## Prediction baselines

Compare the exact finite risk with the exact population risk and with the first- and second-order log-width formulas. A finite-noise correction must depend on innovation distribution, count, and rank. A useful deliberately simple correction is the exact oracle finite-order-statistic excess risk for that `(distribution,n,rank)` added uniformly to the population prediction. It is a baseline, not an asserted asymptotic theorem for heterogeneous scores.

Keep a native-risk-plus-the-same-noise baseline. It cannot distinguish the two matched-marginal orderings, which makes it a clear null comparator. Also retain the simple current-minus-past-average error statistic; the proposed theory is closely related to this baseline, so an algebraically identical implementation must not be presented as an independent sophisticated method.

Report error both in absolute interval-score units and relative to the actual temporal-order contrast. Relative error alone is unstable near zero. An overall correlation dominated by amplitude 0.5 is not evidence of local prediction accuracy. Show amplitudes separately.

## Endogenous boundary test

A minimal causal example is

`delta_t = a * sign(|epsilon_(t-1)| - median(|epsilon|))`.

This creates a two-level, equal-probability log-scale error with a precisely defined lag. Compare the ordinary coupled stream with a decoupled stream using **the same delta trajectory** and independent innovations for its archived scores and current target. This preserves the realized error ordering while removing the score/scale coupling. Include zero-amplitude and constant-error controls.

The main output is the paired coupled-minus-decoupled difference and the failure, if any, of the exogenous prediction. It is not necessary for the exogenous theory to pass this deliberately invalid-assumption test. Conversely, a null difference in this one toy does not establish general robustness to endogeneity. Do not call the difference causal evidence about real GRU/TCN fits.

## Proposed gates before any neural bridge

Freeze exact thresholds with the executable protocol. A defensible narrow set is:

1. **Implementation gate:** population-root CDF residual below `1e-10`; matched native metrics and constant-scale cancellation agree within `1e-10`; IID quadrature and an independently implemented sampler agree within their combined numerical/sampling tolerance; rank mixture and no-future-data checks pass. Any failure blocks interpretation.
2. **Existence gate:** the root proposes at least six of the eight primary temporal contrasts having the first-order predicted sign and a simultaneous 99.375% interval excluding zero in that direction. This is reasonable as a feasibility gate. For a bridge claiming scope over both distributions and spans, require at least three of four successes within each distribution and within each span; otherwise keep the overall gate but restrict the bridge to the supported conditions. A threshold of two of four would add nothing because it already follows from six of eight. Retain all cell results. Only count a predicted direction when its absolute magnitude exceeds a fixed numerical tolerance such as `1e-10`. Failing this gate does not disprove the population algebra; it means the operational finite-window effect is not sufficiently demonstrated under this fixed budget.
3. **Quantitative gate:** the root proposes median relative contrast error at most 0.5, with denominator `max(abs(actual contrast),0.001)`. Compute this over **all eight primary cells**, not only successful or significant cells. Report absolute error as well and name 0.001 a numerical floor. Also show performance against the zero-contrast/native-quality null, without selecting windows, distributions, or ranks after seeing the results. Keep amplitude 0.5 stress out of this gate. If the improvement is confined to one structural corner, narrow the neural question accordingly rather than silently expanding the success claim.
4. **Boundary gate:** document finite-noise and endogenous failures, including the practical difference between population and empirical quantiles. This is a reporting/completeness gate, not a requirement that every stress case succeed.

These gates authorize only a **small diagnostic study of newly trained models**, with new training panels, distinct seeds across panels, new common diagnostic paths, and untouched test paths. Known simulated true scale may be used to test the mechanism, but then the quantity is an oracle diagnostic, not an implementable financial input. Architecture identity, native NLL, native interval score, a static rescaling, and direct validation interval score remain necessary comparators. Avoid treating paths or phase observations as independent trained-model units.

If the quantitative gate fails but the existence gate passes, the justified conclusion is a finite-sample explanatory mechanism. First refine the finite-noise approximation on a declared development set and reserve a new confirmation design, rather than immediately training many networks. If both fail, preserve the negative result and reassess the question; adding architectures would not repair it.
