# Decision gates and next experiments

Status: exploratory design frozen on 2026-10-11. This is a plan, not evidence that a new algorithm exists.

## Gate 1: distinguish input context from effective scale response

Use the current GRU checkpoints and paired simulation paths to measure context truncation sensitivity, time-resolved scale response, and sensitivity to earlier observations. Compare a second neural architecture (small causal TCN or attention model) trained on the same independent generator. Train longer according to validation loss, since the GRU pilot's best validation epoch is the final eighth epoch. Do not tune to the known test pulses.

Intervene on **current** scale context and **archived-score denominator** context separately, crossing both with calibration window. Off-diagonal combinations are diagnostic interventions; they are not a newly guaranteed conformal method. Verify the diagonal reproduces the original GRU experiment. This separates a current scale response from a change to archived normalized scores, which the original context intervention changed together.

## Gate 2: survive controls and closest prior work

On identical issued predictions, run tuned-on-validation rolling calibration, ACI/AgACI, SF-OGD/SAOCP and at least one neural/PID calibration baseline. Add a properly fitted GARCH baseline, retaining the current EWMA control. Source-code pins, initialization and scope of any theoretical assumptions must be explicit. The current strong-baseline defaults are feasibility checks, not an optimized leaderboard.

Complete the full-text novelty check against the Cho–Matteson TSFM/GARCH workshop paper; current access yielded its original abstract and official acceptance, not the complete paper. Distinguish NCC, AdaptNC, RetroAdj and RareCP in the final literature discussion. Resetting, replay, gating, multiple windows and neural residual controllers already have direct precedents.

## Gate 3: generalize a bounded claim

Before new runs, freeze an independent generator grid over event duration, magnitude, persistence and tail family. Use new path seeds; keep exploratory results visible. Student-t innovations require the correct Student-t oracle, not the Gaussian oracle. Include a stochastic-volatility generator so conclusions are not limited to known piecewise-constant variance. Avoid labelling retrospectively chosen real-market dates as ground-truth regimes.

The ECB 2021–2026 interval has already been inspected. Future changes evaluated on it remain exploratory. A confirmatory component can use a newly frozen independent simulation grid; any new real-data holdout should be chosen and sealed before inspecting outcomes. Four EUR crosses do not establish broad financial generalization.

## Gate 4: decide whether a new module is justified

Only if a neural-specific failure persists beyond controls, design a small causal module informed by that mechanism. Compare it with a simple rule/linear model using the same inputs. No event onset, true scale or future duration may enter a deployable module. Do not count a common-current-scale cancellation as a method: `s_t * Q(|e_i| / s_t) = Q(|e_i|)`.

If effective-memory changes have little effect, the interaction changes sign without an explanation, or EWMA/GARCH already explains everything, narrow to a bounded empirical result or pivot the contribution. A null result by itself is not evidence of A+ quality.

## Schedule

| Date | Concrete outcome |
|---|---|
| October 11–13 | Review proposal and primary papers; inspect pilot code/results; define precise hypotheses and independent test grid. |
| October 14–16 | Student revises the one-page proposal, checks course policy and seeks teaching-staff feedback; submit by October 16, 23:59. |
| October 17–25 | Effective-memory diagnostics, a second architecture, sufficient validation-based training, strong-baseline validation. |
| October 26–November 8 | Freeze contribution and experimental protocol; run independent robustness grid. |
| November 9–22 | Ablations, public-data check, uncertainty and failure-case analysis. |
| November 23–December 3 | Write four-page main report, prepare figures and reproducibility appendix. |
| December 4–10 | Slides, record 8–10-minute video, verify submission artifacts. |
| December 11 | Submit report, slides and video before 23:59. |
