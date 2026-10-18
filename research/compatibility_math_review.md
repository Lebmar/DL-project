# A bounded analytic direction: temporal compatibility of scale errors and calibration

Date: 2026-10-18. Status: exploratory algebra and deterministic feasibility, not a preregistered experiment, new empirical confirmation, or established novelty claim. This note does not modify the frozen Stage 2/3, rank-sensitivity, FX, or course-report results.

## Recommendation

Investigate whether the **temporal structure of relative scale error**, beyond native predictive accuracy, explains when normalized-score calibration improves or degrades interval quality. Start with exact population quantiles and controlled exogenous trajectories; only then add finite calibration samples and learned forecasters. The first demonstration below is feasible and mathematically consistent, but does not yet explain the observed FX ordering or establish a publishable contribution.

The useful distinction is between a slowly varying relative scale error that the calibrator can correct, and a rapidly varying error for which its archived correction is less relevant. It is not a universal statement that a longer model context or longer calibration window is beneficial.

## 1. Exact population identity and local expansion

Let

\[
y_t=\sigma_t\epsilon_t,\quad s_t=\sigma_t e^{\delta_t},\quad
F(x)=P(|\epsilon_t|\leq x),\quad z=F^{-1}(1-\alpha).
\]

For this controlled calculation, the trajectories \(\sigma_t>0\) and \(\delta_t\) are deterministic, or exogenous and independent of all innovations. Innovations share a continuous absolute-value distribution with sufficiently smooth density \(f\) and \(f(z)>0\). With weights \(w_j\geq0\), \(\sum_jw_j=1\), the population archived-score mixture is

\[
G_t(q)=\sum_jw_jF(qe^{\delta_{t-j}}),\qquad
G_t(q_t)=1-\alpha.
\]

Its interval halfwidth relative to the correct scale is

\[
\frac{h_t}{\sigma_t}=e^{\delta_t}q_t.
\]

This is a population-mixture quantile. It is **not** the expectation of a finite-window empirical quantile.

Write \(\mu_t=\sum_jw_j\delta_{t-j}\), \(v_t=\sum_jw_j(\delta_{t-j}-\mu_t)^2\), and \(\eta_t=\log(q_t/z)\). For uniformly small errors \(\delta_j=O(\varepsilon)\), Taylor expansion of \(F(ze^u)\) yields

\[
\eta_t=-\mu_t-\frac12\left(1+\frac{zf'(z)}{f(z)}\right)v_t+O(\varepsilon^3),
\]

and therefore

\[
\xi_t:=\log\frac{h_t}{\sigma_t z}
=\delta_t-\mu_t-\frac12\left(1+\frac{zf'(z)}{f(z)}\right)v_t+O(\varepsilon^3).
\]

To verify the coefficient, set \(H(u)=F(ze^u)\). Then \(H'(0)=zf(z)\), \(H''(0)=zf(z)+z^2f'(z)\). Substituting \(\eta=-\mu+c\), with \(c=O(\varepsilon^2)\), in \(\sum_jw_j[H(\eta+\delta_j)-H(0)]=0\) gives \(c=-H''(0)v/[2H'(0)]+O(\varepsilon^3)\).

For standard Gaussian innovations and \(\alpha=0.1\), \(F(x)=2\Phi(x)-1\), \(f(x)=2\phi(x)\), and \(f'(z)/f(z)=-z\). Thus

\[
\xi_t=\delta_t-\mu_t+0.8527717270\,v_t+O(\varepsilon^3).
\]

The positive second-order term reflects within-window heterogeneity of relative scale error. It does not imply every interval becomes too wide: the first-order current-minus-archive term can have either sign.

For a constant relative scale error \(\delta_j=c\), cancellation is exact: \(q=ze^{-c}\) and \(h_t=\sigma_tz\). Common multiplicative rescaling of every archived and current scale also cancels exactly for any finite empirical order statistic. This invariance is a basic normalization property, not a proposed new theorem or method.

## 2. What interval-score optimality adds

Assume also \(E|\epsilon|<\infty\). For a zero-centered symmetric interval with halfwidth \(\sigma_t a\), its conditional expected interval score is

\[
R_t(a)=2\sigma_t a+\frac{2\sigma_t}{\alpha}E[(|\epsilon|-a)_+].
\]

Then \(R'_t(z)=0\), \(R''_t(z)=2\sigma_tf(z)/\alpha\). For log-relative halfwidth error \(\xi\),

\[
R_t(ze^\xi)-R_t(z)=\frac{\sigma_t z^2f(z)}{\alpha}\xi^2+O(\xi^3).
\]

The Gaussian coefficient at \(\alpha=0.1\) is \(5.5807591340\,\sigma_t\). The factor of two in \(f=2\phi\) matters.

To second order, native and calibrated excess risks are consequently proportional to

\[
\sum_t\sigma_t\delta_t^2
\quad\text{and}\quad
\sum_t\sigma_t(\delta_t-\mu_t)^2,
\]

respectively. Calibration helps under this approximation when

\[
2\sum_t\sigma_t\delta_t\mu_t>\sum_t\sigma_t\mu_t^2.
\]

This is a local population criterion, not a deployment guarantee. For constant \(\sigma\), the change is governed by the covariance of present and archived error as well as the variance of the archived average. With zero-mean exogenous white error of variance \(V\) and a uniform past window \(W\), the first-order calibrated variance is \(V(1+1/W)\), exceeding native \(V\). With persistent bias, the opposite may occur. Finite score noise and endogenous scale estimation are additional effects.

## 3. The filtering interpretation and a controlled ranking construction

For a uniform window, define

\[
B_W(\omega)=\frac1W\sum_{j=1}^W e^{-i\omega j}.
\]

The first-order response to a sinusoidal relative error has gain \(|1-B_W(\omega)|\). This is a high-pass interpretation; its gain need not be below one. For stationary exogenous errors and constant \(\sigma\), integrated excess risk is locally weighted by \(|1-B_W(\omega)|^2\) across the error spectrum. This is a standard linear-filter calculation and requires a literature check before any novelty claim.

For an illustrative log-volatility response template \(\widehat{x}=H x\), with \(x=\log\sigma\), the relative error is \(\delta=(H-I)x\); the calibrated first-order error is \((I-B_W)(H-I)x\). Such a template uses known simulated volatility to isolate a response mechanism. It is not an implementable forecast that knows future volatility.

For example, after an upward step a monotone slow scale response gives negative error that approaches zero. Its archived errors remain more negative than its current error, so calibration can eventually produce positive \(\delta_t-\mu_t\) even while the base scale is still below truth. A downward transition gives a complementary possibility. This provides a discriminating lag/overshoot test; it does not establish that the GRU or TCN actually follows that template.

The implemented deterministic probe uses \(\sigma_t=1\), a periodic 256-phase history, \(W=32\), and

\[
\delta_t=\varepsilon\sin\bigl(2\pi(kt\bmod256)/256\bigr),\qquad k\in\{1,17\}.
\]

Both multipliers are coprime with 256, so the two trajectories are exact permutations of the same error values. Native expected Gaussian NLL, interval score, coverage, and width are consequently identical. Only temporal ordering differs. Frequencies and amplitudes are an exploratory mathematical construction, not a preregistered or randomly selected empirical test set.

At amplitude 0.05, the native expected interval score is **4.1323965352** for both. The population-calibrated score is **4.1265298284** for the slow ordering and **4.1317641877** for the fast ordering; the oracle score is **4.1254256150**. Both improve in this particular construction. The result demonstrates different compatibility at identical native metrics, not universal harm from fast errors.

The first-order excess-risk ratios relative to native risk are 0.15829955 and 0.90810851. Hence a slow-error amplitude divided by a fast-error amplitude between 1 and \(\sqrt{0.90810851/0.15829955}\approx2.395\) gives a local ranking reversal. The already-computed pair slow amplitude 0.05 versus fast amplitude 0.025 provides an exact numerical illustration: native scores **4.1323965352 versus 4.1271692877** favor fast; population-calibrated scores **4.1265298284 versus 4.1270095727** favor slow. This pair is an explanatory construction after the initial demonstration, not an independent statistical finding.

## 4. Numerical checks completed

Run `scripts/check_compatibility_expansion.py --output <new-or-empty-directory>` using the repository environment. Existing output directories containing files are refused. Saved artifacts are in `outputs/compatibility_math_probe/`.

- Fixed amplitudes: 0.2, 0.1, 0.05, 0.025, 0.0125; no fitting, training, market data, or bootstrap.
- Maximum population-root CDF residual: \(6.99\times10^{-15}\).
- Largest difference between matched native risk/coverage/width/NLL: \(8.88\times10^{-16}\).
- Constant-rescaling cancellation error: \(2.22\times10^{-16}\).
- On halving the last two amplitudes, first-order approximation error has observed orders 2.0000 and 2.0010; second-order approximation error has observed orders 3.0332 and 3.2069. The latter can converge faster than cubic at finite amplitude because of higher-order terms; it is not required to equal 3 exactly.
- An independent finite difference of the absolute-Gaussian CDF verifies the density factor; a finite difference of exact Gaussian expected interval score gives the log-risk coefficient 5.5807591437, matching the analytic value.
- The final script was rerun in a temporary fresh directory; both numerical artifacts reproduced byte-for-byte. Only the manifest provenance was refreshed after adding overwrite protection and explicit exploratory-status metadata. Existing-output refusal was checked.

These checks verify algebra and implementation. They are not evidence that these formulas are new, robust to all distributions, or sufficient to predict neural-model behavior.

## 5. Next bounded study: separate history span, sample count, and rank

Do not immediately train a large model grid. A small, separately frozen mechanism study can use the following controls:

1. **Population control:** compute the mixture root for fixed exogenous error templates and history spans \(L=32,128\). This changes historical composition with no finite-order-statistic noise or rank discretization. Retain both exact roots and the local approximation so large-error failure is visible.
2. **Independent count/span control:** hold a span \(L\) fixed and create an idealized calibration bank with \(n=32,128\) independent scores drawn from its mixture. A draw chooses a historical age uniformly and then draws an independent innovation at that age's fixed scale error. Span and sample count are thereby distinct. Reusing an observed score at a repeated age would not produce independent samples and must not be substituted silently. This bank is a mechanism experiment, not an available real-data calibration strategy.
3. **Finite-rank control:** compare the usual ceiling rank with adjacent-rank randomization whose mean rank is \(0.9(n+1)\). Average interval scores/coverage for each candidate interval, not the score of the averaged width. Rank matching removes an idealized finite-rank difference, not temporal non-exchangeability.
4. **Practical-window bridge:** also evaluate contiguous historical scores where count equals span. This is deliberately a different condition, preserving the actual observation schedule and dependence. It tests whether idealized conclusions survive the operational setting.

For the IID mixture-bank control, finite-sample calculations can even avoid Monte Carlo: \(G(Q_{(k)})\) follows a Beta\((k,n+1-k)\) distribution, so expected risk is an integral of the exact inverse mixture CDF. This identity applies to IID draws from the mixture, not to the non-identically distributed contiguous window. For the latter, the order-statistic CDF can instead be expressed through a Poisson-binomial exceedance count when exogenous errors and independent innovations hold. Either route is established order-statistic mathematics; first implement only the population and small IID-mixture controls unless this additional precision answers a clear question.

## 6. What would justify further project investment

Proceed toward learned models only if an independently fixed study shows that the mechanism distinguishes otherwise similarly accurate predictors across multiple response types and innovation distributions, with a finite-sample account of where it breaks. Then test on new synthetic paths and training panels, using a diagnostic chosen without those outcomes. Compare it with native NLL, native interval score, a single constant scale recalibration, and simple error persistence/volatility baselines.

The current result cannot be transferred directly to the learned-model setting. For a learned adaptive scale, \(\delta_i\) depends on prior innovations; conditioning on the whole realized error trajectory can reveal information about earlier innovations through later scales. Thus the exogenous mixture identity is not automatically a conditional law for the original process. Current residuals, rolling quantiles, and scale-estimation errors can also be correlated. Known synthetic volatility can diagnose this gap; real volatility is latent, so an operational diagnostic must not use true \(\sigma_t\).

A publishable contribution would require more than the elementary expansion or a constructed reversal: for example, a carefully delimited predictive compatibility result with a useful finite-sample extension, or a validated decision rule that improves held-out model/calibrator selection beyond simple controls. If these additions fail, report the mechanism as an explanatory analysis and retain the course project as a rigorous empirical study. Do not manufacture a neural-specific claim from a phenomenon that already appears in simple filters.
