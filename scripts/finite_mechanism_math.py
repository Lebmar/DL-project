"""Numerical primitives for an exogenous scale-error mechanism experiment.

The archived log relative scale errors are deterministic (or exogenous and
independent of every innovation). ``iid_mixture_metrics`` describes independent
draws *from the archived mixture*, not a contiguous, non-identically distributed
calibration window. It is not a conditional law for learned adaptive scales.

Every metric is conditional on the supplied current/archived errors and takes
expectations over the calibration bank and an independent future innovation.
The true current scale is one. Multiplying ``width`` and ``interval_score`` by a
known positive current true scale gives their original-unit values.

Only toy numerical checks run under ``--self-check``; importing has no side
effects and this module does not save or run research conditions.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, ndtr, ndtri, roots_jacobi, stdtr, stdtrit
from scipy.stats import beta, norm, t


_T_DF = 5.0
_T_SCALE = np.sqrt((_T_DF - 2) / _T_DF)
_SQRT_TWO_PI = np.sqrt(2 * np.pi)
_DISTRIBUTIONS = ("gaussian", "t5")


def _check_distribution(distribution: str) -> None:
    if distribution not in _DISTRIBUTIONS:
        raise ValueError(f"distribution must be one of {_DISTRIBUTIONS}")


def _nonnegative_array(x):
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)) or np.any(x < 0):
        raise ValueError("absolute-value arguments must be nonnegative and not NaN")
    return x


def _alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return alpha


def absolute_cdf(x, distribution: str = "gaussian"):
    """CDF of |epsilon| for Gaussian or unit-variance Student t with df=5."""
    _check_distribution(distribution)
    x = _nonnegative_array(x)
    if distribution == "gaussian":
        return erf(x / np.sqrt(2))
    # The df=5 integral has an elementary form. It is both faster inside the
    # mixture root solver and more accurate near zero than 1-2*StudentSF(x).
    v = np.where(np.isfinite(x), x, 0) / np.sqrt(3)
    reciprocal = 1 / (1 + v**2)
    cdf = 2 / np.pi * np.arctan(v) + 2 / np.pi * v * reciprocal + 4 / (3 * np.pi) * v * reciprocal**2
    return np.where(np.isfinite(x), cdf, 1.0)


def absolute_pdf(x, distribution: str = "gaussian"):
    """Right-hand density of |epsilon|, including its factor of two."""
    _check_distribution(distribution)
    x = _nonnegative_array(x)
    if distribution == "gaussian":
        return 2 * np.exp(-x**2 / 2) / _SQRT_TWO_PI
    # For df=5 and scale=sqrt(3/5), the normalized density is
    # 16 / (3*pi*sqrt(3)) * (1 + x^2/3)^(-3).
    return 16 / (3 * np.pi * np.sqrt(3)) * (1 + x**2 / 3) ** -3


def absolute_pdf_derivative(x, distribution: str = "gaussian"):
    """Derivative of the absolute-value density for nonnegative arguments."""
    _check_distribution(distribution)
    x = _nonnegative_array(x)
    finite_x = np.where(np.isfinite(x), x, 0)
    slope = -finite_x if distribution == "gaussian" else -6 * finite_x / (3 + finite_x**2)
    return slope * absolute_pdf(x, distribution)


def absolute_ppf(probabilities, distribution: str = "gaussian"):
    """Absolute-value quantile, permitting p=0 and p=1 endpoint values."""
    _check_distribution(distribution)
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    # The lower tail representation avoids rounding (1+p)/2 up to one.
    tail = (1 - p) / 2
    if distribution == "gaussian":
        return -ndtri(tail)
    return -_T_SCALE * stdtrit(_T_DF, tail)


def expected_interval_score(halfwidth, distribution: str = "gaussian", alpha: float = 0.1):
    """Exact expected symmetric IS_alpha when the true innovation scale is one.

    IS_alpha([-a,a], y) = 2*a + (2/alpha)*max(|y|-a, 0).
    The t5 expression integrates the Student-t tail analytically.
    """
    _check_distribution(distribution)
    alpha = _alpha(alpha)
    a = _nonnegative_array(halfwidth)
    safe_a = np.where(np.isfinite(a), a, 0)
    if distribution == "gaussian":
        stop_loss = 2 * (np.exp(-safe_a**2 / 2) / _SQRT_TWO_PI - safe_a * ndtr(-safe_a))
    else:
        x = safe_a / _T_SCALE
        first_tail_moment = _T_SCALE * (_T_DF + x**2) / (_T_DF - 1) * t.pdf(x, _T_DF)
        stop_loss = 2 * (first_tail_moment - safe_a * stdtr(_T_DF, -x))
    # Only remove floating-point cancellation in a nonnegative tail integral.
    return np.where(np.isfinite(a), 2 * a + 2 / alpha * np.maximum(stop_loss, 0), np.inf)


def _archive(archived_delta, weights=None):
    archive = np.asarray(archived_delta, dtype=np.float64)
    if archive.ndim != 2 or 0 in archive.shape or not np.all(np.isfinite(archive)):
        raise ValueError("archived_delta must be a finite, nonempty [phase, age] array")
    if np.max(np.abs(archive)) > 100:
        raise ValueError("|archived_delta| > 100 is outside this numerical engine's supported range")
    if weights is None:
        weight = np.full(archive.shape, 1 / archive.shape[1])
    else:
        try:
            weight = np.broadcast_to(np.asarray(weights, dtype=np.float64), archive.shape).copy()
        except ValueError as exc:
            raise ValueError("weights must broadcast to [phase, age]") from exc
        if np.any(~np.isfinite(weight)) or np.any(weight < 0) or np.any(weight.sum(axis=1) <= 0):
            raise ValueError("weights must be nonnegative, finite, with a positive sum per phase")
        weight /= weight.sum(axis=1, keepdims=True)
    return archive, weight


def mixture_cdf(quantiles, archived_delta, distribution: str = "gaussian", weights=None):
    """Evaluate G_p(q)=sum_j w_pj F(q*exp(delta_pj)).

    ``quantiles`` is [phase] or [phase, node]; the output has the same shape.
    """
    archive, weight = _archive(archived_delta, weights)
    q = _nonnegative_array(quantiles)
    if q.ndim not in (1, 2) or q.shape[0] != archive.shape[0]:
        raise ValueError("quantiles must have shape [phase] or [phase, node]")
    if q.ndim == 1:
        return np.sum(weight * absolute_cdf(q[:, None] * np.exp(archive), distribution), axis=-1)
    return np.sum(weight[:, None, :] * absolute_cdf(q[:, :, None] * np.exp(archive[:, None, :]), distribution), axis=-1)


def population_mixture_quantile(
    archived_delta, probabilities=0.9, distribution: str = "gaussian", weights=None,
    *, iterations: int = 48, phase_chunk_size: int = 16,
):
    """Invert the exogenous archived mixture by vectorized log-space bisection.

    A scalar probability returns [phase]; a 1-D vector returns [phase, node].
    Probabilities must be interior to (0,1). The quantile bracket follows from
    min/max archived errors, so no tail truncation or finite search bound is
    imposed. ``iterations=48`` makes the log-bracket shorter than 7.2e-13 even
    across the largest supported error range; intended small-error studies are
    appreciably more accurate. Weighted mixtures are supported.
    """
    _check_distribution(distribution)
    archive, weight = _archive(archived_delta, weights)
    p = np.asarray(probabilities, dtype=np.float64)
    scalar = p.ndim == 0
    if p.ndim > 1 or np.any(~np.isfinite(p)) or np.any((p <= 0) | (p >= 1)):
        raise ValueError("probabilities must be a scalar or vector strictly in (0, 1)")
    p = np.atleast_1d(p)
    if p.size == 0 or iterations < 1 or phase_chunk_size < 1:
        raise ValueError("nonempty probabilities, positive iterations and positive chunk size required")
    result = np.empty((len(archive), len(p)), dtype=np.float64)
    base_log = np.log(absolute_ppf(p, distribution))
    for start in range(0, len(archive), phase_chunk_size):
        stop = min(len(archive), start + phase_chunk_size)
        local = archive[start:stop]
        local_weight = weight[start:stop]
        low = base_log[None, :] - local.max(axis=1)[:, None]
        high = base_log[None, :] - local.min(axis=1)[:, None]
        exp_archive = np.exp(local[:, None, :])
        for _ in range(iterations):
            middle = (low + high) / 2
            cdf = np.sum(local_weight[:, None, :] * absolute_cdf(np.exp(middle)[:, :, None] * exp_archive, distribution), axis=-1)
            below = cdf < p[None, :]
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        result[start:stop] = np.exp((low + high) / 2)
    return result[:, 0] if scalar else result


def rank_mixture(n: int, alpha: float = 0.1, rule: str = "ceiling"):
    """Return (rank, probability) pairs for finite supported order statistics.

    Randomization has mean rank (1-alpha)*(n+1). Metrics must be integrated
    separately for each rank; scoring an average halfwidth is incorrect.
    This engine requires all selected ranks in 1..n (no infinite sentinel).
    """
    alpha = _alpha(alpha)
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    target = (1 - alpha) * (n + 1)
    nearest = round(target)
    if abs(target - nearest) < 1e-12:
        target = float(nearest)
    low, high = int(np.floor(target)), int(np.ceil(target))
    if rule == "ceiling":
        ranks = ((high, 1.0),)
    elif rule == "randomized":
        ranks = ((low, 1.0),) if low == high else ((low, high - target), (high, target - low))
    else:
        raise ValueError("rule must be 'ceiling' or 'randomized'")
    if any(k < 1 or k > n for k, _ in ranks):
        raise ValueError("selected order-statistic rank is outside 1..n; infinite/zero sentinels are unsupported")
    return ranks


@lru_cache(maxsize=64)
def beta_quadrature(n: int, rank: int, quadrature_nodes: int = 64):
    """Gauss-Jacobi nodes/weights for Beta(rank,n+1-rank), without tail cuts.

    Returned arrays are read-only because this deterministic rule is cached.
    """
    if any(isinstance(v, bool) or int(v) != v for v in (n, rank, quadrature_nodes)):
        raise ValueError("n, rank, and quadrature_nodes must be integers")
    n, rank, quadrature_nodes = int(n), int(rank), int(quadrature_nodes)
    if not 1 <= rank <= n or quadrature_nodes < 1:
        raise ValueError("require 1 <= rank <= n and quadrature_nodes >= 1")
    a, b = rank, n + 1 - rank
    x, weight = roots_jacobi(quadrature_nodes, b - 1, a - 1)
    if not np.isfinite(weight).all() or not np.isfinite(weight.sum()) or weight.sum() <= 0:
        raise FloatingPointError("Jacobi normalization overflowed; use a smaller calibration count")
    nodes = (x + 1) / 2
    weight = weight / weight.sum()
    nodes.setflags(write=False)
    weight.setflags(write=False)
    return nodes, weight


def _current(current_delta, n_phases: int):
    current = np.asarray(current_delta, dtype=np.float64)
    if current.shape != (n_phases,) or not np.all(np.isfinite(current)) or np.max(np.abs(current)) > 100:
        raise ValueError("current_delta must be a finite [phase] array with |delta| <= 100")
    return current


def _point_metrics(q, current, distribution, alpha):
    z = float(absolute_ppf(1 - alpha, distribution))
    halfwidth = np.exp(current[:, None]) * q
    xi = current[:, None] + np.log(q / z)
    return {
        "coverage": absolute_cdf(halfwidth, distribution),
        "width": 2 * halfwidth,
        "interval_score": expected_interval_score(halfwidth, distribution, alpha),
        "quantile_mean": q,
        "log_relative_halfwidth_mean": xi,
        "log_relative_halfwidth_second_moment": xi**2,
    }


def population_metrics(current_delta, archived_delta, distribution: str = "gaussian", alpha: float = 0.1, weights=None, *, diagnostics=None):
    """Return a dictionary of six [phase] metrics for the population root."""
    alpha = _alpha(alpha)
    archive, weight = _archive(archived_delta, weights)
    current = _current(current_delta, len(archive))
    q = population_mixture_quantile(archive, 1 - alpha, distribution, weight)
    if diagnostics is not None:
        diagnostics["root_cdf_max_residual"] = float(np.max(np.abs(mixture_cdf(q, archive, distribution, weight) - (1 - alpha))))
    return {key: value[:, 0] for key, value in _point_metrics(q[:, None], current, distribution, alpha).items()}


def iid_mixture_metrics(
    current_delta, archived_delta, n: int, rule: str = "ceiling",
    distribution: str = "gaussian", alpha: float = 0.1,
    quadrature_nodes: int = 64, weights=None, *, quantile_cache=None, diagnostics=None,
) -> Mapping[str, np.ndarray]:
    """Return six [phase] expected metrics for an IID finite mixture bank.

    Conditional on exogenous archived errors, each of n independent scores
    chooses a fresh age from ``weights`` and a fresh independent innovation.
    For its k-th order statistic Q, G(Q)~Beta(k,n+1-k). Gauss-Jacobi integrates
    each metric through that exact identity. Only numerical quadrature/bisection
    is approximate; callers must compare quadrature orders in their protocol.

    Keys: coverage, width, interval_score, quantile_mean,
    log_relative_halfwidth_mean, log_relative_halfwidth_second_moment.
    No averaging over phases is performed here. An optional mutable
    ``quantile_cache`` reuses roots across rank rules/current errors. Its key
    includes exact archive/weight contents, distribution, count, rank, and node
    count. An optional mutable ``diagnostics`` receives
    ``root_cdf_max_residual``. The result remains only [phase] metric arrays.
    """
    alpha = _alpha(alpha)
    archive, weight = _archive(archived_delta, weights)
    current = _current(current_delta, len(archive))
    result = None
    max_residual = 0.0
    archive_key = hashlib.sha256(archive.tobytes() + weight.tobytes()).hexdigest() if quantile_cache is not None else None
    for rank, probability in rank_mixture(n, alpha, rule):
        nodes, node_weight = beta_quadrature(n, rank, quadrature_nodes)
        cache_key = (archive.shape, archive_key, distribution, int(n), rank, int(quadrature_nodes))
        cached = quantile_cache.get(cache_key) if quantile_cache is not None else None
        if cached is None:
            q = population_mixture_quantile(archive, nodes, distribution, weight)
            residual = float(np.max(np.abs(mixture_cdf(q, archive, distribution, weight) - nodes[None, :])))
            if quantile_cache is not None:
                q.setflags(write=False)
                quantile_cache[cache_key] = (q, residual)
        else:
            q, residual = cached
        max_residual = max(max_residual, residual)
        metrics = _point_metrics(q, current, distribution, alpha)
        integrated = {key: probability * (value @ node_weight) for key, value in metrics.items()}
        if result is None:
            result = integrated
        else:
            for key in result:
                result[key] += integrated[key]
    if diagnostics is not None:
        diagnostics["root_cdf_max_residual"] = max_residual
    return result


def self_check():
    """Independent toy identities; no experiment trajectories or result files."""
    checks = {}
    x = np.concatenate((np.array([0.0, 0.17, 1.2, 3.5]), np.logspace(-8, 3, 100)))
    for distribution in _DISTRIBUTIONS:
        dist = norm if distribution == "gaussian" else t(_T_DF, scale=_T_SCALE)
        np.testing.assert_allclose(absolute_cdf(x, distribution), 2 * dist.cdf(x) - 1, atol=3e-16, rtol=3e-14)
        np.testing.assert_allclose(absolute_pdf(x, distribution), 2 * dist.pdf(x), atol=3e-16, rtol=3e-14)
        z = float(absolute_ppf(0.9, distribution))
        dx = 1e-5
        numerical_density = (absolute_cdf(z + dx, distribution) - absolute_cdf(z - dx, distribution)) / (2 * dx)
        numerical_density_prime = (absolute_pdf(z + dx, distribution) - absolute_pdf(z - dx, distribution)) / (2 * dx)
        np.testing.assert_allclose(numerical_density, absolute_pdf(z, distribution), rtol=2e-9)
        np.testing.assert_allclose(numerical_density_prime, absolute_pdf_derivative(z, distribution), rtol=2e-9)
        for a in (0.0, 0.7, z, 3.0):
            stop_loss = quad(lambda value: 2 * dist.sf(value), a, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
            reference = 2 * a + 20 * stop_loss
            np.testing.assert_allclose(expected_interval_score(a, distribution), reference, atol=2e-10, rtol=2e-10)
        log_step = 1e-4
        numeric_curvature = (expected_interval_score(z * np.exp(log_step), distribution) + expected_interval_score(z * np.exp(-log_step), distribution) - 2 * expected_interval_score(z, distribution)) / (2 * log_step**2)
        coefficient = z**2 * absolute_pdf(z, distribution) / 0.1
        np.testing.assert_allclose(numeric_curvature, coefficient, rtol=2e-6)
        checks[f"{distribution}_abs_density_and_risk"] = True

        homogeneous = np.full((1, 7), 0.13)
        probabilities = np.array([0.02, 0.7, 0.99])
        q = population_mixture_quantile(homogeneous, probabilities, distribution)
        np.testing.assert_allclose(q[0], absolute_ppf(probabilities, distribution) * np.exp(-0.13), atol=3e-14, rtol=2e-14)
        current = np.array([0.13])
        for n in (32, 128):
            for rule in ("ceiling", "randomized"):
                result = iid_mixture_metrics(current, homogeneous, n, rule, distribution, quadrature_nodes=128)
                target_coverage = sum(prob * k / (n + 1) for k, prob in rank_mixture(n, rule=rule))
                np.testing.assert_allclose(result["coverage"], target_coverage, atol=5e-14, rtol=0)
                reference = 0.0
                for k, prob in rank_mixture(n, rule=rule):
                    beta_dist = beta(k, n + 1 - k)
                    reference += prob * quad(
                        lambda u: float(expected_interval_score(absolute_ppf(u, distribution), distribution)) * beta_dist.pdf(u),
                        0, 1, epsabs=1e-9, epsrel=1e-10, limit=200,
                    )[0]
                np.testing.assert_allclose(result["interval_score"], reference, atol=8e-8, rtol=2e-8)
                checks[f"{distribution}_n{n}_{rule}_coverage_error"] = abs(float(result["coverage"][0]) - target_coverage)
                checks[f"{distribution}_n{n}_{rule}_independent_integral_error"] = abs(float(result["interval_score"][0]) - reference)
                if n == 32 and rule == "ceiling":
                    errors = []
                    for node_count in (32, 64, 128):
                        refined = iid_mixture_metrics(current, homogeneous, n, rule, distribution, quadrature_nodes=node_count)
                        errors.append(abs(float(refined["interval_score"][0]) - reference))
                    assert errors[2] < errors[1] < errors[0]
                    checks[f"{distribution}_homogeneous_quadrature_errors_32_64_128"] = errors

        # Nonconstant errors: translation equivariance is exact for every u,
        # and weights/roots are checked independently by forward substitution.
        archive = np.array([[-0.15, 0.02, 0.11], [0.08, -0.09, 0.04]])
        current = np.array([0.03, -0.02])
        weight = np.array([0.2, 0.5, 0.3])
        q = population_mixture_quantile(archive, probabilities, distribution, weight)
        residual = np.max(np.abs(mixture_cdf(q, archive, distribution, weight) - probabilities[None, :]))
        assert residual < 2e-12
        base = iid_mixture_metrics(current, archive, 32, "randomized", distribution, quadrature_nodes=32, weights=weight)
        shifted = iid_mixture_metrics(current + 0.4, archive + 0.4, 32, "randomized", distribution, quadrature_nodes=32, weights=weight)
        for key in base:
            expected = base[key] * np.exp(-0.4) if key == "quantile_mean" else base[key]
            np.testing.assert_allclose(shifted[key], expected, atol=3e-13, rtol=3e-13)
        cache, diagnostics = {}, {}
        iid_mixture_metrics(current, archive, 32, "ceiling", distribution, quadrature_nodes=32, weights=weight, quantile_cache=cache)
        cached_result = iid_mixture_metrics(current, archive, 32, "randomized", distribution, quadrature_nodes=32, weights=weight, quantile_cache=cache, diagnostics=diagnostics)
        assert len(cache) == 2  # Shared upper rank was reused, rather than recomputed/stored.
        assert diagnostics["root_cdf_max_residual"] < 2e-12
        for key in base:
            np.testing.assert_array_equal(base[key], cached_result[key])
        checks[f"{distribution}_mixture_root_residual"] = float(residual)

    for n in (32, 128):
        for rank, _ in rank_mixture(n, rule="randomized"):
            nodes, weight = beta_quadrature(n, rank, 32)
            np.testing.assert_allclose(weight.sum(), 1, atol=3e-15)
            np.testing.assert_allclose(nodes @ weight, rank / (n + 1), atol=3e-15)
            np.testing.assert_allclose(nodes**2 @ weight, rank * (rank + 1) / ((n + 1) * (n + 2)), atol=3e-15)
    # For a centered trajectory, E[(delta-mu)^2] must retain the paired
    # covariance term. This catches a common error in mechanism summaries.
    current = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    archive_mean = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    mse = np.mean((current - archive_mean)**2)
    identity = np.mean(current**2) + np.mean(archive_mean**2) - 2 * np.mean(current * archive_mean)
    np.testing.assert_allclose(mse, identity, atol=1e-15)
    assert abs(mse - (np.mean(current**2) + np.mean(archive_mean**2))) > 1
    checks["beta_polynomial_moments_and_paired_covariance"] = True
    checks["passed"] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="Run deterministic toy identities only.")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("This is a reusable numerical module; only --self-check is executable.")
    print(json.dumps(self_check(), indent=2))


if __name__ == "__main__":
    main()
