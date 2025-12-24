from dataclasses import dataclass
from math import sqrt
from typing import NamedTuple
import numpy as np
from scipy.stats import norm, beta


@dataclass
class ABResult:
    variant_a_rate: float
    variant_b_rate: float
    lift: float
    z_score: float
    p_value: float


def frequentist_z_test(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
) -> ABResult:
    """Two-sided z-test on difference in proportions."""
    p_a = success_a / total_a
    p_b = success_b / total_b
    p_pool = (success_a + success_b) / (total_a + total_b)

    se = sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))
    z = (p_b - p_a) / se
    p = 2 * (1 - norm.cdf(abs(z)))

    lift = (p_b - p_a) / p_a if p_a > 0 else 0.0

    return ABResult(
        variant_a_rate=p_a,
        variant_b_rate=p_b,
        lift=lift,
        z_score=z,
        p_value=p,
    )


def bayesian_prob_beats_control(
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    samples: int = 20000,
) -> float:
    """Estimate P(B > A) using Beta–Bernoulli posteriors."""
    alpha_a, beta_a = success_a + 1, total_a - success_a + 1
    alpha_b, beta_b = success_b + 1, total_b - success_b + 1

    draws_a = beta.rvs(alpha_a, beta_a, size=samples)
    draws_b = beta.rvs(alpha_b, beta_b, size=samples)

    return float(np.mean(draws_b > draws_a))
