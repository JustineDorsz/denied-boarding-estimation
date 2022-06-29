"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-06-20"


from math import exp, log, sqrt
from scipy.stats import norm


def shifted_exp_CDF(x: float, inverse_mean: float, shift: float) -> float:
    """Cumulative distribution of shifted exponential law."""
    if x < shift:
        return 0
    else:
        return 1 - exp(inverse_mean * (x - shift))


def shifted_exp_PDF(x: float, inverse_mean: float, shift: float) -> float:
    """Density function of shifted exponential law."""
    if x < shift:
        return 0
    else:
        return exp(inverse_mean * (x - shift)) * 1 / inverse_mean


def gaussian_CDF(x: float, mean: float, std: float) -> float:
    """Cumulative distribution of normal law."""
    return norm.cdf(x, mean, std)


def gaussian_PDF(x: float, mean: float, std: float) -> float:
    """Density function of normal law."""
    return norm.pdf(x, mean, std)


def bivariate_gaussian_CDF(
    x: float,
    distance_mean: float,
    distance_std: float,
    speed_mean: float,
    speed_std: float,
    covariance: float,
) -> float:
    y_x = sqrt(distance_std**2 + (x * speed_std) ** 2 - 2 * x * covariance)
    num_phi = x * speed_mean - distance_mean
    return norm.cdf(num_phi / y_x)


def bivariate_gaussian_PDF(
    x: float,
    distance_mean: float,
    distance_std: float,
    speed_mean: float,
    speed_std: float,
    covariance: float,
) -> float:
    y_x = sqrt(distance_std**2 + (x * speed_std) ** 2 - 2 * x * covariance)
    phi = norm.pdf((x * speed_mean - distance_mean) / y_x)
    num = speed_mean * (distance_std**2 - x * covariance) + distance_mean * (
        x * speed_std**2 - covariance
    )
    denom = y_x**3
    return num * phi / denom


def log_normal_quotient_CDF(x: float, mean: float, std: float) -> float:
    if x == 0:
        x = 0.01
    return norm.cdf((log(x) - mean) / std)


def log_normal_quotient_PDF(x: float, mean: float, std: float) -> float:
    if x == 0:
        x = 0.01
    frac = 1 / (x * std)
    return frac * norm.pdf((log(x) - mean) / std)
