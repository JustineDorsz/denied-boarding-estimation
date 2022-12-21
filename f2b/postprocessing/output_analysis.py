"""
Estimation of fail-to-board probabilities (also referred to as delayed boarding 
probabilities) by train run from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-11"

from math import sqrt

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood import Likelihood
from numpy import array, delete, linalg
from pandas import DataFrame


def load_estimated_f2b(station: str, method: str = "") -> array:
    """Load fail-to-board estimated probabilities from file.

    Args:
        - station(int): code of the origin station of the estimated f2b
        - method(str): estimation method with which was computed f2b

    Return:
        -array: estimated fail-to-board probabilities by run in chronological order
    """
    f2b_file_path = "output/f2b_results/" + method + "_" + station + ".csv"
    with open(f2b_file_path, "r") as f2b_file:
        f2b_file_content = f2b_file.read()
        f2b_estimated = f2b_file_content.split(",")
        f2b_estimated = array([float(f2b) for f2b in f2b_estimated])
        return f2b_estimated


def get_estimation_results_and_indicators(
    data: Data,
    likelihood: Likelihood,
) -> DataFrame:
    """Loads estimated fail-to-board estimated probabilities.
    Computes gradient and asymptotic confidence interval bounds
    for each run.

    Args:
        - data(Data): trips and runs info
        - likelihood(Likelihood): likelihood at estimated fail to board probability

    Return:
        DataFrame: estimation results and indicators for each run.
        Each run corresponds to a row, each column to a value of interest:
            - "departure_time"(Timestamp): departure time form the origin station
            - "estimated_f2b"(float): estimated fail-to-board probability
            - "trip_number"(int): number of trips associated to the run
            - "log_likelihood"(float): sum of the log-likelihoods contributions
            of the trips associated to the run, evaluated at the estimated
            fail-to-board probability
            - "gradient"(float): log-likelihood gradient term corresponding to the run
            evaluated at the estimated fail-to-board probability
            - "asymptotic_confidence"(float): diagonal term of the asymptotic variance
            divided by the number of associated trips to the run.
            The asymptotic variance is computed as the inverse of the opposite of
            the hessian at the estimated fail-to-board probability.
    """

    estimation_info_df = DataFrame(
        data=likelihood.f2b_probas, columns=["estimated_f2b"]
    )

    estimation_info_df["departure_time"] = [
        data.runs[run_code].departure_times[data.origin_station]
        for run_code in data.runs_chronological_order
    ]

    log_likelihood_contributions_by_run = array([0.0 for _ in range(data.runs_number)])
    # loop over run and associated trips for likelihoods contribution by run

    for run_position, run_code in enumerate(data.runs_chronological_order):
        for trip_id in data.runs[run_code].associated_trips:
            log_likelihood_contributions_by_run[
                run_position
            ] += likelihood.individual_log_likelihoods[trip_id]

    estimation_info_df["log_likelihood"] = log_likelihood_contributions_by_run
    estimation_info_df["trips_number"] = [
        data.runs[run_code].associated_trips_number
        for run_code in data.runs_chronological_order
    ]

    estimation_info_df["gradient"] = likelihood.get_global_log_likelihood_gradient()
    hessian_log_likelihood = likelihood.get_global_log_likelihood_hessian()

    # Singular runs (with zero associated trips) imply null terms in the log-likelihood
    # hessian matrix. Remove corresponding row and column in hessian matrix
    # for inversion.
    list_of_singular_runs = []
    for (run_position, run_code) in enumerate(data.runs_chronological_order):
        if not data.runs[run_code].associated_trips:
            list_of_singular_runs.append(run_position)
    hessian_without_singular_rows = delete(
        hessian_log_likelihood, list_of_singular_runs, 0
    )
    hessian_without_singular_runs = delete(
        hessian_without_singular_rows, list_of_singular_runs, 1
    )

    # asymptotic variance is the inverse of -hessian
    fischer_info = -hessian_without_singular_runs
    asymptotic_variance_diag = linalg.inv(fischer_info).diagonal().tolist()
    # add zero term for singular runs for dimensional compatibility
    shift = len(list_of_singular_runs)
    for singular_run_position in list_of_singular_runs:
        asymptotic_variance_diag.insert(singular_run_position - shift, 0)
        shift -= 1

    # divide by the numbre of observations for asymptotic confidence
    asymptotic_confidence = array([0.0 for _ in range(data.runs_number)])
    for run_position, run_code in enumerate(data.runs_chronological_order):
        if data.runs[run_code].associated_trips:
            asymptotic_confidence[run_position] = sqrt(
                abs(
                    asymptotic_variance_diag[run_position]
                    / data.runs[run_code].associated_trips_number
                )
            )

    estimation_info_df["asymptotic_confidence"] = asymptotic_confidence

    return estimation_info_df
