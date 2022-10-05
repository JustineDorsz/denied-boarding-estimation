"""
Analysis of differential informations at estimated f2b.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-09-23"

from math import log

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood_recursive_blocks import (
    compute_access_individual_likelihoods,
    compute_egress_individual_likelihoods,
    gradient_log_likelihood_global,
    hessian_log_likelihood_global,
    log_likelihood_global_with_egress_auxiliary_variables_updates,
)
from f2b.postprocessing.output_plots import load_estimated_f2b
from matplotlib import pyplot
from numpy import array, linalg
from pandas import DataFrame
from yaml import safe_load


def get_estimation_info_by_run(
    f2b_estimated: list,
    origin_station: str,
    destination_stations: str,
    date: str,
) -> DataFrame:

    estimation_info_df = DataFrame(data=f2b_estimated, columns=["estimated_f2b"])

    data = Data(date, origin_station, destination_stations)

    access_individual_likelihoods = compute_access_individual_likelihoods(
        data, parameters
    )
    egress_individual_likelihoods = compute_egress_individual_likelihoods(
        data, parameters
    )
    individual_likelihoods = {}
    individual_log_likelihood_gradients = {}
    access_auxiliary_variables = {}
    egress_auxiliary_variables = {}

    log_likelihood_global_with_egress_auxiliary_variables_updates(
        f2b_estimated,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    log_likelihood_contributions_by_run = array(
        [0.0 for _ in range(len(f2b_estimated))]
    )
    trip_number_by_run = array([0.0 for _ in range(len(f2b_estimated))])
    for run_position in range(len(f2b_estimated)):
        run_code = data.runs[run_position]
        trip_number_by_run[run_position] = len(data.concerned_trips_by_run[run_code])
        for trip_id in data.concerned_trips_by_run[run_code]:
            log_likelihood_contributions_by_run[run_position] += log(
                individual_likelihoods[trip_id]
            )
    estimation_info_df["log_likelihood"] = log_likelihood_contributions_by_run
    estimation_info_df["trips_number"] = trip_number_by_run

    gradient_log_likelihood = gradient_log_likelihood_global(
        f2b_estimated,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_log_likelihood_gradients,
        data,
    )
    estimation_info_df["gradient"] = gradient_log_likelihood

    trip_of_interest = 0
    hessian_log_likelihood = hessian_log_likelihood_global(
        f2b_estimated,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_log_likelihood_gradients,
        trip_of_interest,
        data,
    )

    fischer_info = -hessian_log_likelihood

    asymptotic_variance = linalg.inv(fischer_info)

    asymptotic_confidence = array(
        [asymptotic_variance[i, i] for i in range(len(data.runs))]
    ) / len(data.trips)

    estimation_info_df["asymptotic_confidence"] = asymptotic_confidence

    return estimation_info_df


if __name__ == "__main__":
    origin_station = "VIN"
    destination_stations = ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"]
    date = "04/02/2020"

    with open(f"f2b/parameters_{origin_station}.yml") as file:
        parameters = safe_load(file)

    f2b_estimated = load_estimated_f2b(origin_station, True)

    estimation_info_df = get_estimation_info_by_run(
        f2b_estimated, origin_station, destination_stations, date
    )
    estimation_info_df_not_zero = estimation_info_df[
        estimation_info_df["estimated_f2b"] != 0
    ].copy()
    print("Partial diff infos.")
    print(estimation_info_df_not_zero["gradient"].describe())

    print("Asymptotic confidence infos.")
    print(estimation_info_df_not_zero["asymptotic_confidence"].describe())
    non_negative_count = (
        (estimation_info_df_not_zero["asymptotic_confidence"] >= 0).sum().sum()
    )
    print(f"Number of non negative values: {non_negative_count}.")

    pyplot.plot(estimation_info_df_not_zero["asymptotic_confidence"])
    pyplot.yscale("log")
    pyplot.show()
