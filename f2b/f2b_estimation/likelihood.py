"""
Estimation of fail-to-board probabilities from AFC and AVL data.
Likelihood functions.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-02-11"

from time import time

from numpy import linspace, log
from pandas import Timedelta, Timestamp
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm
from yaml import safe_load

from data import Data

from probability_laws import (
    bivariate_gaussian_CDF,
    bivariate_gaussian_PDF,
    log_normal_quotient_CDF,
    log_normal_quotient_PDF,
)

# ---------------------------------------------------------------------------------------
#
#                                  Offline computations
#
# ---------------------------------------------------------------------------------------


def compute_likelihood_blocks(
    data: Data,
    param: dict,
    distributed_speed: bool = False,
    time_distribution_assumption: str = None,
) -> dict:
    """Compute likelihood blocks for each couple of feasible headway and
    run per trip, and store the result in a dictionnary with tuple keys."""

    # Dictionnary with tuple index.
    blocks_by_trip_and_run_pair = {}

    print("Computing blocks...")
    for trip_id in tqdm(data.trips.index):
        for headway_boarded_pair in data.headway_boarded_run_pair_by_trip[trip_id]:

            # Pairs are ordered (headway_run, boarded_run)
            headway_run = headway_boarded_pair[0]
            boarded_run = headway_boarded_pair[1]

            if time_distribution_assumption:
                blocks_by_trip_and_run_pair[
                    trip_id, headway_run, boarded_run
                ] = compute_one_likelihood_block(
                    data,
                    param,
                    trip_id,
                    boarded_run,
                    headway_run,
                    time_distribution_assumption,
                )

    return blocks_by_trip_and_run_pair


def compute_one_likelihood_block(
    data: Data,
    param: dict,
    trip_id: int,
    boarded_run: str,
    headway_run: str,
    time_distribution_assumption: str,
) -> float:
    """Compute and return likelihood term for one trip, one headway run and one boarded run.
    Offline computation."""

    station_access_time = Timestamp(data.trips.at[trip_id, "access_time"])
    station_egress_time = Timestamp(data.trips.at[trip_id, "egress_time"])
    station_destination = data.trips.at[trip_id, "egress_station"]
    headway_run_departure_origin_station = Timestamp(
        data.runs_departures[data.date, headway_run, data.station_origin]
    )
    boarded_run_arrival_dest_station = Timestamp(
        data.runs_arrivals[data.date, boarded_run, station_destination]
    )

    access_time_upper_bound = (
        headway_run_departure_origin_station - station_access_time
    ).total_seconds()

    access_time_lower_bound = 0

    headway_previous_run = data.previous_run[date, headway_run]
    if headway_previous_run:
        try:
            headway_previous_run_departure_time = Timestamp(
                data.runs_departures[
                    data.date, headway_previous_run, data.station_origin
                ]
            )
            access_time_lower_bound = max(
                0,
                (
                    headway_previous_run_departure_time - station_access_time
                ).total_seconds(),
            )  # max in case station entrance after the departure of headway_previous_run

        except KeyError:  # runs feasible for no trips, information missing
            pass

    egress_time = (
        station_egress_time - boarded_run_arrival_dest_station
    ).total_seconds()

    if time_distribution_assumption == "gaussian":
        access_proba_difference = bivariate_gaussian_CDF(
            access_time_upper_bound,
            param["gaussian"][data.station_origin]["distance_mean"],
            param["gaussian"][data.station_origin]["distance_std"],
            param["gaussian"][data.station_origin]["speed_mean"],
            param["gaussian"][data.station_origin]["speed_std"],
            param["gaussian"][data.station_origin]["covariance"],
        ) - bivariate_gaussian_CDF(
            access_time_lower_bound,
            param["gaussian"][data.station_origin]["distance_mean"],
            param["gaussian"][data.station_origin]["distance_std"],
            param["gaussian"][data.station_origin]["speed_mean"],
            param["gaussian"][data.station_origin]["speed_std"],
            param["gaussian"][data.station_origin]["covariance"],
        )

        egress_proba = bivariate_gaussian_PDF(
            egress_time,
            param["gaussian"][station_destination]["distance_mean"],
            param["gaussian"][station_destination]["distance_std"],
            param["gaussian"][station_destination]["speed_mean"],
            param["gaussian"][station_destination]["speed_std"],
            param["gaussian"][station_destination]["covariance"],
        )

    if time_distribution_assumption == "log_normal":
        access_proba_diff = log_normal_quotient_CDF(
            access_time_upper_bound,
            param["log_normal"][data.station_origin]["time_mean"],
            param["log_normal"][data.station_origin]["time_std"],
        )

        if access_time_lower_bound > 0:
            access_proba_diff -= log_normal_quotient_CDF(
                access_time_lower_bound,
                param["log_normal"][data.station_origin]["time_mean"],
                param["log_normal"][data.station_origin]["time_std"],
            )

        egress_proba = log_normal_quotient_PDF(
            egress_time,
            param["log_normal"][station_destination]["time_mean"],
            param["log_normal"][station_destination]["time_std"],
        )

    return egress_proba * access_proba_difference


# ---------------------------------------------------------------------------------------
#
#                                  Inline computations
#
# ---------------------------------------------------------------------------------------


def minus_log_likelihood_global(
    f2b_probabilities: list, iteration: list, data: Data, precomputed_blocks: dict
) -> float:
    """Compute and return the inverse of the sum of indiviual log-likelihoods."""
    iteration[0] += 1
    minus_log_likelihood = -log_likelihood_global(
        f2b_probabilities, data, precomputed_blocks
    )
    if (iteration[0] % 100) == 0:
        print(f"At iteration {iteration[0]}: ")
        print(f2b_probabilities[0:10])
        print(f"Minus log likelihood value to minimize: {minus_log_likelihood} \n")

    return minus_log_likelihood


def log_likelihood_global(
    f2b_probabilities: list, data: Data, precomputed_blocks: dict
) -> float:
    """Compute and return the sum of the individual log-likelihoods."""

    data.AFC_df["Log-likelihood"] = data.AFC_df["index"].apply(
        compute_log_likelihood_indiv,
        args=(data, precomputed_blocks, f2b_probabilities),
    )
    return data.AFC_df["Log-likelihood"].sum()


def compute_log_likelihood_indiv(
    trip_id: int,
    data: Data,
    precomputed_blocks: dict,
    f2b_probabilities: list,
) -> None:
    """Compute the individual log-likelihood for trip_id and store in data_AFC."""

    first_feasible_run = data.feasible_runs_dict["first_feasible_run", trip_id]
    last_feasible_run = data.feasible_runs_dict["last_feasible_run", trip_id]

    exit_time_PDF = 0.0
    # another...
    for boarded_run in range(first_feasible_run, last_feasible_run + 1):
        for headway_run in range(first_feasible_run, boarded_run + 1):

            block = precomputed_blocks[trip_id, headway_run, boarded_run]
            exit_time_PDF += (
                transition_probability(boarded_run, headway_run, f2b_probabilities)
                * block
            )

    # Artificial to avoid 0 in log --> shouldn't be needed !!
    # if abs(exit_time_PDF) < 1.0e-60:
    # exit_time_PDF += 1.0e-60

    return log(exit_time_PDF)


def transition_probability(
    boarded_run: int, headway_run: int, f2b_probabilities: list
) -> float:
    """Function returning the probability of boarding boarded_run
    when arriving at headway_run according to the f2b_probability."""

    if headway_run == boarded_run:
        return 1 - f2b_probabilities[headway_run]

    elif boarded_run > headway_run:
        failure_probability = 1
        for run in range(headway_run, boarded_run):
            failure_probability = f2b_probabilities[run] * failure_probability
        return (1 - f2b_probabilities[boarded_run]) * failure_probability


if __name__ == "__main__":
    station_origin = "VIN"
    stations_destination = ["NAT", "LYO"]
    date = "03/02/2020"
    direction = "west"

    initialization = False

    with open("parameters.yml", "r") as file:
        param = safe_load(file)

    data = Data(date, direction, station_origin, stations_destination)

    # Offline precomputations.

    likelihood_blocks = compute_likelihood_blocks(data, param, False, "gaussian")
    print(likelihood_blocks)

    # Initialize f2b proba for tests.
    if initialization:
        initial_probability_range = linspace(0, 1, 50)
        objective_values = []
        for initial_probability in initial_probability_range:
            f2b_probabilities = [initial_probability for _ in range(data.mission_nbr)]
            start_time = time()
            objective_values.append(
                minus_log_likelihood_global(
                    f2b_probabilities, [0], data, likelihood_blocks
                )
            )
            print(
                f"Objective function evaluation execution time : {time() - start_time:.2}s"
            )

        min_value = min(objective_values)
        index = objective_values.index(min_value)
        print(f"best initial probability:{initial_probability_range[index]}")
        print(f"best initial likelihood: {objective_values[index]}")
