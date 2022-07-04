"""
Estimation of fail-to-board probabilities from AFC and AVL data.
Likelihood functions.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-02-11"

from time import time

from numpy import linspace, log
from pandas import Timestamp
from tqdm import tqdm
import scipy.stats
from yaml import safe_load

from f2b.f2b_estimation.data import Data

# ---------------------------------------------------------------------------------------
#
#                                  Offline computations
#
# ---------------------------------------------------------------------------------------


def compute_likelihood_blocks(
    data: Data,
    param: dict,
) -> dict:
    """Compute likelihood blocks for each couple of feasible headway and
    run per trip, and store the result in a dictionnary with tuple keys."""

    # Dictionnary with tuple index.
    blocks_by_trip_and_run_pair = {}

    print("(Offline) Compute likelihood blocks...")
    for trip_id in tqdm(data.trips.index):
        try:
            headway_boarded_pairs = data.headway_boarded_run_pair_by_trip[trip_id]
        # Skip trips with no feasible run.
        except KeyError:
            continue

        for headway_boarded_pair in headway_boarded_pairs:
            # Pairs are ordered (headway_run, boarded_run)
            headway_run = headway_boarded_pair[0]
            boarded_run = headway_boarded_pair[1]

            blocks_by_trip_and_run_pair[
                trip_id, headway_run, boarded_run
            ] = compute_one_likelihood_block(
                data,
                param,
                trip_id,
                boarded_run,
                headway_run,
            )
    return blocks_by_trip_and_run_pair


def compute_one_likelihood_block(
    data: Data,
    param: dict,
    trip_id: int,
    boarded_run: str,
    headway_run: str,
) -> float:
    """Compute and return likelihood term for one trip, one headway run and one boarded run.
    Offline computation."""

    # Get distributions and distributions parameters of walking duration, in origin and
    # destination station, stored in param.
    destination_station = data.trips.at[trip_id, "egress_station"]
    station_access_distrib_name = param[data.origin_station]["distribution"]
    station_access_duration_distribution = eval(
        "scipy.stats." + station_access_distrib_name
    )
    station_access_duration_distribution_params = param[data.origin_station][
        "parameters"
    ]
    station_access_distrib_name = param[destination_station]["distribution"]
    station_egress_duration_distribution = eval(
        "scipy.stats." + station_access_distrib_name
    )
    station_egress_duration_distribution_params = param[destination_station][
        "parameters"
    ]
    #  Get individual trips access and egress time, runs arrival and departure times.
    station_access_time = Timestamp(data.trips.at[trip_id, "access_time"])
    station_egress_time = Timestamp(data.trips.at[trip_id, "egress_time"])
    headway_run_departure_origin_station = Timestamp(
        data.runs_departures[headway_run, data.origin_station]
    )
    boarded_run_arrival_dest_station = Timestamp(
        data.runs_arrivals[boarded_run, destination_station]
    )

    access_duration_upper_bound = (
        headway_run_departure_origin_station - station_access_time
    ).total_seconds()

    access_duration_lower_bound = 0

    headway_previous_run = data.previous_run[headway_run, destination_station]
    if headway_previous_run:
        try:
            headway_previous_run_departure_time = Timestamp(
                data.runs_departures[headway_previous_run, data.origin_station]
            )
            access_duration_lower_bound = max(
                0,
                (
                    headway_previous_run_departure_time - station_access_time
                ).total_seconds(),
            )  # max in case station entrance after the departure of headway_previous_run

        except KeyError:  # runs feasible for no trips, information missing
            pass

    egress_duration = (
        station_egress_time - boarded_run_arrival_dest_station
    ).total_seconds()

    # Compute joint trip with headway and feasible run probability.
    access_proba_difference = station_access_duration_distribution.cdf(
        access_duration_upper_bound, **station_access_duration_distribution_params
    ) - station_access_duration_distribution.cdf(
        access_duration_lower_bound, **station_access_duration_distribution_params
    )

    egress_proba = station_egress_duration_distribution.pdf(
        egress_duration, **station_egress_duration_distribution_params
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

    data.trips["log-likelihood"] = data.trips["index"].apply(
        compute_log_likelihood_indiv,
        args=(data, precomputed_blocks, f2b_probabilities),
    )
    return data.trips["log-likelihood"].sum()


def compute_log_likelihood_indiv(
    trip_id: int,
    data: Data,
    precomputed_blocks: dict,
    f2b_probabilities: list,
) -> None:
    """Compute the individual log-likelihood for trip_id and store in data_AFC."""

    conditional_exit_duration_proba = 0

    for headway_boarded_pair in data.headway_boarded_run_pair_by_trip[trip_id]:
        # Pairs are ordered (headway_run, boarded_run)
        headway_run = headway_boarded_pair[0]
        boarded_run = headway_boarded_pair[1]
        block = precomputed_blocks[trip_id, headway_run, boarded_run]
        conditional_exit_duration_proba += (
            transition_probability(
                trip_id, data, boarded_run, headway_run, f2b_probabilities
            )
            * block
        )

    return log(conditional_exit_duration_proba)


def transition_probability(
    trip_id: int,
    data: Data,
    boarded_run: str,
    headway_run: str,
    f2b_probabilities: list,
) -> float:
    """Function returning the probability of boarding boarded_run
    when arriving at headway_run according to the f2b_probability."""

    headway_run_index = data.runs.index(headway_run)
    boarded_run_index = data.runs.index(boarded_run)

    if headway_run == boarded_run:
        return 1 - f2b_probabilities[headway_run_index]

    else:
        failure_probability = 1

        # relative indexing in the list of feasible runs in
        # chronological order for each trip.
        trip_headway_run_relative_index = data.feasible_runs_by_trip[trip_id].index(
            headway_run
        )
        trip_boarded_run_relative_index = data.feasible_runs_by_trip[trip_id].index(
            boarded_run
        )

        for missed_run in data.feasible_runs_by_trip[trip_id][
            trip_headway_run_relative_index:trip_boarded_run_relative_index
        ]:
            missed_run_index = data.runs.index(missed_run)
            failure_probability = (
                f2b_probabilities[missed_run_index] * failure_probability
            )
        return (1 - f2b_probabilities[boarded_run_index]) * failure_probability


if __name__ == "__main__":
    start_time = time()
    origin_station = "CHL"
    destination_stations = [
        "AUB",
        "ETO",
        "DEF",
        "NAP",
        "NAU",
        "NAV",
        "RUE",
        "CRO",
        "VES",
        "PEC",
        "GER",
    ]
    date = "03/02/2020"

    parameters = {
        "CHL": {"distribution": "cauchy", "parameters": {"loc": 94.8, "scale": 17.9}},
        "AUB": {
            "distribution": "lognorm",
            "parameters": {"s": 0.379, "loc": 16.6, "scale": 89.7},
        },
        "ETO": {
            "distribution": "chi2",
            "parameters": {"df": 16.0, "loc": 29.9, "scale": 7.2},
        },
        "DEF": {
            "distribution": "lognorm",
            "parameters": {"s": 0.31, "loc": -10.7, "scale": 102.0},
        },
        "NAP": {
            "distribution": "rayleigh",
            "parameters": {"loc": 0.636, "scale": 57.6},
        },
        "NAU": {
            "distribution": "lognorm",
            "parameters": {"s": 0.421, "loc": 26.4, "scale": 62.3},
        },
        "NAV": {
            "distribution": "gamma",
            "parameters": {"a": 2.83, "loc": 35.1, "scale": 24.3},
        },
        "RUE": {
            "distribution": "chi2",
            "parameters": {"df": 6.09, "loc": 13.9, "scale": 9.66},
        },
        "CRO": {
            "distribution": "rayleigh",
            "parameters": {"loc": -0.715, "scale": 44.3},
        },
        "VES": {
            "distribution": "gamma",
            "parameters": {"a": 1.56, "loc": 29.0, "scale": 33.7},
        },
        "PEC": {
            "distribution": "lognorm",
            "parameters": {"s": 0.579, "loc": 14.8, "scale": 39.6},
        },
        "GER": {
            "distribution": "gamma",
            "parameters": {"a": 6.8, "loc": 0.434, "scale": 13.7},
        },
    }

    initialization = True

    with open("../parameters.yml", "r") as file:
        param = safe_load(file)

    data = Data(date, origin_station, destination_stations)

    # Offline precomputations.

    likelihood_blocks = compute_likelihood_blocks(data, parameters)

    # Guess best uniform f2b proba.
    if initialization:
        initial_probability_range = linspace(0, 1, 50)
        objective_values = []
        for initial_probability in initial_probability_range:
            f2b_probabilities = [initial_probability for _ in range(len(data.runs))]
            objective_values.append(
                minus_log_likelihood_global(
                    f2b_probabilities, [0], data, likelihood_blocks
                )
            )

        min_value = min(objective_values)
        index = objective_values.index(min_value)
        print(f"best initial probability:{initial_probability_range[index]}")
        print(f"best initial likelihood: {objective_values[index]}")
