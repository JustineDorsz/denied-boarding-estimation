"""
Estimation of fail-to-board probabilities from AFC and AVL data.
Likelihood function, gradient and hessian in linear time with recursive blocks.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-09-21"

from math import log
from time import time

import scipy.stats
from f2b.f2b_estimation.data import Data
from numpy import array, ndarray
from pandas import Timestamp
from tqdm import tqdm
from yaml import safe_load

# ---------------------------------------------------------------------------------------
#
#                                  Offline computations
#
# ---------------------------------------------------------------------------------------


def compute_egress_individual_likelihoods(data: Data, param: dict) -> dict:
    egress_individual_likelihood = {}

    print("(Offline) Compute egress individual likelihood...")
    for trip_id in tqdm(data.trips.index):
        destination_station = data.trips.at[trip_id, "egress_station"]
        station_egress_distrib_name = param[destination_station]["distribution"]
        station_egress_duration_distribution = eval(
            "scipy.stats." + station_egress_distrib_name
        )
        station_egress_duration_distribution_params = param[destination_station][
            "parameters"
        ]
        station_egress_time = Timestamp(data.trips.at[trip_id, "egress_time"])

        for run in reversed(data.feasible_runs_by_trip[trip_id]):
            boarded_run_arrival_dest_station = Timestamp(
                data.runs_arrivals[run, destination_station]
            )
            egress_duration = (
                station_egress_time - boarded_run_arrival_dest_station
            ).total_seconds()

            egress_proba = station_egress_duration_distribution.pdf(
                egress_duration, **station_egress_duration_distribution_params
            )
            egress_individual_likelihood[trip_id, run] = egress_proba

    return egress_individual_likelihood


def compute_access_individual_likelihoods(data: Data, param: dict) -> dict:
    access_individual_likelihood = {}

    print("(Offline) Compute access individual likelihood...")
    for trip_id in tqdm(data.trips.index):
        destination_station = data.trips.at[trip_id, "egress_station"]
        station_access_distrib_name = param[data.origin_station]["distribution"]
        station_access_duration_distribution = eval(
            "scipy.stats." + station_access_distrib_name
        )
        station_access_duration_distribution_params = param[data.origin_station][
            "parameters"
        ]
        station_access_time = Timestamp(data.trips.at[trip_id, "access_time"])

        for run in reversed(data.feasible_runs_by_trip[trip_id]):
            headway_run_departure_origin_station = Timestamp(
                data.runs_departures[run, data.origin_station]
            )

            access_duration_upper_bound = (
                headway_run_departure_origin_station - station_access_time
            ).total_seconds()

            access_duration_lower_bound = 0

            headway_previous_run = data.previous_run[run, destination_station]

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

            access_proba_difference = station_access_duration_distribution.cdf(
                access_duration_upper_bound,
                **station_access_duration_distribution_params,
            ) - station_access_duration_distribution.cdf(
                access_duration_lower_bound,
                **station_access_duration_distribution_params,
            )

            access_individual_likelihood[trip_id, run] = access_proba_difference

    return access_individual_likelihood


# ---------------------------------------------------------------------------------------
#
#                                  Online computations
#
# ---------------------------------------------------------------------------------------


def log_likelihood_global_with_egress_auxiliary_variables_updates(
    f2b_probabilities: ndarray,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
) -> float:

    log_likelihood_global = 0

    for trip_id in data.trips.index:
        individual_likelihood = 0

        for position, run in reversed(
            list(enumerate(data.feasible_runs_by_trip[trip_id]))
        ):
            run_index = data.runs.index(run)
            egress_auxiliary_variables[trip_id, run] = egress_individual_likelihoods[
                trip_id, run
            ] * (1 - f2b_probabilities[run_index])

            #  If not last possible run, add the term of recursive dependance.
            if position != len(data.feasible_runs_by_trip[trip_id]) - 1:
                next_run = data.feasible_runs_by_trip[trip_id][position + 1]
                egress_auxiliary_variables[trip_id, run] += (
                    f2b_probabilities[run_index]
                    * egress_auxiliary_variables[trip_id, next_run]
                )

            individual_likelihood += (
                access_individual_likelihoods[trip_id, run]
                * egress_auxiliary_variables[trip_id, run]
            )
        individual_likelihoods[trip_id] = individual_likelihood
        log_likelihood_global += log(individual_likelihood)

    return log_likelihood_global


def indiv_likelihood_and_auxiliary_egress_update_by_run(
    component: int,
    f2b_probabilities: ndarray,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
) -> None:
    run_interest = data.runs[component]
    for trip_id in data.concerned_trips_by_run[run_interest]:
        individual_likelihood = 0

        for position, run in reversed(
            list(enumerate(data.feasible_runs_by_trip[trip_id]))
        ):
            run_index = data.runs.index(run)
            egress_auxiliary_variables[trip_id, run] = egress_individual_likelihoods[
                trip_id, run
            ] * (1 - f2b_probabilities[run_index])

            #  If not last possible run, add the term of recursive dependance.
            if position != len(data.feasible_runs_by_trip[trip_id]) - 1:
                next_run = data.feasible_runs_by_trip[trip_id][position + 1]
                egress_auxiliary_variables[trip_id, run] += (
                    f2b_probabilities[run_index]
                    * egress_auxiliary_variables[trip_id, next_run]
                )

            individual_likelihood += (
                access_individual_likelihoods[trip_id, run]
                * egress_auxiliary_variables[trip_id, run]
            )
        individual_likelihoods[trip_id] = individual_likelihood


def differential_log_likelihood_global_by_component(
    component_diff: int,
    f2b_probabilities: ndarray,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
) -> float:
    diff_log_likelihood_global_by_component = 0
    run_diff = data.runs[component_diff]
    for trip_id in data.concerned_trips_by_run[run_diff]:

        # First update egress auxiliary variables for concerned trips,
        # with recursive relation in reverse order.
        for position, run in reversed(
            list(enumerate(data.feasible_runs_by_trip[trip_id]))
        ):
            run_index = data.runs.index(run)
            egress_auxiliary_variables[trip_id, run] = egress_individual_likelihoods[
                trip_id, run
            ] * (1 - f2b_probabilities[run_index])

            #  If not last possible run, add the term of recursive dependance.
            if position != len(data.feasible_runs_by_trip[trip_id]) - 1:
                next_run = data.feasible_runs_by_trip[trip_id][position + 1]
                egress_auxiliary_variables[trip_id, run] += (
                    f2b_probabilities[run_index]
                    * egress_auxiliary_variables[trip_id, next_run]
                )
        # Then update access auxiliary variables for concerned trips,
        # with recursive relation in direct order.
        for position, run in enumerate(data.feasible_runs_by_trip[trip_id]):

            access_auxiliary_variables[trip_id, run] = access_individual_likelihoods[
                trip_id, run
            ]
            if position != 0:
                previous_run = data.feasible_runs_by_trip[trip_id][position - 1]
                previous_run_index = data.runs.index(previous_run)
                access_auxiliary_variables[trip_id, run] += (
                    f2b_probabilities[previous_run_index]
                    * access_auxiliary_variables[trip_id, previous_run]
                )

        #  Directionnal diff of individual likelihood.
        if run_diff != data.feasible_runs_by_trip[trip_id][-1]:
            run_diff_index = data.feasible_runs_by_trip[trip_id].index(run_diff)
            next_run = data.feasible_runs_by_trip[trip_id][run_diff_index + 1]
            egress_contribution = egress_auxiliary_variables[trip_id, next_run]
        else:
            egress_contribution = 0

        if individual_likelihoods[trip_id] == 0:
            print(trip_id)
        diff_individual_likelihood = (
            access_auxiliary_variables[trip_id, run_diff]
            * (egress_contribution - egress_individual_likelihoods[trip_id, run_diff])
            / individual_likelihoods[trip_id]
        )

        diff_log_likelihood_global_by_component += diff_individual_likelihood
    return diff_log_likelihood_global_by_component


def gradient_log_likelihood_global(
    f2b_probabilities: ndarray,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    individual_log_likelihood_gradients: dict,
    data: Data,
) -> ndarray:
    gradient_log_likelihood_global = array([0.0 for _ in range(len(data.runs))])
    for trip_id in data.trips.index:
        for position, run in enumerate(data.feasible_runs_by_trip[trip_id]):

            individual_log_likelihood_gradients[trip_id, run] = 0.0
            run_index = data.runs.index(run)

            access_auxiliary_variables[trip_id, run] = access_individual_likelihoods[
                trip_id, run
            ]
            if position != 0:
                previous_run = data.feasible_runs_by_trip[trip_id][position - 1]
                previous_run_index = data.runs.index(previous_run)
                access_auxiliary_variables[trip_id, run] += (
                    f2b_probabilities[previous_run_index]
                    * access_auxiliary_variables[trip_id, previous_run]
                )
            if run != data.feasible_runs_by_trip[trip_id][-1]:
                next_run = data.feasible_runs_by_trip[trip_id][position + 1]
                egress_contribution = egress_auxiliary_variables[trip_id, next_run]
            else:
                egress_contribution = 0

            gradient_term = (
                access_auxiliary_variables[trip_id, run]
                * (egress_contribution - egress_individual_likelihoods[trip_id, run])
                / individual_likelihoods[trip_id]
            )

            individual_log_likelihood_gradients[trip_id, run] = gradient_term
            gradient_log_likelihood_global[run_index] += gradient_term

    return gradient_log_likelihood_global


def hessian_log_likelihood_global(
    f2b_probabilities: ndarray,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    individual_log_likelihood_gradients: dict,
    trip_id_restriction: float,
    data: Data,
) -> ndarray:
    hessian_log_likelihood_global = array(
        [[0.0 for _ in range(len(data.runs))] for _ in range(len(data.runs))]
    )
    if trip_id_restriction == 0:
        trips_loop = data.trips.index
    else:
        trips_loop = [trip_id_restriction]
        # for trip_id in trips_loop:
    for trip_id in data.trips.index:
        for position, first_run in enumerate(data.feasible_runs_by_trip[trip_id]):
            first_run_index = data.runs.index(first_run)
            hessian_log_likelihood_global[first_run_index, first_run_index] += (
                -individual_log_likelihood_gradients[trip_id, first_run] ** 2
            )
            auxiliary_proba = 1

            for second_run in data.feasible_runs_by_trip[trip_id][position + 1 :]:
                if second_run != data.feasible_runs_by_trip[trip_id][-1]:
                    next_run = data.feasible_runs_by_trip[trip_id][position + 2]
                    egress_contribution = (
                        egress_auxiliary_variables[trip_id, next_run]
                        - egress_individual_likelihoods[trip_id, second_run]
                    )
                else:
                    egress_contribution = -egress_individual_likelihoods[
                        trip_id, second_run
                    ]

                second_run_index = data.runs.index(second_run)
                hessian_log_likelihood_global[first_run_index, second_run_index] += (
                    egress_contribution
                    * auxiliary_proba
                    * access_auxiliary_variables[trip_id, first_run]
                    / individual_likelihoods[trip_id]
                    - individual_log_likelihood_gradients[trip_id, first_run]
                    * individual_log_likelihood_gradients[trip_id, second_run]
                )
                hessian_log_likelihood_global[
                    second_run_index, first_run_index
                ] = hessian_log_likelihood_global[first_run_index, second_run_index]
                auxiliary_proba *= f2b_probabilities[second_run_index]
    return hessian_log_likelihood_global


if __name__ == "__main__":
    start_time = time()
    origin_station = "VIN"
    destination_stations = ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"]
    date = "04/02/2020"

    with open(f"f2b/parameters/parameters_{origin_station}.yml") as file:
        parameters = safe_load(file)

    data = Data(date, origin_station, destination_stations)
    f2b_probabilities = array([0 for _ in range(len(data.runs))])

    access_individual_likelihoods = compute_access_individual_likelihoods(
        data, parameters
    )
    egress_individual_likelihoods = compute_egress_individual_likelihoods(
        data, parameters
    )
    individual_likelihoods = {}
    individual_likelihood_gradients = {}
    access_auxiliary_variables = {}
    egress_auxiliary_variables = {}

    start = time()
    log_likelihood = log_likelihood_global_with_egress_auxiliary_variables_updates(
        f2b_probabilities,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    print(f"One evaluation of likelihood: {time()-start}s.")

    start = time()
    gradient_log_likelihood_global = gradient_log_likelihood_global(
        f2b_probabilities,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_likelihood_gradients,
        data,
    )
    print(f"One evaluation of likelihood gradient: {time()-start}s.")

    start = time()
    hessian_log_likelihood_global = hessian_log_likelihood_global(
        f2b_probabilities,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_likelihood_gradients,
        data,
    )
    print(f"One evaluation of likelihood hessian: {time() - start}s.")
