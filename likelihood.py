"""
Estimation of fail-to-board probabilities from AFC and AVL data.
Likelihood functions.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-02-11"

from time import time

from matplotlib import pyplot
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

    # Select with label index from trip_id would be better ?
    print(data.trips.at[trip_id, "access_time"])
    station_access_time = Timestamp(data.trips.at[trip_id, "access_time"])
    station_egress_time = Timestamp(data.trips.at[trip_id, "egress_time"])
    station_destination = data.trips.at[trip_id, "egress_station"]

    boarded_run_departure_origin_station = Timestamp(
        data.runs_info[data.date, boarded_run, data.station_origin][1]
    )
    boarded_run_arrival_dest_station = Timestamp(
        data.runs_info[data.date, boarded_run, data.station_origin][0]
    )

    access_time_upper_bound = (
        boarded_run_departure_origin_station - station_access_time
    ).total_seconds()

    #  TODO: get run immediately before headway run
    # Station entrance after the beginning of the headway.

    before_headway_run_departure_time = Timestamp(0)  # TODO: affect !
    if before_headway_run_departure_time - station_access_time < Timedelta(0):
        access_time_lower_bound = 0

    # Station entrance before the beginning of the headway.
    else:
        access_time_lower_bound = (
            before_headway_run_departure_time - station_access_time
        ).total_seconds()

    egress_time = (
        station_egress_time - boarded_run_arrival_dest_station
    ).total_seconds()

    if time_distribution_assumption == "gaussian":
        access_proba_difference = bivariate_gaussian_CDF(
            access_time_upper_bound,
            param["gaussian"]["mean_access"],
            param["gaussian"]["std_access"],
            param["walking_speed_mean"],
            param["gaussian"]["std_walking_speed_access"],
            param["gaussian"]["covariance_access"],
        ) - bivariate_gaussian_CDF(
            access_time_lower_bound,
            param["gaussian"]["mean_access"],
            param["gaussian"]["std_access"],
            param["walking_speed_mean"],
            param["gaussian"]["std_walking_speed_access"],
            param["gaussian"]["covariance_access"],
        )

        egress_proba = bivariate_gaussian_PDF(
            egress_time,
            param["gaussian"]["mean_egress"],
            param["gaussian"]["std_egress"],
            param["walking_speed_mean"],
            param["gaussian"]["std_walking_speed_egress"],
            param["gaussian"]["covariance_egress"],
        )

    if time_distribution_assumption == "log_normal":
        access_proba_diff = log_normal_quotient_CDF(
            access_time_upper_bound,
            param["log_normal"]["mean_access_time"],
            param["log_normal"]["std_access_time"],
        )

        if access_time_lower_bound > 0:
            access_proba_diff -= log_normal_quotient_CDF(
                access_time_lower_bound,
                param["log_normal"]["mean_access_time"],
                param["log_normal"]["std_access_time"],
            )

        egress_proba = log_normal_quotient_PDF(
            egress_time,
            param["log_normal"]["mean_egress_time"],
            param["log_normal"]["std_egress_time"],
        )

    return egress_proba * access_proba_difference


def compute_one_likelihood_block_distributed(
    data: Data, param: dict, trip_id: int, boarded_run: int, headway_run: int
) -> float:
    """Compute and return likelihood terms independant from f2b
    probabilities for trip_id, integrated over the speed distribution"""

    station_entry_time = data.AFC_df.loc[trip_id, "H_O"]
    station_exit_time = data.AFC_df.loc[trip_id, "H_D"]
    walked_time_O_upper_bound = (
        data.AVL_df.loc[headway_run, data.station_origin + "_departure"]
        - station_entry_time
    ).total_seconds()

    # Station entrance before the first run of the day.
    if headway_run == 0:
        walked_time_O_lower_bound = 0

    # Station entrance after the beginning of the headway.
    elif (
        data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
        <= station_entry_time
    ):
        walked_time_O_lower_bound = 0

    # Station entrance before the beginning of the headway.
    else:
        walked_time_O_lower_bound = (
            data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
            - station_entry_time
        ).total_seconds()

    walked_time_destination = (
        station_exit_time
        - data.AVL_df.loc[boarded_run, data.station_destination + "_arrival"]
    ).total_seconds()

    return quad(
        likelihood_block_integrand,
        0,
        1,
        args=(
            param,
            walked_time_O_upper_bound,
            walked_time_O_lower_bound,
            walked_time_destination,
        ),
    )[0]


def likelihood_block_integrand(
    w: float,
    param: dict,
    walked_time_O_upper_bound: float,
    walked_time_O_lower_bound: float,
    walked_time_destination: float,
):
    """Compute likelihood integrand term depending on integration variable w."""
    speed = norm.ppf(
        w, param["walking_speed_mean"], param["gaussian"]["std_walking_speed_access"]
    )

    walked_distance_O_upper_bound = speed * walked_time_O_upper_bound
    walked_distance_O_lower_bound = speed * walked_time_O_lower_bound

    walk_distance_diff = gaussian_CDF(
        walked_distance_O_upper_bound,
        param["gaussian"]["mean_access"],
        param["gaussian"]["std_access"],
    ) - gaussian_CDF(
        walked_distance_O_lower_bound,
        param["gaussian"]["mean_access"],
        param["gaussian"]["std_access"],
    )

    walked_distance_destination = speed * walked_time_destination

    walk_distance_exit = gaussian_PDF(
        walked_distance_destination,
        param["gaussian"]["mean_egress"],
        param["gaussian"]["std_egress"],
    )

    return speed * walk_distance_exit * walk_distance_diff


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

    # Initialize f2b proba for tests.
    if initialization:
        initial_probability_range = linspace(0, 1, 50)
        objective_values = []
        for initial_probability in initial_probability_range:
            f2b_probabilities = [initial_probability for i in range(data.mission_nbr)]
            start_time = time()
            objective_values.append(
                minus_log_likelihood_global(
                    f2b_probabilities, [0], data, precomputed_likelihood_blocks
                )
            )
            print(
                f"Objective function evaluation execution time : {time() - start_time:.2}s"
            )

        min_value = min(objective_values)
        index = objective_values.index(min_value)
        print(f"best initial probability:{initial_probability_range[index]}")
        print(f"best initial likelihood: {objective_values[index]}")
