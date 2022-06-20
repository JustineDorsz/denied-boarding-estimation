"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-02-11"

from time import time

from math import sqrt
from matplotlib import pyplot
from numpy import exp, linspace, log
from scipy.integrate import quad
from scipy.stats import norm
from tqdm import tqdm
from yaml import safe_load

from data import Data

# ---------------------------------------------------------------------------------------
#
#                                      Parameters
#
# ---------------------------------------------------------------------------------------


def read_parameters() -> dict:
    """Read parameters from file and return a dictionnary."""
    with open("parameters.yml", "r") as file:
        parameters = safe_load(file)

    return parameters


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


# ---------------------------------------------------------------------------------------
#
#                                  Offline computations
#
# ---------------------------------------------------------------------------------------


def compute_likelihood_blocks(
    data: Data, param: dict, distributed_speed: bool, time_distribution: bool
) -> dict:
    """Compute likelihood blocks for each couple of feasible headway and
    run per trip, and store the result in a dictionnary with tuple keys."""

    # Dictionnary with tuple index.
    precomputed_blocks_dict = {}

    walked_distance_entrance = []
    walked_distance_exit = []

    print("Computing blocks...")
    for trip_id in tqdm(data.AFC_df.index):
        first_feasible_run = data.feasible_runs_dict["first_feasible_run", trip_id]
        last_feasible_run = data.feasible_runs_dict["last_feasible_run", trip_id]

        trip_first_possible_run = True
        for boarded_run in range(first_feasible_run, last_feasible_run + 1):
            for headway_run in range(first_feasible_run, boarded_run + 1):

                if time_distribution:

                    (
                        new_block,
                        walked_distance_O_upper_bound,
                        walked_distance_destination,
                    ) = compute_one_likelihood_block_time_distribution_lognormal(
                        data, param, trip_id, boarded_run, headway_run
                    )
                    precomputed_blocks_dict[
                        trip_id, headway_run, boarded_run
                    ] = new_block

                elif distributed_speed:

                    new_block = compute_one_likelihood_block_distributed(
                        data, param, trip_id, boarded_run, headway_run
                    )
                    precomputed_blocks_dict[
                        trip_id, headway_run, boarded_run
                    ] = new_block

                else:
                    (
                        new_block,
                        walked_distance_O_upper_bound,
                        walked_distance_destination,
                    ) = compute_one_likelihood_block(
                        data, param, trip_id, boarded_run, headway_run
                    )
                    precomputed_blocks_dict[
                        trip_id, headway_run, boarded_run
                    ] = new_block

                if trip_first_possible_run:
                    walked_distance_entrance.append(walked_distance_O_upper_bound)
                    walked_distance_exit.append(walked_distance_destination)
                    trip_first_possible_run = False

    return precomputed_blocks_dict, walked_distance_entrance, walked_distance_exit


def compute_one_likelihood_block(
    data: Data, param: dict, trip_id: int, boarded_run: int, headway_run: int
) -> float:
    """Compute and return likelihood terms independant from f2b
    probabilities for trip_id."""

    station_entry_time = data.AFC_df.loc[trip_id, "H_O"]
    station_exit_time = data.AFC_df.loc[trip_id, "H_D"]

    walked_distance_O_upper_bound = (
        param["walking_speed_mean"]
        * (
            data.AVL_df.loc[headway_run, data.station_origin + "_departure"]
            - station_entry_time
        ).total_seconds()
    )

    # Station entrance before the first run of the day.
    if headway_run == 0:
        walked_distance_O_lower_bound = 0

    # Station entrance after the beginning of the headway.
    elif (
        data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
        <= station_entry_time
    ):
        walked_distance_O_lower_bound = 0

    # Station entrance before the beginning of the headway.
    else:
        walked_distance_O_lower_bound = (
            param["walking_speed_mean"]
            * (
                data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
                - station_entry_time
            ).total_seconds()
        )

    walk_distance_diff = gaussian_CDF(
        walked_distance_O_upper_bound,
        param["gaussian"]["mean_access"],
        param["gaussian"]["std_access"],
    ) - gaussian_CDF(
        walked_distance_O_lower_bound,
        param["gaussian"]["mean_access"],
        param["gaussian"]["std_access"],
    )

    walked_distance_destination = (
        param["walking_speed_mean"]
        * (
            station_exit_time
            - data.AVL_df.loc[boarded_run, data.station_destination + "_arrival"]
        ).total_seconds()
    )
    walk_distance_exit = gaussian_PDF(
        walked_distance_destination,
        param["gaussian"]["mean_egress"],
        param["gaussian"]["std_egress"],
    )

    return (
        param["walking_speed_mean"] * walk_distance_exit * walk_distance_diff,
        walked_distance_O_upper_bound,
        walked_distance_destination,
    )


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


def compute_one_likelihood_block_time_distribution_gaussian(
    data: Data, param: dict, trip_id: int, boarded_run: int, headway_run: int
) -> float:
    """Compute and return likelihood terms independant from f2b probabilities
    for trip_id, relying on bivariate gaussian distributions for access
    and egress time."""

    station_entry_time = data.AFC_df.loc[trip_id, "H_O"]
    station_exit_time = data.AFC_df.loc[trip_id, "H_D"]

    access_time_upper_bound = (
        data.AVL_df.loc[headway_run, data.station_origin + "_departure"]
        - station_entry_time
    ).total_seconds()

    # Station entrance before the first run of the day.
    if headway_run == 0:
        access_time_lower_bound = 0

    else:
        access_time_lower_bound = max(
            0,
            (
                data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
                - station_entry_time
            ).total_seconds(),
        )

    access_time_diff = bivariate_gaussian_CDF(
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

    egress_time = (
        station_exit_time
        - data.AVL_df.loc[boarded_run, data.station_destination + "_arrival"]
    ).total_seconds()

    egress_time_distribution = bivariate_gaussian_PDF(
        egress_time,
        param["gaussian"]["mean_egress"],
        param["gaussian"]["std_egress"],
        param["walking_speed_mean"],
        param["gaussian"]["std_walking_speed_egress"],
        param["gaussian"]["covariance_egress"],
    )

    return (
        egress_time_distribution * access_time_diff,
        access_time_upper_bound,
        egress_time,
    )


def compute_one_likelihood_block_time_distribution_lognormal(
    data: Data, param: dict, trip_id: int, boarded_run: int, headway_run: int
) -> float:
    """Compute and return likelihood terms independant from f2b probabilities
    for trip_id, relying on lognormal distributions for access and egress time."""

    station_entry_time = data.AFC_df.loc[trip_id, "H_O"]
    station_exit_time = data.AFC_df.loc[trip_id, "H_D"]

    access_time_upper_bound = (
        data.AVL_df.loc[headway_run, data.station_origin + "_departure"]
        - station_entry_time
    ).total_seconds()

    # Station entrance before the first run of the day.
    if headway_run == 0:
        access_time_lower_bound = 0

    else:
        access_time_lower_bound = (
            data.AVL_df.loc[headway_run - 1, data.station_origin + "_departure"]
            - station_entry_time
        ).total_seconds()

    # access_time_lower_bound = max(16, access_time_lower_bound)
    # access_time_upper_bound = max(16, access_time_upper_bound)

    access_time_diff = log_normal_quotient_CDF(
        access_time_upper_bound,
        param["log_normal"]["mean_access_time"],
        param["log_normal"]["std_access_time"],
    )

    if access_time_lower_bound > 0:
        access_time_diff -= log_normal_quotient_CDF(
            access_time_lower_bound,
            param["log_normal"]["mean_access_time"],
            param["log_normal"]["std_access_time"],
        )

    egress_time = (
        station_exit_time
        - data.AVL_df.loc[boarded_run, data.station_destination + "_arrival"]
    ).total_seconds()

    # egress_time = max(16, egress_time)

    egress_time_distribution = log_normal_quotient_PDF(
        egress_time,
        param["log_normal"]["mean_egress_time"],
        param["log_normal"]["std_egress_time"],
    )

    return (
        egress_time_distribution * access_time_diff,
        access_time_upper_bound,
        egress_time,
    )


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
    PATHS = "data/AVL-AFC-2015/"
    station_origin = "VINCENNES"
    station_destination = "LA_DEFENSE_GRANDE_ARCHE"
    date = "2015-03-16"
    direction = "west"
    distributed_speed = False
    time_distribution = True

    plot = True
    initialization = True

    param = read_parameters()
    data = Data(PATHS, date, direction, station_origin, station_destination)

    # Offline precomputations.
    data.compute_feasible_runs()

    (
        precomputed_likelihood_blocks,
        walking_distance_entrance,
        walking_distance_exit,
    ) = compute_likelihood_blocks(data, param, distributed_speed, time_distribution)

    with open("tests/blocks.txt", "w") as file:
        file.write(str((precomputed_likelihood_blocks)))

    with open("tests/access_time.txt", "w") as file:
        file.write(str((walking_distance_entrance)))

    with open("tests/egress_time.txt", "w") as file:
        file.write(str((walking_distance_exit)))

    # Plot walking distance distribution from data, and probability exponential law.
    if plot:
        n_bins = 100
        fig, axs = pyplot.subplots(1, 2)

        axs[0].hist(walking_distance_entrance, bins=n_bins, range=[0, 400])

        ax01 = axs[0].twinx()
        distance = linspace(0, 400, 400)
        ax01.plot(
            distance,
            [
                bivariate_gaussian_PDF(
                    y,
                    param["gaussian"]["mean_access"],
                    param["gaussian"]["std_access"],
                    param["walking_speed_mean"],
                    param["gaussian"]["std_walking_speed_access"],
                    param["gaussian"]["covariance_access"],
                )
                for y in distance
            ],
            color="orange",
        )

        axs[1].hist(walking_distance_exit, bins=n_bins, range=[0, 750])
        ax11 = axs[1].twinx()

        distance = linspace(0, 750, 750)
        ax11.plot(
            distance,
            [
                bivariate_gaussian_PDF(
                    y,
                    param["gaussian"]["mean_egress"],
                    param["gaussian"]["std_egress"],
                    param["walking_speed_mean"],
                    param["gaussian"]["std_walking_speed_egress"],
                    param["gaussian"]["covariance_egress"],
                )
                for y in distance
            ],
            color="orange",
        )

        pyplot.show()

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
