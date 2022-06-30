"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"

from time import time

from csv import writer
from matplotlib import pyplot
from numpy.random import rand
from scipy import optimize

from data import Data
from likelihood import (
    compute_likelihood_blocks,
    minus_log_likelihood_global,
    read_parameters,
)

PATHS = "data/AVL-AFC-2015/"
station_origin = "VINCENNES"
station_destination = "LA_DEFENSE_GRANDE_ARCHE"
date = "2015-03-16"
direction = "west"
distributed_speed = False
time_distribution = True


if __name__ == "__main__":
    # Data import and preparation.
    data = Data(PATHS, date, direction, station_origin, station_destination)

    # Offline precomputations.
    param = read_parameters()
    data = Data(PATHS, date, direction, station_origin, station_destination)

    # Offline precomputations.
    data.compute_feasible_runs()
    (
        precomputed_likelihood_blocks,
        walking_distance_entrance,
        walking_distance_exit,
    ) = compute_likelihood_blocks(data, param, distributed_speed, time_distribution)

    likelihood_headway_run = []
    likelihood_next_run = []
    likelihood_next_next_run = []
    likelihood_next_next_next_run = []
    nb_feasible_run = []
    for trip_id in data.AFC_df.index:
        first_feasible_run = data.feasible_runs_dict["first_feasible_run", trip_id]
        nb_feasible_run.append(
            data.feasible_runs_dict["last_feasible_run", trip_id] - first_feasible_run
        )
        likelihood_headway_run.append(
            precomputed_likelihood_blocks[
                trip_id, first_feasible_run, first_feasible_run
            ]
        )
        try:
            likelihood_next_run.append(
                precomputed_likelihood_blocks[
                    trip_id, first_feasible_run, first_feasible_run + 1
                ]
            )
        except KeyError:
            likelihood_next_run.append(0)
        try:
            likelihood_next_next_run.append(
                precomputed_likelihood_blocks[
                    trip_id, first_feasible_run, first_feasible_run + 2
                ]
            )
        except KeyError:
            likelihood_next_next_run.append(0)
        try:
            likelihood_next_next_next_run.append(
                precomputed_likelihood_blocks[
                    trip_id, first_feasible_run, first_feasible_run + 2
                ]
            )
        except KeyError:
            likelihood_next_next_next_run.append(0)

    pyplot.rcParams["figure.figsize"] = [20, 10]

    fig, ax = fig, axs = pyplot.subplots()
    ax.plot(likelihood_headway_run)
    ax.plot(likelihood_next_run)
    ax.plot(likelihood_next_next_run)
    ax.plot(likelihood_next_next_next_run)
    pyplot.savefig("../Note_F2B_Justine/figures/likelihood.png")
    # pyplot.show()
    # F2B probabilites initialization.
    f2b_probabilities_data = [rand() for i in range(data.mission_nbr)]

    # Likelihood optimization.
    # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
    iteration = [0]
    start_time = time()
    f2b_estimated = optimize.minimize(
        minus_log_likelihood_global,
        f2b_probabilities_data,
        method="Powell",
        tol=0.01,
        args=(iteration, data, precomputed_likelihood_blocks),
        bounds=[(0, 1) for i in range(data.mission_nbr)],
    )

    print(f"Optimization execution time: {time() - start_time:.2}s")

    with open("output/f2b_result_lognormal.csv", "w") as f:
        write = writer(f)
        write.writerow(f2b_estimated.x)

    # pyplot.plot(data.AVL_df[data.station_origin + "_departure"], f2b_estimated.x)
    # pyplot.show()
