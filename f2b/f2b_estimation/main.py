"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"

from csv import writer
from time import time

from matplotlib import pyplot
from scipy import optimize
from yaml import safe_load


from data import Data
import f2b
from likelihood import (
    compute_likelihood_blocks,
    minus_log_likelihood_global,
)

start_time = time()
origin_station = "NAT"
destination_stations = ["LYO", "CHL", "AUB", "ETO", "DEF"]

date = "04/02/2020"

with open(f"f2b/parameters_{origin_station}.yml") as file:
    parameters = safe_load(file)

if __name__ == "__main__":

    write_output = True

    data = Data(date, origin_station, destination_stations)

    # Offline precomputations.
    likelihood_blocks = compute_likelihood_blocks(data, parameters)
    f2b_probabilities_initial = [0.02 for _ in range(len(data.runs))]

    # Likelihood optimization.
    # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
    iteration = [0]
    start_time = time()
    f2b_estimated = optimize.minimize(
        minus_log_likelihood_global,
        f2b_probabilities_initial,
        method="Powell",
        tol=0.05,
        args=(iteration, data, likelihood_blocks),
        bounds=[(0, 1) for i in range(len(data.runs))],
    ).x

    print(f"Optimization execution time: {time() - start_time:.2}s")
    print(f2b_estimated)

    if write_output:
        with open(
            "f2b/output/f2b_results_" + origin_station + ".csv", "w"
        ) as output_file:
            writer = writer(output_file)
            writer.writerow(f2b_estimated)

    pyplot.plot(f2b_estimated)
    pyplot.show()
