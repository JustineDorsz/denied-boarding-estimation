"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"

from csv import writer
from time import time

from matplotlib import pyplot
from numpy.random import rand
from scipy import optimize

from data import Data
from likelihood import (
    compute_likelihood_blocks,
    minus_log_likelihood_global,
)

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


if __name__ == "__main__":

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
        tol=0.1,
        args=(iteration, data, likelihood_blocks),
        bounds=[(0, 1) for i in range(len(data.runs))],
    ).x

    print(f"Optimization execution time: {time() - start_time:.2}s")
    print(f2b_estimated)

    with open("../output/f2b_results_" + origin_station + ".txt", "w") as output_file:
        output_file.write(f2b_estimated)

    pyplot.plot(f2b_estimated)
    pyplot.show()
