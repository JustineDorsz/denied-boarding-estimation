from matplotlib import pyplot

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood import (
    compute_likelihood_blocks,
)

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

    # data = Data(date, origin_station, destination_stations)

    # Offline precomputations.
    # likelihood_blocks = compute_likelihood_blocks(data, parameters)

    with open(
        "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/f2b_code/f2b/output/"
        + origin_station
        + ".txt",
        "r",
    ) as f2b_file:
        f2b_file_content = f2b_file.read()
        f2b_estimated = f2b_file_content.split()
        f2b_estimated = [float(f2b) for f2b in f2b_estimated]
        print(f2b_estimated)
