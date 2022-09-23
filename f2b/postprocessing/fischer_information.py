from statistics import mean, median, quantiles, stdev
from time import time

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood import (
    compute_likelihood_blocks,
    log_likelihood_global,
)
from matplotlib import pyplot
from numdifftools import Gradient
from numpy import array, linalg
from pandas import DataFrame, read_csv
from yaml import safe_load

from output_analysis import load_estimated_f2b


def compute_fischer_matrix(
    origin_station: str,
    destination_stations: list,
    date: str,
    number_of_trips: int,
):

    f2b_estimated = load_estimated_f2b(origin_station, False)

    with open(f"f2b/parameters_{origin_station}.yml") as file:
        parameters = safe_load(file)

    data = Data(date, origin_station, destination_stations)

    data.trips = data.trips.head(number_of_trips)
    likelihood_blocks = compute_likelihood_blocks(data, parameters)

    start = time()

    estimated_score_vector = (
        1
        / data.trips.shape[0]
        * array(Gradient(log_likelihood_global)(f2b_estimated, data, likelihood_blocks))
    )

    estimated_info_matrix = array(
        [
            [
                estimated_score_vector[i] * estimated_score_vector[j]
                for i in range(len(estimated_score_vector))
            ]
            for j in range(len(estimated_score_vector))
        ]
    )

    print(f"Computation time: {time() - start}s.")

    DataFrame(estimated_info_matrix).to_csv(
        "f2b/output/fischer_info_" + origin_station + ".csv", header=None, index=None
    )


if __name__ == "__main__":

    origin_station = "NAT"
    destination_stations = ["LYO", "CHL", "AUB", "ETO", "DEF"]

    date = "04/02/2020"
    fischer_info = read_csv(
        "f2b/output/fischer_info_" + origin_station + ".csv", sep=",", header=None
    ).values

    diago_terms = [fischer_info[i, i] for i in range(fischer_info.shape[0])]
    print(diago_terms)

    w, v = linalg.eig(fischer_info)
    print(w)

    # mle_variance = linalg.inv( fischer_info)
    # print(mle_variance)
    # diago_terms = [mle_variance[i, i] for i in range(mle_variance.shape[0])]
    # print(mean(diago_terms))
    # print(stdev(diago_terms))
    # print(median(diago_terms))
