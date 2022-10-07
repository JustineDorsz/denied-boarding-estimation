"""Estimation of station access/egress distribution.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-06-29"

from math import sqrt

import f2b.db.db as db
from matplotlib import pyplot
from numdifftools import Hessian
from numpy import exp, linalg, log, mean, ndarray, zeros
from pandas import Timestamp, concat
from scipy import optimize
from scipy.stats import norm

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"

WRITE_OUPUT = False


class Data:
    def __init__(
        self,
        station_estimation: str,
        dates: str,
        direction: str,
        stations_origin: str,
    ):
        """Load trips with one feasible run from database,
        store in datadrame attribute self.trips.
        Compute egress time of each trip,
        store in dict attribute self.trips_egress_times."""

        self.dates = dates
        self.station_estimation = station_estimation
        self.stations_origin = stations_origin
        self.db_path = DB_PATH

        self._get_all_trips()
        self.trips = self.trips.set_index("id")

        self._filter_trips_with_one_feasible_run()

        self._get_egress_times()

    def _get_all_trips(self) -> None:
        for date in self.dates:
            for station_origin in self.stations_origin:
                trips_to_destination = db.get_trips_filtered_by(
                    self.db_path,
                    date,
                    station_origin,
                    self.station_estimation,
                )
                if date == self.dates[0]:
                    self.trips = trips_to_destination
                else:
                    self.trips = concat([self.trips, trips_to_destination])

    def _filter_trips_with_one_feasible_run(self) -> None:
        self.feasible_run_by_trips = {}
        for trip_id in self.trips.index:
            feasible_runs = db.get_feasible_runs_for_one_trip(self.db_path, trip_id)
            if len(feasible_runs) == 1:
                self.feasible_run_by_trips[trip_id] = feasible_runs[0]
            else:
                self.trips.drop(trip_id, axis=0)

    def _get_egress_times(self) -> None:
        self.trips_egress_times = {}
        for trip_id in self.feasible_run_by_trips:
            feasible_run = self.feasible_run_by_trips[trip_id]
            date = self.trips.at[trip_id, "date"]
            run_arrival_time = db.get_run_arrivals(
                self.db_path, date, feasible_run, [self.station_estimation]
            )[date, feasible_run, self.station_estimation]
            trip_egress_validation_time = self.trips.at[trip_id, "egress_time"]
            self.trips_egress_times[trip_id] = (
                Timestamp(trip_egress_validation_time) - Timestamp(run_arrival_time)
            ).seconds


def egress_time_log_likelihood(
    parameters: list, x: float, distribution: str, fixed_values: list = None
) -> float:
    if distribution == "normal":
        time_mean = parameters[0]
        time_sd = parameters[1]
        return log(norm.pdf(x, loc=time_mean, scale=time_sd))

    if distribution == "bivariate-normal":
        param_position = 0

        distance_mean = parameters[param_position]
        param_position += 1
        distance_sd = parameters[param_position]
        param_position += 1
        try:
            speed_mean = fixed_values[0]
        except TypeError:
            speed_mean = parameters[param_position]
            param_position += 1
        except IndexError:
            speed_mean = parameters[param_position]
            param_position += 1

        speed_sd = parameters[param_position]
        param_position += 1
        covariance = 0

        y_x = sqrt(distance_sd**2 + (x * speed_sd) ** 2 + -2 * x * covariance)
        phi = norm.pdf((x * speed_mean - distance_mean) / y_x)
        num = speed_mean * (distance_sd**2 - x * covariance) + distance_mean * (
            x * speed_sd**2 - covariance
        )
        denom = y_x**3
        return log(num * phi / denom)


def log_normal_distribution(x: float, parameters: list) -> float:
    composed_mean = parameters[0]
    composed_sd = parameters[1]
    frac = 1 / (composed_sd * x)
    phi = norm.pdf((log(x) - composed_mean) / composed_sd)
    return log(frac * phi)


def minus_sum_log_likelihood(
    parameters: list, data: Data, distribution: str, fixed_values: list = None
) -> float:
    total_minus_log_likelihood = 0
    for trip_id in data.trips_egress_times:
        egress_time = data.trips_egress_times[trip_id]
        total_minus_log_likelihood -= egress_time_log_likelihood(
            parameters, egress_time, distribution, fixed_values
        )
    print(total_minus_log_likelihood)
    return total_minus_log_likelihood


def plot_distributions(
    data: Data, distribution: str, parameters: list, fixed_values: list = None
):
    egress_time_list = list(data.trips_egress_times.values())

    egress_time_list.sort()

    if distribution == "bivariate-log-normal":
        egress_time_distribution = [
            exp(log_normal_distribution(x, parameters)) for x in egress_time_list
        ]

    else:
        egress_time_distribution = [
            exp(egress_time_log_likelihood(parameters, x, distribution, fixed_values))
            for x in egress_time_list
        ]
    fig, ax = pyplot.subplots()
    ax.hist(egress_time_list, bins=200)
    ax2 = ax.twinx()
    ax2.plot(egress_time_list, egress_time_distribution, color="red")
    pyplot.show()


def hessian_log_likelihood(
    param: list, egress_times: list, distribution: str, values: list = None
) -> ndarray:
    hess = Hessian(egress_time_log_likelihood)

    # initialization of a matrix of size param*param to store the sum of hessian contributions
    sum_hessian = zeros((len(param), len(param)))
    for x in egress_times:
        sum_hessian = sum_hessian + hess(param, x, distribution, values)

    return sum_hessian


if __name__ == "__main__":
    station_estimation = "VIN"
    stations_origin = ["DEF"]
    dates = ["03/02/2020"]

    data = Data(station_estimation, dates, "east", stations_origin)
    distribution = "bivariate-normal"
    print(len(data.trips_egress_times))

    if distribution == "normal":
        parameters_optimal = optimize.minimize(
            minus_sum_log_likelihood,
            (112, 47),
            args=(data, distribution),
            bounds=[(0, None), (0, None)],
        ).x
        print(parameters_optimal)
        # egress_times = list(data.trips_egress_times.values())
        # print(hessian_log_likelihood(parameters_optimal, egress_times, distribution))
        plot_distributions(data, distribution, parameters_optimal)

    if distribution == "bivariate-normal":
        fixed_values = []
        parameters_optimal = optimize.minimize(
            minus_sum_log_likelihood,
            (100, 20, 1.2, 0.1),
            args=(data, distribution, fixed_values),
            bounds=[(0, None), (0, None), (0, None), (0, None)],
        ).x
        print(parameters_optimal)
        egress_times = list(data.trips_egress_times.values())
        hessian_at_minimum = hessian_log_likelihood(
            parameters_optimal, egress_times, distribution, fixed_values
        )
        print(hessian_at_minimum)
        print(linalg.eigvals(hessian_at_minimum))
        plot_distributions(data, distribution, parameters_optimal, fixed_values)

    if distribution == "bivariate-log-normal":
        egress_time_list = list(data.trips_egress_times.values())
        composed_mean = mean(log(egress_time_list))
        composed_sd = sqrt(
            mean([(time - composed_mean) ** 2 for time in log(egress_time_list)])
        )
        parameters_optimal = [composed_mean, composed_sd]
        log_likelihood = -sum(
            [
                log_normal_distribution(egress_time, parameters_optimal)
                for egress_time in egress_time_list
            ]
        )
        print(log_likelihood)
        print(parameters_optimal)
        plot_distributions(data, distribution, parameters_optimal)


if WRITE_OUPUT:
    ...
