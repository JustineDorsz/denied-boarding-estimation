"""Estimation of station access/egress distribution.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-06-29"

from time import time

from tqdm import tqdm


import f2b.db as db
from fitter import Fitter, get_common_distributions
from matplotlib import axes, pyplot
from pandas import Timestamp, concat
from sigfig import round
import scipy.stats
from statistics import mean, stdev
from yaml import dump

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002_test.db"


class Data:
    def __init__(
        self,
        station_estimation: str,
        dates: str,
        stations_origin: str,
    ):
        """Load trips with one feasible run from database,
        store in datadrame attribute self.trips.
        Compute egress time of each trip,
        store in dict attribute self.egress_times."""

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
                if date == self.dates[0] and station_origin == self.stations_origin[0]:
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
                self.trips = self.trips.drop(trip_id, axis=0)

    def _get_egress_times(self) -> None:
        self.egress_times = []
        for trip_id in self.feasible_run_by_trips:
            feasible_run = self.feasible_run_by_trips[trip_id]
            date = self.trips.at[trip_id, "date"]
            run_arrival_time = db.get_run_arrivals(
                self.db_path, date, feasible_run, [self.station_estimation]
            )[feasible_run, self.station_estimation]
            trip_egress_validation_time = self.trips.at[trip_id, "egress_time"]
            self.egress_times.append(
                (
                    Timestamp(trip_egress_validation_time) - Timestamp(run_arrival_time)
                ).seconds
            )


def plot_distributions_and_estimations(
    axs: axes.Axes,
    plot_position: list,
    egress_times_by_station: dict,
    fitted_laws: dict,
    station_estimation: str,
) -> None:
    "Plot the observed distribution and the estimated law of egress times in each station."

    colors_4am = ["#2a225d", "#c83576", "#ffbe7d", "#e9f98f", "#eaf7ff"]

    plot_row = plot_position[0]
    plot_column = plot_position[1]
    axs[plot_row, plot_column].hist(
        egress_times_by_station[station_estimation], bins=300, color=colors_4am[0]
    )
    axs[plot_row, plot_column].axis(ymin=0, ymax=250)
    axs[plot_row, plot_column].set_title(station_estimation)

    ax2 = axs[plot_row, plot_column].twinx()
    egress_times_by_station[station_estimation].sort()

    for law in best_law_info.keys():
        fitted_distrib = law
        fitted_distrib_function = eval("scipy.stats." + law)

    ax2label = fitted_distrib + "\n"
    for param_key in best_law_info[fitted_distrib].keys():
        ax2label += (
            str(param_key) + ":" + str(best_law_info[fitted_distrib][param_key]) + "\n"
        )
    ax2.plot(
        egress_times_by_station[station_estimation],
        fitted_distrib_function.pdf(
            egress_times_by_station[station_estimation],
            **best_law_info[fitted_distrib],
        ),
        color=colors_4am[1],
        label=ax2label,
    )
    ax2.axis(ymin=0, ymax=0.018)
    pyplot.legend(loc="upper right")
    pyplot.xlim([0, 300])


if __name__ == "__main__":
    start_time = time()

    write_output = False
    save_fig = True

    dates = ["04/02/2020"]

    stations = [
        "NAT",
        "LYO",
        "CHL",
        "AUB",
        "ETO",
        "DEF",
    ]

    distributions_to_test = [
        "cauchy",
        "chi2",
        "expon",
        "exponpow",
        "gamma",
        "lognorm",
        "norm",
        "powerlaw",
        "uniform",
    ]

    egress_times_by_station = {}

    row_nbr = 2
    column_nbr = 3
    fig, axs = pyplot.subplots(row_nbr, column_nbr, figsize=(14, 12))
    plot_row = 0
    plot_column = 0

    result_output_writing = {}

    print("Estimation running...")
    for station_estimation in tqdm(stations):
        station_estimation_position = stations.index(station_estimation)
        if station_estimation_position == 0:
            # Access station, we consider the trips in the other direction.
            # Access time distribution in one direction is supposed identical to
            # egress time distribution from the other direction.
            stations_origin = stations[1:]
        else:
            stations_origin = stations[:station_estimation_position]

        # Get list of egress times of all trips between stations_origin and
        # station_estimation.
        data = Data(station_estimation, dates, stations_origin)
        egress_times_by_station[station_estimation] = data.egress_times
        print(
            f"{station_estimation}: {len(data.egress_times)} egress_times, average = {mean(data.egress_times)}, standard deviation = {stdev(data.egress_times)}."
        )

        # Find best probability law fitting the egress time distribution.
        f = Fitter(
            egress_times_by_station[station_estimation],
            distributions=distributions_to_test,
        )
        f.fit()
        best_law_info = f.get_best()

        # Format estimated parameters, write distributions and parameters in fitted_laws.
        for law in best_law_info.keys():
            param_dict = {}
            for param_name in best_law_info[law].keys():
                best_law_info[law][param_name] = round(
                    best_law_info[law][param_name], sigfigs=3
                )
                param_dict.update({param_name: float(best_law_info[law][param_name])})
            result_output_writing.update(
                {station_estimation: {"distribution": law, "parameters": param_dict}}
            )

        # Plot egress time distribution and fitted law.
        if plot_row < column_nbr:
            plot_distributions_and_estimations(
                axs,
                [plot_row, plot_column],
                egress_times_by_station,
                best_law_info,
                station_estimation,
            )

        # Update plot position.
        plot_row = plot_row + (plot_column + 1) // column_nbr
        plot_column = (plot_column + 1) % column_nbr

    print(result_output_writing)
    if write_output:
        with open("f2b/parameters.yml", "w+") as parameters_file:
            dump(result_output_writing, parameters_file)

    print(f"Execution time: {time() - start_time}s.")

    if save_fig:
        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/fitted_egress_times_scaled.pdf"
        )
    else:
        pyplot.show()
