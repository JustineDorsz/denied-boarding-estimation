"""Estimation of station access/egress distribution.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-06-29"

from time import time

from tqdm import tqdm


import f2b.db as db
from fitter import Fitter, get_common_distributions
from matplotlib import pyplot
from pandas import Timestamp, concat
from sigfig import round
import scipy.stats
from yaml import dump

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"

WRITE_OUTPUT = True


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
        self.egress_times = []
        for trip_id in self.feasible_run_by_trips:
            feasible_run = self.feasible_run_by_trips[trip_id]
            date = self.trips.at[trip_id, "date"]
            run_arrival_time = db.get_run_arrivals(
                self.db_path, date, feasible_run, [self.station_estimation]
            )[date, feasible_run, self.station_estimation]
            trip_egress_validation_time = self.trips.at[trip_id, "egress_time"]
            self.egress_times.append(
                (
                    Timestamp(trip_egress_validation_time) - Timestamp(run_arrival_time)
                ).seconds
            )


if __name__ == "__main__":
    start_time = time()
    dates = ["03/02/2020", "04/02/2020", "05/02/2020"]

    stations = [
        "CHL",
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

    egress_times_by_station = {}

    fig, axs = pyplot.subplots(3, 4)
    plot_row = 0
    plot_column = 0

    fitted_laws = {}

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

        # Find best probability law fitting the egress time distribution.
        f = Fitter(
            egress_times_by_station[station_estimation],
            distributions=get_common_distributions(),
        )
        f.fit()
        best_law_info = f.get_best()
        fitted_laws[station_estimation] = best_law_info

        for law in best_law_info.keys():
            fitted_distrib = law
            fitted_distrib_function = eval("scipy.stats." + law)
            for param_name in best_law_info[law].keys():
                best_law_info[law][param_name] = round(
                    best_law_info[law][param_name], sigfigs=3
                )

        # Plot egress time distribution and fitted law.
        axs[plot_row, plot_column].hist(
            egress_times_by_station[station_estimation], bins=500
        )
        axs[plot_row, plot_column].set_title(station_estimation)
        ax2 = axs[plot_row, plot_column].twinx()
        egress_times_by_station[station_estimation].sort()
        ax2.plot(
            egress_times_by_station[station_estimation],
            fitted_distrib_function.pdf(
                egress_times_by_station[station_estimation],
                **best_law_info[fitted_distrib],
            ),
            color="red",
            label=fitted_distrib + " \n" + str(best_law_info[fitted_distrib]),
        )
        pyplot.legend(loc="upper right")
        pyplot.xlim([0, 300])

        plot_row = plot_row + (plot_column + 1) // 4
        plot_column = (plot_column + 1) % 4

    if WRITE_OUTPUT:
        with open("../parameters.yml", "w+") as parameters_file:
            # dump(
            #     {station_estimation: best_law}, parameters_file, default_flow_style=False
            # )
            # parameters_file.write(dump(fitted_laws))
            parameters_file.write(str(fitted_laws))
            parameters_file.close()

    print(str(fitted_laws))
    print(f"Execution time: {time() - start_time}s.")
    pyplot.show()
