"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"


from time import time
from pandas import concat
from tqdm import tqdm

import f2b.db as db

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"

MAX_FEASIBLE_RUNS = 5


class Data:
    def __init__(
        self,
        date: str,
        origin_station: str,
        possible_destination_stations: list,
    ):
        """Loading trips and runs from database.
        param db_path: database location path
        param date: day of interest
        param station_origin: station of the fail-to-board estimation
        param possible_destination_stations: possible trip destinations"""

        self.date = date
        self.origin_station = origin_station
        self.possible_destination_stations = possible_destination_stations
        self.db_path = DB_PATH

        self._get_all_trips()
        self.trips["index"] = self.trips["trip_id"]
        self.trips = self.trips.set_index("trip_id")

        self._get_feasible_runs()

        self._concerned_trips_by_run()

        self._get_runs_time_info()

        self._order_runs_list()

        self._order_feasible_runs_lists()

        self._construct_feasible_run_pairs()

    def _get_all_trips(self):
        #  get trips by destination station
        for station_destination in self.possible_destination_stations:
            trips_to_destination = db.get_trips_filtered_by(
                self.db_path, self.date, self.origin_station, station_destination
            )
            if station_destination == self.possible_destination_stations[0]:
                self.trips = trips_to_destination
            else:
                self.trips = concat([self.trips, trips_to_destination])

    def _get_feasible_runs(self):
        self.feasible_runs_by_trip = {}
        self.runs = []
        self.destination_stations_by_run = (
            {}
        )  # Each run do not necessarily serves all destination stations.

        print("Get feasible runs...")
        too_long_trip = 0
        for trip_id in tqdm(self.trips.index):
            trip_destination = self.trips.at[trip_id, "egress_station"]
            feasible_runs = db.get_feasible_runs_for_one_trip(self.db_path, trip_id)
            self.feasible_runs_by_trip[trip_id] = feasible_runs

            # Remove trips with 0 feasible run.
            if not feasible_runs:
                self.trips = self.trips.drop(trip_id)

            if len(feasible_runs) > MAX_FEASIBLE_RUNS:
                too_long_trip += 1
                self.trips = self.trips.drop(trip_id)

            for feasible_run in feasible_runs:
                if feasible_run not in self.runs:
                    self.runs.append(feasible_run)
                try:
                    if (
                        trip_destination
                        not in self.destination_stations_by_run[feasible_run]
                    ):
                        self.destination_stations_by_run[feasible_run].append(
                            trip_destination
                        )
                except KeyError:
                    self.destination_stations_by_run[feasible_run] = [trip_destination]
        print(f"number of too long trips removed: {too_long_trip}")

    def _concerned_trips_by_run(self):
        self.concerned_trips_by_run = {}
        for trip_id in tqdm(self.trips.index):
            for run in self.feasible_runs_by_trip[trip_id]:
                try:
                    self.concerned_trips_by_run[run].append(trip_id)
                except KeyError:
                    self.concerned_trips_by_run[run] = [trip_id]

    def _get_runs_time_info(self):
        self.previous_run = {}
        self.runs_arrivals = {}
        self.runs_departures = {}
        stations = [self.origin_station] + self.possible_destination_stations

        print("Get runs info...")
        for run in tqdm(self.runs):
            for destination_station in self.destination_stations_by_run[run]:
                previous_run = db.get_previous_run(
                    self.db_path,
                    self.date,
                    run,
                    self.origin_station,
                    destination_station,
                )
                self.previous_run[run, destination_station] = previous_run

                run_arrivals = db.get_run_arrivals(
                    self.db_path, self.date, run, stations
                )

                run_departures = db.get_run_departures(
                    self.db_path, self.date, run, stations
                )
                self.runs_arrivals.update(run_arrivals)
                self.runs_departures.update(run_departures)

    def _order_runs_list(self):
        runs_departure_time = [
            self.runs_departures[run, self.origin_station] for run in self.runs
        ]
        zipped_lists = zip(runs_departure_time, self.runs)
        sorted_zipped_lists = sorted(zipped_lists)
        runs_ordered = [run for _, run in sorted_zipped_lists]
        self.runs = runs_ordered

    def _order_feasible_runs_lists(self):
        print("Order feasible run lists...")
        for trip_id in tqdm(self.trips.index):
            feasible_runs_list = self.feasible_runs_by_trip[trip_id]
            feasible_runs_departure_time = [
                self.runs_departures[run, self.origin_station]
                for run in feasible_runs_list
            ]
            zipped_lists = zip(feasible_runs_departure_time, feasible_runs_list)
            sorted_zipped_lists = sorted(zipped_lists)
            feasible_runs_list_ordered = [run for _, run in sorted_zipped_lists]
            self.feasible_runs_by_trip[trip_id] = feasible_runs_list_ordered

    def _construct_feasible_run_pairs(self):
        self.headway_boarded_run_pair_by_trip = {}

        print("Construct feasible run pairs...")
        for trip_id in tqdm(self.trips.index):
            for headway_run_index in range(0, len(self.feasible_runs_by_trip[trip_id])):
                for boarded_run_index in range(
                    headway_run_index, len(self.feasible_runs_by_trip[trip_id])
                ):
                    headway_run = self.feasible_runs_by_trip[trip_id][headway_run_index]
                    boarded_run = self.feasible_runs_by_trip[trip_id][boarded_run_index]
                    try:
                        self.headway_boarded_run_pair_by_trip[trip_id].append(
                            (headway_run, boarded_run)
                        )
                    except KeyError:
                        self.headway_boarded_run_pair_by_trip[trip_id] = [
                            (headway_run, boarded_run)
                        ]


if __name__ == "__main__":
    start_time = time()
    origin_station = "VIN"
    destination_stations = ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"]
    date = "04/02/2020"

    data = Data(date, origin_station, destination_stations)

    print(f"Data construction {time() - start_time}s.")
