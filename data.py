"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"


from time import time
from pandas import concat
from tqdm import tqdm

import db

DB_PATH = "/home/justine/Cired/Data/AFC_AVL_2020_02/RERA_202002.db"


class Data:
    def __init__(
        self,
        date: str,
        direction: str,
        station_origin: str,
        possible_station_destination: list,
    ):
        """Loading trips and runs from database.
        param db_path: database location path
        param date: day of interest
        param station_origin: station of the fail-to-board estimation
        param possible_stations_destination: possible trip destinations"""

        self.date = date
        self.station_origin = station_origin
        self.possible_stations_destination = possible_station_destination
        self.db_path = DB_PATH

        #  loop on possible destinations
        for station_destination in possible_station_destination:
            trips_to_destination = db.get_trips_filtered_by(
                self.db_path, date, station_origin, station_destination
            )
            if station_destination == possible_station_destination[0]:
                self.trips = trips_to_destination
            else:
                self.trips = concat([self.trips, trips_to_destination])
        self.trips = self.trips.set_index("id")

        self.feasible_runs_by_trip = {}
        self.concerned_runs = []
        for trip_id in self.trips.index:
            feasible_runs = db.get_feasible_runs_for_one_trip(self.db_path, trip_id)
            self.concerned_runs = list(set(self.concerned_runs) | set(feasible_runs))
            self.feasible_runs_by_trip[trip_id] = feasible_runs

        self.runs_info = {}
        stations = [station_origin] + possible_station_destination
        for run in self.concerned_runs:
            new_run_info = db.get_run_info(self.db_path, date, run, stations)
            self.runs_info.update(new_run_info)

        self._construct_feasible_run_pairs()

    def _construct_feasible_run_pairs(self):
        self.headway_boarded_run_pair_by_trip = {}

        for trip_id in self.trips.index:

            # Get all possible pairs for headway run and boarded run.
            nbr_feasible_runs = len(self.feasible_runs_by_trip[trip_id])
            possible_run_pair = []
            for left_term_index in range(nbr_feasible_runs):
                for right_term_index in range(left_term_index, nbr_feasible_runs):
                    pair_left = self.feasible_runs_by_trip[trip_id][left_term_index]
                    pair_right = self.feasible_runs_by_trip[trip_id][right_term_index]

                    # Compare runs departure time of each possible pair
                    # to store in order (headway_run, boarded_run).
                    departure_time_left_term = self.runs_info[
                        self.date, pair_left, self.station_origin
                    ][1]
                    departure_time_right_term = self.runs_info[
                        self.date, pair_right, self.station_origin
                    ][1]
                    if (
                        departure_time_left_term
                        and departure_time_right_term
                        and (departure_time_left_term > departure_time_right_term)
                    ):
                        pair_left, pair_right = pair_right, pair_left

                    possible_run_pair.append((pair_left, pair_right))
            self.headway_boarded_run_pair_by_trip[trip_id] = possible_run_pair


if __name__ == "__main__":
    start_time = time()
    station_origin = "VIN"
    stations_destination = ["NAT", "LYO"]
    date = "03/02/2020"
    direction = "west"

    data = Data(date, direction, station_origin, stations_destination)
    print(data.trips.head(30))

    print(f"Data construction {time() - start_time}s.")
