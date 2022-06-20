"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"


from time import time


def create_connection(path):
    ...


def get_trips_filtered_by():
    ...


def get_feasible_runs_for_one_trip():
    ...


def get_run_infos():
    ...


class Data:
    def __init__(
        self,
        db_path: str,
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

        conn = create_connection(db_path)

        #  loop on possible destinations
        for station_destination in possible_station_destination:
            self.trips = get_trips_filtered_by(
                conn, date, station_origin, station_destination
            )  # concatenate for all destinations

        self.feasible_runs_by_trip = {}
        concerned_runs = []
        #  loop on trips
        trip_id = ...

        self.feasible_runs_by_trip[trip_id] = get_feasible_runs_for_one_trip(
            conn, trip_id
        )
        # add to concerned run list

        # loop on distinct runs
        for run in concerned_runs:
            self.run_infos = {}  # distinct runs from feasible_run_list
            stations = [station_origin] + possible_station_destination
            self.runs_infos = get_run_infos(conn, run, stations)


if __name__ == "__main__":
    start_time = time()
    PATHS = "data/AVL-AFC-2015/"
    station_origin = "VINCENNES"
    station_destination = "LA_DEFENSE_GRANDE_ARCHE"
    date = "2015-03-16"
    direction = "west"

    data = Data(PATHS, date, direction, station_origin, station_destination)

    print(f"Data construction {time() - start_time}s.")
