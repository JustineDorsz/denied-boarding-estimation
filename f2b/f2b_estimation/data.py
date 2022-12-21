"""
Estimation of fail-to-board probabilities (also referred to as delayed boarding 
probabilities) by train run from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-10"

from dataclasses import dataclass
from time import time
from typing import List

import f2b.db.db as db
from pandas import Timestamp, concat
from tqdm import tqdm


# The maximum number of assigned run supported for one trip.
# Trips with more than MAX_FEASIBLE_RUNS assigned runs are deleted from
# data.trips at initialization.
MAX_FEASIBLE_RUNS = 5


@dataclass
class Run:
    """The Run class contains spatio-temporal info of one run.

    Attributes:
        - run_code(str): unique run identifier
        - arrival_times(dict): arrival date and time (Timestamp format) at
        different stations, by station_code
        - departure_times(str): departure date and time (Timestamp format) at
        different stations, by station_code
        - previous_run(str): code of the run right before in chronological order
        - associated_trips(list): list of trips associated to the run
        - associated_trips_number(int): number of associated trips
    """

    run_code: str
    arrival_times: dict
    departure_times: dict
    previous_run: str
    associated_trips: List
    associated_trips_number: int


@dataclass
class Trip:
    """The Trip class contains spatio-temporal info of one trip.

    Attributes:
        - trip_id(str): unique trip identifier
        - access_time(Timestamp): access validation date and time
        - access_station(str): code of the access station
        - egress_time(Timestamp): egress validation date and time
        - egress_station(str): code of the egress station
        - associated_runs(list): run_code of assigned runs to the trip,
        ie runs possibly boarded.
        - associated_run_number(int): number of associated runs
    """

    trip_id: str
    access_time: Timestamp
    access_station: str
    egress_time: Timestamp
    egress_station: str
    associated_runs: List
    associated_runs_number: int


class Data:
    """Trips and runs at given date, between one origin and several destination stations.

    The Data class stores the trips and their assigned runs at a given date,
    from one origin station to some possible destination stations.
    The trips, runs and trips to runs info are loaded from a database containing
    AFC and AVL records.
    The matching runs to trips is computed during initialization.


    Attributes:
        - date(str): date of the trips and runs
        - origin_station(str): code of the origin station of the trips
        - destination_stations(list): codes of the destination stations of the trips
        - direction(int): 1 eastwards, 2 westwards
        - trips(dict): dictionnary of Trip objects with trip_id key. We select trips
        between the origin and the possible destination stations of the given day,
        filtering with reasonable duration (< 3 hours, > 2 minutes), and with at least
        one associated run.
        Trips with more than MAX_FEASIBLE_RUNS assigned runs are deleted from
        data.trips at initialization.
        - runs(dict): dictionnary of Run objects by run_code. We fill the dictionnary
        with every run of the given day concerned with at least one trip in self.trips.
        - runs_chronological_order(list): the runs ordered by chronological order
        by run_code
        - runs_number(int): number of runs
        - db_path(str): location path of the database



    """

    def __init__(
        self,
        date: str,
        origin_station: str,
        possible_destination_stations: list,
        direction: int,
    ):
        """Load trips and runs from database, format data, construct trips dictionnary
         of Trip and runs of Run. Compute runs to trip assignment.

        Args:
            - db_path(str): database location path
            - date(str): date of the day of interest
            - station_origin: code of the origin station
            - possible_destination_stations(list): codes of the possible trip destinations
            - direction(int): 1 eastwards, 2 westwards
        """

        self.date = date
        self.origin_station = origin_station
        self.possible_destination_stations = possible_destination_stations
        self.direction = direction

        self._construct_trips()
        self._construct_runs()
        self._sort_runs_chronological_order()

    def _construct_trips(self):
        """Construct the attribute trips from trips of the database."""

        # get trips for all destination stations
        # and store in common DataFrame trips
        for station_destination in self.possible_destination_stations:
            trips_to_destination = db.get_trips_filtered_by(
                date=self.date,
                access_station=self.origin_station,
                egress_station=station_destination,
            )
            if station_destination == self.possible_destination_stations[0]:
                trips_dataframe = trips_to_destination
            else:
                trips_dataframe = concat([trips_dataframe, trips_to_destination])

        # get assigned runs for each trips
        self.trips = {}
        trips_dataframe["assigned_runs"] = trips_dataframe["trip_id"].apply(
            db.get_assigned_runs_for_one_trip
        )
        trips_dataframe["number_assigned_runs"] = trips_dataframe[
            "assigned_runs"
        ].apply(len)

        # filter by number of assigned runs
        trips_dataframe = trips_dataframe[
            (trips_dataframe["number_assigned_runs"] >= 1)
            & (trips_dataframe["number_assigned_runs"] <= MAX_FEASIBLE_RUNS)
        ].copy()

        # construct trips
        print("Trips construction...")
        for trip_id in tqdm(trips_dataframe["trip_id"]):
            self.trips[trip_id] = Trip(
                *list(
                    (trips_dataframe.loc[trips_dataframe["trip_id"] == trip_id]).values[
                        0
                    ]
                )
            )

    def _construct_runs(self):
        """Get all runs serving the origin station of the study of the given day.
        Construct the attribute runs from the assigned runs stored in trips.
        For now runs of the same operating day but after midnight are not taken
        into account."""

        # get all run codes
        run_codes = db.get_runs_filtered_by(
            date=self.date,
            direction=self.direction,
            served_station_departure=self.origin_station,
            time_slot_end="23:59:59",
        )

        # initialize list of runs
        self.runs = {}
        print("Runs construction...")
        for run_code in tqdm(run_codes):

            # fill with arrival and departure time at origin_station
            self.runs[run_code] = Run(
                run_code,
                {
                    self.origin_station: db.get_run_arrival_time_at_station(
                        run_code,
                        self.origin_station,
                        self.date,
                    )
                },
                {
                    self.origin_station: db.get_run_departure_time_at_station(
                        run_code,
                        self.origin_station,
                        self.date,
                    )
                },
                None,
                [],
                0,
            )

        # fill runs with trip dependent info
        for trip_id in self.trips:
            for run_code in self.trips[trip_id].associated_runs:

                # store run -> trip association
                self.runs[run_code].associated_trips.append(trip_id)
                self.runs[run_code].associated_trips_number += 1

                # get and store previous run
                # TODO: on the central trunk, does not depend on origin and
                # destination stations.
                if not self.runs[run_code].previous_run:
                    self.runs[run_code].previous_run = db.get_previous_run(
                        self.date,
                        run_code,
                        self.trips[trip_id].access_station,
                        self.trips[trip_id].egress_station,
                    )

                # fill with arrival info at egress station of each associated trip
                if (
                    self.trips[trip_id].egress_station
                    not in self.runs[run_code].arrival_times
                ):
                    arrival_time_destination_station = (
                        db.get_run_arrival_time_at_station(
                            run_code,
                            self.trips[trip_id].egress_station,
                            self.date,
                        )
                    )
                    self.runs[run_code].arrival_times[
                        self.trips[trip_id].egress_station
                    ] = arrival_time_destination_station

        self.runs_number = len(self.runs)

    def _sort_runs_chronological_order(self):
        """Fill attribute self.runs_chronological_order with run_codes ordered
        by chronological departure time from the origin station."""

        runs_departure_times = [
            self.runs[run_code].departure_times[self.origin_station]
            for run_code in self.runs
        ]
        zipped_lists = zip(runs_departure_times, self.runs.keys())
        sorted_zipped_lists = sorted(zipped_lists)
        self.runs_chronological_order = [
            run_code for _, run_code in sorted_zipped_lists
        ]

        # sort the list of assigned runs of each trip in chrono order
        for trip_id in self.trips:
            indiv_runs_departure_times = [
                self.runs[run_code].departure_times[self.origin_station]
                for run_code in self.trips[trip_id].associated_runs
            ]
            zipped_lists = zip(
                indiv_runs_departure_times, self.trips[trip_id].associated_runs
            )
            sorted_zipped_lists = sorted(zipped_lists)
            self.trips[trip_id].associated_runs = [
                run_code for _, run_code in sorted_zipped_lists
            ]
