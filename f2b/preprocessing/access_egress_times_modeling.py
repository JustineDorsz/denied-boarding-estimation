"""
Estimation of fail-to-board probabilities (also referred to as delayed boarding 
probabilities) by train run from AFC and AVL data.

Estimation of distributions and parameters modeling access and egress times 
distributions in different stations.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-26"

import f2b.db.db as db
from fitter import Fitter
from pandas import concat
from sigfig import round
from yaml import safe_load, safe_dump

DISTRIBUTIONS = [
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


def get_upstream_stations_on_same_branch(
    reference_station: str, direction: int
) -> list:
    """Get the stations upstream a given station in a given direction on the same
    branch of the line RER A.

    Args:
        - reference_station(str): code of the reference station
        - direction(int): the direction in which we look for upstream stations
        1 for east, 2 for west

    Return:
        list: codes of upstream stations
    """
    # TODO: assert direction in [1,2]

    with open("parameters/idfm_codes.yml", "r") as idfm_info_file:
        idfm_code_file = safe_load(idfm_info_file)
        stations_order = idfm_code_file["station_code_order"]

    upstream_stations = []
    reference_station_order = stations_order[reference_station]
    branch_code = reference_station_order // 100
    for station in stations_order:
        # station on central trunk
        if stations_order[station] // 100 == branch_code:

            # station upstream
            if direction == 1:
                if stations_order[station] > reference_station_order:
                    upstream_stations.append(station)
            else:
                if stations_order[station] < reference_station_order:
                    upstream_stations.append(station)

    return upstream_stations


class ModelEgressTime:
    """Class to find a probability distribution and estimate the best parameters
    to model the distribution of egress times in a given transit station, in a given
    direction.

    A set of egress times observations is constituted from trips with one and only one
    associated run on a given day. The best distribution and its parameters are
    identified with fitter package, proceeding in two steps:
     - the parameters of several candidates distributions are estimated with a maximum
     likelihood principle,
     - the best distribution is chosen as least square error minimizer.

    Attributes:
        - station(str): code of the station in which we consider the egress times
        - direction(int): direction of the trips we consider the egress times.
        1 for eastward, 2 for westward
        - date(str): date of th trips we consider the egress times
        - origin_stations(list): set of origin stations for the trips
        - trips_df(DataFrame): information of the trips forming the observation,
        in particular egress time of each trip
        - best_distribution(str): name of probability distribution fitting the observed
         egress time distribution
        - best_parameters(dict): name and values of the estimated parameters associated
         to the best distributions

    Methods:
        - fit_distribution(): find best probability distribution and corresponding
        parameters fitting the observed egress times distribution
        - write_fitted_results(path: str, file: str): write distribution and parameters
        in some file
    """

    def __init__(self, station: str, direction: int, date: str):
        self.station = station
        self.direction = direction
        self.date = date

        self.origin_stations = get_upstream_stations_on_same_branch(
            self.station, self.direction
        )
        # don't take NAP into account
        # if "NAP" in self.origin_stations:
        #     self.origin_stations.remove("NAP")

        # get trips for all origin stations
        for origin_station in self.origin_stations:
            trips_from_origin = db.get_trips_filtered_by(
                date=self.date,
                access_station=origin_station,
                egress_station=self.station,
            )
            if origin_station == self.origin_stations[0]:
                self.trips_df = trips_from_origin
            else:
                self.trips_df = concat([self.trips_df, trips_from_origin])
        self.trips_df["assigned_runs"] = self.trips_df["trip_id"].apply(
            db.get_assigned_runs_for_one_trip
        )
        self.trips_df["number_assigned_runs"] = self.trips_df["assigned_runs"].apply(
            len
        )
        # filter by number of assigned runs
        self.trips_df = self.trips_df[
            (self.trips_df["number_assigned_runs"] == 1)
        ].copy()
        self.trips_df["assigned_runs"] = self.trips_df["assigned_runs"].apply(
            lambda x: x[0]
        )

        # compute egress times
        self.trips_df["run_arrival_time"] = self.trips_df["assigned_runs"].apply(
            db.get_run_arrival_time_at_station, args=(self.station, date)
        )
        self.trips_df["egress_duration"] = (
            self.trips_df["egress_time"] - self.trips_df["run_arrival_time"]
        ).dt.total_seconds()

    def fit_distribution(self) -> None:
        """Find best probability distribution and corresponding parameters fitting
        the observed egress times distribution. For each distribution among
        DISTRIBUTIONS, the parameters are estimated with maximum likelihood principle.
        The best distribution is chosen as least square error minimizer. The best
        distribution and parameters are stored as attributes."""

        f = Fitter(
            self.trips_df["egress_duration"],
            distributions=DISTRIBUTIONS,
        )
        f.fit()
        best_distrib_info = f.get_best()

        # format estimated parameters.
        for distribution in best_distrib_info:
            self.best_distribution = distribution
            self.best_parameters = {}
            for param_name in best_distrib_info[distribution]:
                self.best_parameters[param_name] = float(
                    round(best_distrib_info[distribution][param_name], sigfigs=3)
                )

    def write_fitted_results(self, path: str, file: str) -> None:
        """Write fitted distribution name and parameters to a given .yml file."""
        try:
            with open(path + file, "r") as yml_file:
                cur_yml = safe_load(yml_file)
        except FileNotFoundError:
            with open(path + file, "a+") as yml_file:
                cur_yml = safe_load(yml_file)
                cur_yml = {}

        cur_yml[self.station] = {
            "distribution": self.best_distribution,
            "parameters": self.best_parameters,
        }

        with open(path + file, "w") as yml_file:
            safe_dump(cur_yml, yml_file)


class ModelAccessTime(ModelEgressTime):
    """Class to find a probability distribution and estimate the best parameters
    to model the distribution of access times in a given transit station, in a given
    direction.

    The modeling of access times relies on a symmetry argument between the access
    pathway for trips in a given direction and the egress pathway for trips in the
    opposite direction. Thus the access times observations is consituted from egress
    times observations of trips in the opposite direction, with one and only one
    associated run on a given day. The best distribution and its parameters are
    identified with fitter package, proceeding in two steps:
     - the parameters of several candidates distributions are estimated with a maximum
     likelihood principle,
     - the best distribution is chosen as least square error minimizer.

    Attributes:
        - station(str): code of the station in which we consider the access times
        - direction(int): direction of the trips we consider the egress times.
        1 for eastward, 2 for westward
        - date(str): date of th trips we consider the access times
        - origin_stations(list): set of origin stations for the trips
        - trips_df(DataFrame): information of the trips forming the observation,
        in particular egress time of each trip
        - best_distribution(str): name of probability distribution fitting the observed
         egress time distribution
        - best_parameters(dict): name and values of the estimated parameters associated
         to the best distributions

    Methods:
        - fit_distribution(): find best probability distribution and corresponding
        parameters fitting the observed egress times distribution
        - write_fitted_results(path: str, file: str): write distribution and parameters
        in some file
    """

    def __init__(self, station: str, direction: int, date: str):
        # We assume symmetry between access and egress walk paths in opposite
        # directions in a given station.
        opposite_direction = 2 if direction == 1 else 1
        super().__init__(station, opposite_direction, date)
