"""Estimation of fail-to-board probabilities from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-03-04"


from time import time

from numpy import arange
from pandas import DataFrame, Timestamp, concat, read_csv, to_datetime
from tqdm import tqdm
from yaml import safe_load


class Data:
    def __init__(
        self,
        paths: dict,
        date: Timestamp,
        direction: str,
        station_origin: str,
        station_destination: str,
    ):
        """Loading and preparation of AVL and AFC data from csv files."""

        line_infos = self._read_line_infos()

        self.paths = paths
        self.year = date[0:4]
        self.month = date[5:7]
        self.day = date[8:10]

        self.station_origin = station_origin
        self.station_destination = station_destination
        self.station_origin_short = line_infos["station_name_full_to_short"][
            station_origin
        ]
        self.station_destination_short = line_infos["station_name_full_to_short"][
            station_destination
        ]
        self.direction = line_infos["directions"][direction]

        # Load and prepare AFC data.
        try:
            self._AFC_read_data()

        except FileNotFoundError:
            print("AFC file not found, data loading interrupted.")
            raise
        self._AFC_filter_trips()
        self._AFC_format_frame()

        # Load and prepare AVL data.
        try:
            self._AVL_read_data()
            self._AVL_filter_missions()
            self._AVL_format_frame()
        except FileNotFoundError:
            print("AVL file not found, creating one for tests.")
            self._AVL_create_data()

    def _read_line_infos(self) -> dict:
        """Read stations corresponding full and short names, store to data attribute."""
        with open("line.yml", "r") as file:
            line_param = safe_load(file)
        return line_param

    def _AFC_read_data(self) -> None:
        self.AFC_df = read_csv(
            self.paths
            + "AFC"
            + self.year
            + self.month
            + "/"
            + self.day
            + "mars_rerA.txt",
            delimiter="\t",
        )

    def _AFC_filter_trips(self) -> None:
        """Select trips between the origin and the destination station, reindex."""
        self.AFC_df["LIBL_O"] = self.AFC_df["LIBL_O"].str.replace("-", "_")
        self.AFC_df["LIBL_O"] = self.AFC_df["LIBL_O"].str.replace(" ", "_")
        self.AFC_df["LIBL_D"] = self.AFC_df["LIBL_D"].str.replace("-", "_")
        self.AFC_df["LIBL_D"] = self.AFC_df["LIBL_D"].str.replace(" ", "_")

        self.AFC_df = self.AFC_df[
            (self.AFC_df["LIBL_O"] == self.station_origin)
            & (self.AFC_df["LIBL_D"] == self.station_destination)
        ]

        self.AFC_df["index"] = self.AFC_df.index

    def _AFC_format_frame(self) -> None:
        """Convert AFC date values to TimeStamp format."""

        self.AFC_df["H_O"] = self.AFC_df["JOUR_EXPL_VALD"] + " " + self.AFC_df["H_O"]
        self.AFC_df["H_D"] = self.AFC_df["JOUR_EXPL_VALD"] + " " + self.AFC_df["H_D"]

        self.AFC_df["H_O"] = to_datetime(self.AFC_df["H_O"], format="%d/%m/%Y %H:%M:%S")
        self.AFC_df["H_D"] = to_datetime(self.AFC_df["H_D"], format="%d/%m/%Y %H:%M:%S")

    def _AVL_read_data(self) -> None:
        columns_names = [
            "Voie",
            self.station_origin_short,
            self.station_destination_short,
        ]

        self.AVL_arrival = read_csv(
            self.paths
            + "AVL"
            + self.year
            + self.month
            + "/AVL_arrivee_"
            + self.year
            + self.month
            + self.day
            + ".csv",
            encoding="latin-1",
            skiprows=[0, 1],
            usecols=columns_names,
        )
        self.AVL_departure = read_csv(
            self.paths
            + "AVL"
            + self.year
            + self.month
            + "/AVL_depart_"
            + self.year
            + self.month
            + self.day
            + ".csv",
            encoding="latin-1",
            skiprows=[0, 1],
            usecols=columns_names,
        )

    def _AVL_filter_missions(self) -> None:
        """Select missions desserving origin and destination station in the right direction."""
        self.AVL_arrival = self.AVL_arrival[
            (~self.AVL_arrival[self.station_origin_short].isna())
            & (~self.AVL_arrival[self.station_destination_short].isna())
            & (self.AVL_arrival["Voie"] == self.direction)
        ]
        self.AVL_departure = self.AVL_departure[
            (~self.AVL_departure[self.station_origin_short].isna())
            & (~self.AVL_departure[self.station_destination_short].isna())
            & (self.AVL_departure["Voie"] == self.direction)
        ]

    def _AVL_format_frame(self):
        """Combine arrival and departure AVL data in a single frame."""
        self.AVL_arrival = self.AVL_arrival.rename(
            columns={
                self.station_origin_short: self.station_origin + "_arrival",
                self.station_destination_short: self.station_destination + "_arrival",
            }
        )
        self.AVL_departure = self.AVL_departure.rename(
            columns={
                self.station_origin_short: self.station_origin + "_departure",
                self.station_destination_short: self.station_destination + "_departure",
            }
        )
        self.AVL_departure = self.AVL_departure.drop(columns=["Voie"])
        self.AVL_arrival = self.AVL_arrival.drop(columns=["Voie"])
        self.AVL_df = concat([self.AVL_departure, self.AVL_arrival], axis=1)

        self.AVL_df = self.year + "-" + self.month + "-" + self.day + " " + self.AVL_df
        self.AVL_df = self.AVL_df.applymap(to_datetime, errors="coerce")
        self.AVL_df = self.AVL_df.sort_values(by=self.station_origin + "_arrival")

        # Delete missions after midnight (datetime format problem).
        self.AVL_df = self.AVL_df.dropna()
        self.mission_nbr = len(self.AVL_df)

        self.AVL_df = self.AVL_df.reset_index(drop=True)

    def _AVL_create_data(self) -> None:
        """Create coherent AVL data when no AVL data file available."""
        AVL_columns = [
            self.station_origin + "_arrival",
            self.station_origin + "_departure",
            self.station_destination + "_arrival",
            self.station_destination + "_departure",
        ]

        # Compute chronological passage times uniformly
        # distributed over the entire time period of the travelers.
        earliest_departure = self.AFC_df["H_O"].min()
        latest_arrival = self.AFC_df["H_D"].max()

        # Arbitrary mission number.
        self.mission_nbr = 250  # arbitrary mission number
        self.time_period = to_datetime(
            latest_arrival, format="%Y/%m/%d %H:%M:%S"
        ) - to_datetime(earliest_departure, format="%Y/%m/%d %H:%M:%S")
        self.train_headway = self.time_period / self.mission_nbr
        self.AVL_df = DataFrame(
            self.train_headway, index=arange(self.mission_nbr), columns=AVL_columns
        )
        self.AVL_df.loc[0, self.station_origin + "_arrival"] = to_datetime(
            earliest_departure
        )
        self.AVL_df[self.station_origin + "_arrival"] = self.AVL_df[
            self.station_origin + "_arrival"
        ].cumsum()
        self.AVL_df = self.AVL_df.cumsum(axis=1)
        self.AVL_df = self.AVL_df.applymap(to_datetime)

    def compute_feasible_runs(self) -> None:
        """Compute and store in dictionnary indexes of first and last feasible run for each trip.
        This assumes that the missions in AVL are in chronological order and they
        don't overtake each other.
        Trips without feasible runs are deleted from the AFC dataframe.
        The attribute ndr_unfeasible_run counts these trips.
        """
        self.feasible_runs_dict = {}
        self.nbr_unfeasible_runs = 0

        print("Computing feasible runs...")
        for trip_id in tqdm(self.AFC_df.index):
            station_entry_time = self.AFC_df.loc[trip_id, "H_O"]
            station_exit_time = self.AFC_df.loc[trip_id, "H_D"]
            unfeasible_trip = False

            # Condition (3.11a)
            station_departure = self.station_origin + "_departure"
            possible_board_run = self.AVL_df.query(
                f"{station_departure} >= @station_entry_time"
            ).index.tolist()
            try:
                first_feasible_run = possible_board_run[0]
            except IndexError:
                unfeasible_trip = True

            # Condition (3.11bc)
            station_arrival = self.station_destination + "_arrival"
            possible_alight_run = self.AVL_df.query(
                f"{station_arrival} <= @station_exit_time"
            ).index.tolist()
            try:
                last_feasible_run = possible_alight_run[-1]
            except IndexError:
                unfeasible_trip = True

            # Delete impossible trips where no feasible run.

            if not unfeasible_trip and first_feasible_run > last_feasible_run:
                unfeasible_trip = True

            if unfeasible_trip:
                self.AFC_df = self.AFC_df.drop(labels=[trip_id], axis=0)
                self.nbr_unfeasible_runs += 1
                continue

            self.feasible_runs_dict["first_feasible_run", trip_id] = first_feasible_run
            self.feasible_runs_dict["last_feasible_run", trip_id] = last_feasible_run


if __name__ == "__main__":
    PATHS = "data/AVL-AFC-2015/"
    station_origin = "VINCENNES"
    station_destination = "LA_DEFENSE_GRANDE_ARCHE"
    date = "2015-03-16"
    direction = "west"

    data = Data(PATHS, date, direction, station_origin, station_destination)
    data.AVL_df.to_csv(f"output/{date}-AVL.csv")
    data.AFC_df.to_csv(f"output/{date}-AFC.csv")

    data.compute_feasible_runs()
