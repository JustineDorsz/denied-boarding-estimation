from numpy import linspace
from f2b.f2b_estimation.data import Data
from matplotlib import pyplot
from pandas import Timestamp

MAXIMUM_FEASIBLE_RUNS = 5


def load_estimated_f2b(station: str, method: str = "") -> list:

    f2b_file_path = "output/f2b_results/" + method + "_" + station + ".csv"
    with open(f2b_file_path, "r") as f2b_file:
        f2b_file_content = f2b_file.read()
        f2b_estimated = f2b_file_content.split(",")
        f2b_estimated = [float(f2b) for f2b in f2b_estimated]
        return f2b_estimated


def get_feasible_run_distribution_by_run(data: Data) -> list:
    feasible_run_distributions = [
        [0 for _ in range(len(data.runs))] for _ in range(MAXIMUM_FEASIBLE_RUNS)
    ]
    for trip_id in data.trips.index:
        nbr_feasible_runs = len(data.feasible_runs_by_trip[trip_id])
        for run in data.feasible_runs_by_trip[trip_id]:
            run_index = data.runs.index(run)
            feasible_run_distributions[nbr_feasible_runs - 1][run_index] += 1

    return feasible_run_distributions


if __name__ == "__main__":

    date = "04/02/2020"

    origin_station = "NAT"
    destination_stations = ["LYO", "CHL", "AUB", "ETO", "DEF"]

    data = Data(date, origin_station, destination_stations)
    f2b_estimated = load_estimated_f2b(origin_station, False)
    feasible_run_distributions = get_feasible_run_distribution_by_run(data)

    plot_graph = {
        "feasible run proportion": False,
        "journey time": True,
        "tap-in distribution": False,
        "estimated probabilities": False,
    }
    colors_4am = ["#2a225d", "#c83576", "#ffbe7d", "#e9f98f", "#eaf7ff"]

    # ------------------------------------------------------------------------------------------------------------------------------------

    if plot_graph["feasible run proportion"]:

        run_departure_time = [
            data.runs_departures[run, data.origin_station] for run in data.runs
        ]
        run_departure_time_labels = []
        for count, run in enumerate(data.runs):
            if count % 40 == 0:
                run_departure_time_labels.append(
                    data.runs_departures[run, data.origin_station]
                )
        run_departure_time_labels.append(run_departure_time[-1])

        fig, axs = pyplot.subplots(figsize=(10, 5))
        bottom = [0 for _ in range(len(data.runs))]
        for nbr_feasible_runs in range(MAXIMUM_FEASIBLE_RUNS):
            axs.bar(
                data.runs,
                feasible_run_distributions[nbr_feasible_runs],
                bottom=bottom,
                color=colors_4am[nbr_feasible_runs],
                label=f"{nbr_feasible_runs+1} possible runs",
            )
            axs.legend()
            bottom = [
                bottom[i] + feasible_run_distributions[nbr_feasible_runs][i]
                for i in range(len(data.runs))
            ]
        axs.set_xticks(
            linspace(0, len(run_departure_time), len(run_departure_time_labels)),
            labels=run_departure_time_labels,
        )

        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/feasible_runs_proportion.pdf"
        )

    # ------------------------------------------------------------------------------------------------------------------------------------

    if plot_graph["journey time"]:

        journey_time_morning_peak = {}
        journey_time_afternoon_peak = {}
        journey_time_off_peak = {}
        for destination_station in data.possible_destination_stations:
            journey_time_morning_peak[destination_station] = []
            journey_time_afternoon_peak[destination_station] = []
            journey_time_off_peak[destination_station] = []

        for trip_id in data.trips.index:
            trip_egress_time = data.trips.at[trip_id, "egress_time"]
            trip_journey_length = (
                Timestamp(date + " " + str(data.trips.at[trip_id, "egress_time"]))
                - Timestamp(date + " " + str(data.trips.at[trip_id, "access_time"]))
            ).seconds

            if trip_egress_time > "07:30:00" and trip_egress_time < "09:30:00":
                journey_time_morning_peak[
                    data.trips.at[trip_id, "egress_station"]
                ].append(trip_journey_length)

            if trip_egress_time > "16:30:00" and trip_egress_time < "19:30:00":
                journey_time_afternoon_peak[
                    data.trips.at[trip_id, "egress_station"]
                ].append(trip_journey_length)

            else:
                journey_time_off_peak[data.trips.at[trip_id, "egress_station"]].append(
                    trip_journey_length
                )

        fig, axs = pyplot.subplots(
            2, len(data.possible_destination_stations), figsize=(15, 7)
        )
        for pos, destination_station in enumerate(data.possible_destination_stations):

            axs[0, pos].hist(
                journey_time_off_peak[destination_station],
                bins=40,
                color=colors_4am[0],
                label="off-peak hour in " + destination_station,
            )
            axs[1, pos].hist(
                journey_time_morning_peak[destination_station],
                bins=40,
                color=colors_4am[1],
                label="peak hour in " + destination_station,
            )
            axs[0, pos].legend()
            axs[1, pos].legend()
            axs[0, pos].axis(xmin=0, xmax=2000)
            axs[1, pos].axis(xmin=0, xmax=2000)

        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/journey_time_distribution.pdf"
        )

    # ------------------------------------------------------------------------------------------------------------------------------------

    if plot_graph["tap-in distribution"]:

        tap_in_times = [
            Timestamp(str(date + " " + access_time))
            for access_time in data.trips["access_time"]
        ]

        fig, ax = pyplot.subplots(figsize=(10, 5))
        run_departure_time = [
            data.runs_departures[run, data.origin_station] for run in data.runs
        ]
        run_departure_time_labels = []
        for count, run in enumerate(data.runs):
            if count % 40 == 0:
                run_departure_time_labels.append(
                    data.runs_departures[run, data.origin_station]
                )
        run_departure_time_labels.append(run_departure_time[-1])

        ax.hist(tap_in_times, bins=200, color=colors_4am[0])
        # ax.set_xticks(
        #     linspace(0, len(run_departure_time), len(run_departure_time_labels)),
        #     labels=run_departure_time_labels,
        # )
        # pyplot.show()

        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/tap_in_times_distribution.pdf"
        )

    # ------------------------------------------------------------------------------------------------------------------------------------

    if plot_graph["estimated probabilities"]:
        fig, ax = pyplot.subplots(figsize=(10, 5))
        run_departure_time = [
            data.runs_departures[run, data.origin_station] for run in data.runs
        ]
        run_departure_time_labels = []
        for count, run in enumerate(data.runs):
            if count % 40 == 0:
                run_departure_time_labels.append(
                    data.runs_departures[run, data.origin_station]
                )
        run_departure_time_labels.append(run_departure_time[-1])

        ax.bar(run_departure_time, f2b_estimated, color=colors_4am[0])
        ax.bar(run_departure_time, [-x for x in f2b_estimated], color=colors_4am[1])

        ax.set_xticks(
            linspace(0, len(run_departure_time), len(run_departure_time_labels)),
            labels=run_departure_time_labels,
        )
        pyplot.show()
        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/estimated_denied_probability.pdf"
        )
