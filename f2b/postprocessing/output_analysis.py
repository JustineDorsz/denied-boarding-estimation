from f2b.f2b_estimation.data import Data
from matplotlib import pyplot
from pandas import Timestamp

origin_station = "VIN"
destination_stations = ["LYO", "AUB", "ETO", "DEF", "NAP", "NAU", "RUE", "GER"]


date = "03/02/2020"
MAXIMUM_FEASIBLE_RUNS = 5


def load_estimated_f2b(station: str) -> list:
    with open("f2b/output/f2b_results_" + origin_station + ".csv", "r") as f2b_file:
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

    data = Data(date, origin_station, destination_stations)
    f2b_estimated = load_estimated_f2b("CHL")
    feasible_run_distributions = get_feasible_run_distribution_by_run(data)
    # Renormalization.
    sum_feasible_runs = [0 for _ in range(len(data.runs))]
    for nbr_feasible_runs in range(MAXIMUM_FEASIBLE_RUNS):
        sum_feasible_runs = [
            sum_feasible_runs[i] + feasible_run_distributions[nbr_feasible_runs][i]
            for i in range(len(data.runs))
        ]
    for nbr_feasible_runs in range(MAXIMUM_FEASIBLE_RUNS):
        feasible_run_distributions[nbr_feasible_runs] = [
            feasible_run_distributions[nbr_feasible_runs][i]
            / max(sum_feasible_runs[i], 1)
            for i in range(len(data.runs))
        ]

    tap_in_times = [
        Timestamp(str(date + " " + access_time))
        for access_time in data.trips["access_time"]
    ]

    colors_aubergine = ["#8e194c", "#7c3296", "#005270", "#00a6ba", "#ff6232"]
    colors_4am = ["#2a225d", "#c83576", "#ffbe7d", "#e9f98f", "#eaf7ff"]
    colors_travel = ["#300c71", "#ff6a6b", "#f19390", "#449fd8", "#b34372"]
    colors_eric = ["#f04770", "#ffd167", "#06d7a0", "#108ab1", "#073a4b"]

    fig, axs = pyplot.subplots(3, 1)
    run_departure_time = [
        data.runs_departures[run, data.origin_station] for run in data.runs
    ]
    axs[0].bar(run_departure_time, f2b_estimated, color=colors_eric[0])
    axs[0].set_title(
        "Estimated fail to board probability in Vincennes throughout the day.",
        fontsize=16,
    )
    axs[0].get_xaxis().set_visible(False)

    axs[1].hist(tap_in_times, bins=300, color=colors_eric[0])
    axs[1].set_title("Distribution of tap-in validations.", fontsize=10)

    bottom = [0 for _ in range(len(data.runs))]
    for nbr_feasible_runs in range(MAXIMUM_FEASIBLE_RUNS):
        axs[2].bar(
            data.runs,
            feasible_run_distributions[nbr_feasible_runs],
            bottom=bottom,
            color=colors_eric[nbr_feasible_runs],
            label=f"{nbr_feasible_runs+1} possible runs",
        )
        axs[2].set_title(
            "Proportion of the number of possible runs by trips concerned for each run.",
            fontsize=10,
        )
        axs[2].get_xaxis().set_visible(False)
        axs[2].legend()
        bottom = [
            bottom[i] + feasible_run_distributions[nbr_feasible_runs][i]
            for i in range(len(data.runs))
        ]

    pyplot.show()
