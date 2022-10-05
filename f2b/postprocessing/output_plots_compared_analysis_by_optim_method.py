from numpy import linspace
from f2b.f2b_estimation.data import Data
from matplotlib import pyplot

from f2b.postprocessing.output_plots import load_estimated_f2b


def get_trip_number_by_run(data: Data) -> list:
    trip_number_by_run = [0 for _ in range(len(data.runs))]
    for trip_id in data.trips.index:
        for run in data.feasible_runs_by_trip[trip_id]:
            run_index = data.runs.index(run)
            trip_number_by_run[run_index] += 1

    return trip_number_by_run


if __name__ == "__main__":

    date = "04/02/2020"

    origin_station = "VIN"
    destination_stations = ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"]

    data = Data(date, origin_station, destination_stations)

    f2b_estimated_scipy = load_estimated_f2b(origin_station, False)
    with open(
        "f2b/output/f2b_results_line_search_" + origin_station + ".csv", "r"
    ) as f2b_file:
        f2b_file_content = f2b_file.read()
        f2b_estimated = f2b_file_content.split(",")
        f2b_estimated_line_search = [float(f2b) for f2b in f2b_estimated]

    run_departure_time = [
        data.runs_departures[run, data.origin_station] for run in data.runs
    ]

    # Apply a threshold to significant estimated probabilities.
    proba_threshold = 0.0
    for run in range(len(data.runs)):
        if f2b_estimated_scipy[run] < proba_threshold:
            f2b_estimated_scipy[run] = 0
        if f2b_estimated_line_search[run] < proba_threshold:
            f2b_estimated_line_search[run] = 0

    colors_4am = ["#2a225d", "#c83576", "#ffbe7d", "#e9f98f", "#eaf7ff"]

    plot_graph = {"compared estimated probabilities": True}

    if plot_graph["compared estimated probabilities"]:
        fig, ax = pyplot.subplots(figsize=(10, 5))

        run_departure_time_labels = []
        for count, run in enumerate(data.runs):
            if count % 40 == 0:
                run_departure_time_labels.append(
                    data.runs_departures[run, data.origin_station]
                )
        run_departure_time_labels.append(run_departure_time[-1])

        ax.bar(
            run_departure_time,
            f2b_estimated_scipy,
            color=colors_4am[0],
            label="scipy: log-likelihood = $-45800$, estimation in 600s",
        )
        ax.bar(
            run_departure_time,
            [-x for x in f2b_estimated_line_search],
            color=colors_4am[1],
            label="line search: log-likelihood = -45755, estimation in 7s",
        )

        ax.set_xticks(
            linspace(0, len(run_departure_time), len(run_departure_time_labels)),
            labels=run_departure_time_labels,
        )
        pyplot.title(f"Estimated denied probabilities in {origin_station}.")
        pyplot.legend()
        pyplot.show()
