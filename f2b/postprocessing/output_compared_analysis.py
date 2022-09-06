from numpy import linspace
from f2b.f2b_estimation.data import Data
from matplotlib import pyplot

from f2b.postprocessing.output_analysis import load_estimated_f2b


def get_trip_number_by_run(data: Data) -> list:
    trip_number_by_run = [0 for _ in range(len(data.runs))]
    for trip_id in data.trips.index:
        for run in data.feasible_runs_by_trip[trip_id]:
            run_index = data.runs.index(run)
            trip_number_by_run[run_index] += 1

    return trip_number_by_run


if __name__ == "__main__":

    date = "04/02/2020"
    MAXIMUM_FEASIBLE_RUNS = 5

    destination_stations = {
        "VIN": ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"],
        "NAT": ["LYO", "CHL", "AUB", "ETO", "DEF"],
    }

    data = {
        "VIN": Data(date, "VIN", destination_stations["VIN"]),
        "NAT": Data(date, "NAT", destination_stations["NAT"]),
    }
    f2b_estimated = {"VIN": load_estimated_f2b("VIN"), "NAT": load_estimated_f2b("NAT")}

    trip_number_by_run = {
        "VIN": get_trip_number_by_run(data["VIN"]),
        "NAT": get_trip_number_by_run(data["NAT"]),
    }

    # There is one run that doesn't serve Vincennes and originates from Nation directly,
    # thus remove the train from the estimated probabilities in Nation, in position 302.
    f2b_estimated["NAT"].pop(302)
    trip_number_by_run["NAT"].pop(302)

    # Apply a threshold to significant estimated probabilities.
    proba_threshold = 0.2
    for run in range(len(data["VIN"].runs)):
        if f2b_estimated["VIN"][run] < proba_threshold:
            f2b_estimated["VIN"][run] = 0
        if f2b_estimated["NAT"][run] < proba_threshold:
            f2b_estimated["NAT"][run] = 0

    colors_4am = ["#2a225d", "#c83576", "#ffbe7d", "#e9f98f", "#eaf7ff"]
    plot_graph = {"compared estimated probabilities": True}

    if plot_graph["compared estimated probabilities"]:
        fig, ax = pyplot.subplots(figsize=(10, 5))
        run_departure_time = {
            "VIN": [
                data["VIN"].runs_departures[run, data["VIN"].origin_station]
                for run in data["VIN"].runs
            ],
            "NAT": [
                data["NAT"].runs_departures[run, data["NAT"].origin_station]
                for run in data["NAT"].runs
            ],
        }

        run_departure_time_labels = []
        for count, run in enumerate(data["VIN"].runs):
            if count % 40 == 0:
                run_departure_time_labels.append(
                    data["VIN"].runs_departures[run, data["VIN"].origin_station]
                )
        run_departure_time_labels.append(run_departure_time["VIN"][-1])

        ax.bar(
            run_departure_time["VIN"],
            f2b_estimated["VIN"],
            color=colors_4am[0],
            label="VIN",
        )
        ax.bar(
            run_departure_time["VIN"],
            [-x for x in f2b_estimated["NAT"]],
            color=colors_4am[1],
            label="NAT",
        )

        # ax2 = ax.twinx()
        # ax2.plot(
        #     run_departure_time["VIN"],
        #     trip_number_by_run["VIN"],
        #     label="VIN",
        #     color=colors_4am[0],
        # )
        # ax2.plot(
        #     run_departure_time["VIN"],
        #     [x for x in trip_number_by_run["NAT"]],
        #     label="NAT",
        #     color=colors_4am[1],
        # )

        ax.set_xticks(
            linspace(0, len(run_departure_time["VIN"]), len(run_departure_time_labels)),
            labels=run_departure_time_labels,
        )
        pyplot.legend()
        pyplot.show()
        pyplot.savefig(
            "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/compared_estimated_denied_probability.pdf"
        )
