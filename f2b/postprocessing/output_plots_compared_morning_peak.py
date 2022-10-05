from statistics import mean, stdev

from f2b.f2b_estimation.data import Data
from f2b.postprocessing.output_plots import load_estimated_f2b
from matplotlib import pyplot
from numpy import linspace


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
    f2b_estimated = load_estimated_f2b("NAT", False)
    f2b_estimated_mp = load_estimated_f2b("NAT", True)
    difference = [
        f2b_estimated[i] - f2b_estimated_mp[i] for i in range(len(f2b_estimated))
    ]

    trip_number_by_run = {
        "VIN": get_trip_number_by_run(data["VIN"]),
        "NAT": get_trip_number_by_run(data["NAT"]),
    }

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
            run_departure_time["NAT"],
            f2b_estimated,
            color=colors_4am[0],
            label="NAT with whole day parameters",
        )
        ax.bar(
            run_departure_time["NAT"],
            [-x for x in f2b_estimated_mp],
            color=colors_4am[1],
            label="NAT with morning peak parameters",
        )

        ax.bar(
            run_departure_time["NAT"],
            difference,
            color="green",
            label="difference",
        )

        ax.set_xticks(
            linspace(0, len(run_departure_time["NAT"]), len(run_departure_time_labels)),
            labels=run_departure_time_labels,
        )
        print(
            f"Estimation with parameters calibrated on the whole day: mean = {mean(f2b_estimated)}, {stdev(f2b_estimated)}."
        )
        print(
            f"Estimation with parameters calibrated on morning peak hour: mean = {mean(f2b_estimated_mp)}, {stdev(f2b_estimated_mp)}."
        )

        pyplot.legend()
        pyplot.show()
        # pyplot.savefig(
        # "/home/justine/Nextcloud/Cired/Recherche/Econometrie/fail_to_board_probability/Draft_article/figures/compared_estimated_denied_probability.pdf"
        # )
