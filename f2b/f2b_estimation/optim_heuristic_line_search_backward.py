"""Estimation of fail-to-board probabilities from AFC and AVL data.
Log-likelihood optimization with repeated line search for each component.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-12"

from csv import writer
from time import time

from matplotlib import pyplot
from numpy import array
from tqdm import tqdm
from yaml import safe_load

from data import Data
from likelihood import Likelihood


if __name__ == "__main__":

    # Parameters and data.
    # --------------------------------------------------------------------------------

    origin_station = "VIN"
    destination_stations = ["NAT", "LYO", "CHL", "AUB", "ETO", "DEF"]

    date = "04/02/2020"

    write_output = False

    with open(f"f2b/parameters/parameters_{origin_station}.yml") as file:
        parameters = safe_load(file)

    data = Data(date, origin_station, destination_stations)

    tol = 0.1
    max_global_iterations = 1
    max_dichotomy_iterations = 20

    # Offline precomputations.
    # --------------------------------------------------------------------------------

    access_individual_likelihoods = compute_access_individual_likelihoods(
        data, parameters
    )
    egress_individual_likelihoods = compute_egress_individual_likelihoods(
        data, parameters
    )

    # Initialization.
    # --------------------------------------------------------------------------------

    f2b_probabilities_initial = array([0.0 for _ in range(len(data.runs))])

    individual_likelihoods = {}
    access_auxiliary_variables = {}
    egress_auxiliary_variables = {}

    likelihood_by_iterations = [
        log_likelihood_global_with_egress_auxiliary_variables_updates(
            f2b_probabilities_initial,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )
    ]

    # Iterative optimization with line search by component.
    # --------------------------------------------------------------------------------

    start_time = time()
    f2b = array(
        [f2b_probabilities_initial[i] for i in range(len(f2b_probabilities_initial))]
    )
    gradients_local_log_likelihood = array([0.0 for _ in range(len(f2b))])

    iteration = 0
    diff_likelihood = likelihood_by_iterations[0]
    while iteration < max_global_iterations and abs(diff_likelihood) > tol:
        iteration += 1
        if iteration % 2 == 1:
            run_position_sequence = [i for i in range(len(f2b))]
        else:
            run_position_sequence = [len(f2b) - i - 1 for i in range(len(f2b))]

        for run_position in tqdm(run_position_sequence):
            f2b_left = array([x for x in f2b])
            f2b_right = array([x for x in f2b])
            f2b_left[run_position] = 0
            f2b_right[run_position] = 0.99

            indiv_likelihood_and_auxiliary_egress_update_by_run(
                run_position,
                f2b_left,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )
            derivative_left = differential_log_likelihood_global_by_component(
                run_position,
                f2b_left,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                access_auxiliary_variables,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )

            indiv_likelihood_and_auxiliary_egress_update_by_run(
                run_position,
                f2b_right,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )
            derivative_right = differential_log_likelihood_global_by_component(
                run_position,
                f2b_right,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                access_auxiliary_variables,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )

            if derivative_left * derivative_right >= 0:
                # If the two derivatives have the same sign, we let the denied boarding probability to zero.
                continue

            derivative_at_optimal_point = 0.0
            for _ in range(max_dichotomy_iterations):
                f2b_middle = array(
                    [(f2b_left[i] + f2b_right[i]) / 2.0 for i in range(len(f2b))]
                )

                indiv_likelihood_and_auxiliary_egress_update_by_run(
                    run_position,
                    f2b_middle,
                    access_individual_likelihoods,
                    egress_individual_likelihoods,
                    egress_auxiliary_variables,
                    individual_likelihoods,
                    data,
                )
                derivative_middle = differential_log_likelihood_global_by_component(
                    run_position,
                    f2b_middle,
                    access_individual_likelihoods,
                    egress_individual_likelihoods,
                    access_auxiliary_variables,
                    egress_auxiliary_variables,
                    individual_likelihoods,
                    data,
                )
                derivative_at_optimal_point = derivative_middle
                if derivative_left * derivative_middle < 0:
                    f2b_right = array([x for x in f2b_middle])
                    derivative_right = derivative_middle
                elif derivative_right * derivative_middle < 0:
                    f2b_left = array([x for x in f2b_middle])
                    derivative_left = derivative_middle
                else:
                    break
            f2b = array([x for x in f2b_middle])
            gradients_local_log_likelihood[run_position] = derivative_middle

        likelihood_by_iterations.append(
            log_likelihood_global_with_egress_auxiliary_variables_updates(
                f2b,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )
        )
        diff_likelihood = likelihood_by_iterations[-1] - likelihood_by_iterations[-2]
    f2b_estimated = array([x for x in f2b])
    print(f"Optimization time: {time()-start_time}s.")
    print(likelihood_by_iterations)
    print(gradients_local_log_likelihood)

    # Output.
    # --------------------------------------------------------------------------------
    individual_log_likelihood_gradients = {}

    if write_output:
        with open(
            "f2b/output/f2b_results_line_search_" + origin_station + ".csv", "w"
        ) as output_file:
            writer = writer(output_file)
            writer.writerow(f2b_estimated)

    pyplot.plot(f2b_estimated)
    pyplot.show()
