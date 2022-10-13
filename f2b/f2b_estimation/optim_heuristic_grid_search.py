"""Estimation of fail-to-board probabilities from AFC and AVL data.
Log-likelihood optimization with grid search for each couple of components, 
with projected gradient descent.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-09-28"

from csv import writer
from math import log
from time import time

from numpy import array, array_equal, ndarray, linalg
from tqdm import tqdm
from yaml import safe_load

from data import Data
from f2b.f2b_estimation.likelihood import (
    compute_access_individual_likelihoods,
    compute_egress_individual_likelihoods,
    differential_log_likelihood_global_by_component,
    gradient_log_likelihood_global,
    indiv_likelihood_and_auxiliary_egress_update_by_run,
    log_likelihood_global_with_egress_auxiliary_variables_updates,
)


def projection_on_unit_square(vector: ndarray) -> ndarray:
    """Project a vector on the square [0,1]x[0,1]."""
    projection = array([vector[0], vector[1]])
    if vector[0] < 0:
        projection[0] = 0
    elif vector[0] >= 1:
        projection[0] = 0.999

    if vector[1] < 0:
        projection[1] = 0
    elif vector[1] >= 1:
        projection[1] = 0.999

    return projection


def projection_on_unit_square_two_components(
    vector: ndarray, first_run_position: int, second_run_position: int
) -> ndarray:
    """Project two components of a longer vector on the square [0,1]x[0,1]."""
    projection = array([vector[i] for i in range(len(vector))])
    if vector[first_run_position] < 0:
        projection[first_run_position] = 0
    elif vector[first_run_position] >= 1:
        projection[first_run_position] = 0.999

    if vector[second_run_position] < 0:
        projection[second_run_position] = 0
    elif vector[second_run_position] >= 1:
        projection[second_run_position] = 0.999

    return projection


def find_step_backtracking_line_search(
    first_run_position: int,
    second_run_position: int,
    f2b: ndarray,
    descent_direction_two_components: ndarray,
    log_likelihood_reference: float,
    concerned_trips: list,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
    c_armijo: float = 0.5,
    Niter: int = 10,
) -> float:

    tau = 1

    descent_vector = array([0.0 for _ in range(len(f2b))])
    descent_vector[first_run_position] = descent_direction_two_components[0]
    descent_vector[second_run_position] = descent_direction_two_components[1]

    f2b_updated_tau = projection_on_unit_square_two_components(
        f2b + tau * descent_vector,
        first_run_position,
        second_run_position,
    )

    if array_equal(
        f2b_updated_tau,
        f2b,
    ):
        # When the descent direction points out the admissible set
        # and f2b[first_run_position, second_run_position] is already on the boundary,
        # no descent update, one returns arbitrary step.
        return tau
    diff_first_run_position = differential_log_likelihood_global_by_component(
        first_run_position,
        f2b,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    diff_second_run_position = differential_log_likelihood_global_by_component(
        second_run_position,
        f2b,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    gradient_reference = array([diff_first_run_position, diff_second_run_position])

    indiv_likelihood_and_auxiliary_egress_update_by_run(
        first_run_position,
        f2b_updated_tau,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )

    indiv_likelihood_and_auxiliary_egress_update_by_run(
        second_run_position,
        f2b_updated_tau,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    log_likelihood_at_tau = 0
    for trip_id in concerned_trips:
        log_likelihood_at_tau += log(individual_likelihoods[trip_id])

    iteration = 0
    while (
        iteration < Niter
        and log_likelihood_at_tau
        < log_likelihood_reference + c_armijo * tau * linalg.norm(gradient_reference)
    ):
        iteration += 1
        tau /= 10
        f2b_updated_tau = projection_on_unit_square_two_components(
            f2b + tau * descent_vector,
            first_run_position,
            second_run_position,
        )
        indiv_likelihood_and_auxiliary_egress_update_by_run(
            first_run_position,
            f2b_updated_tau,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )

        indiv_likelihood_and_auxiliary_egress_update_by_run(
            second_run_position,
            f2b_updated_tau,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )
        log_likelihood_at_tau = 0
        for trip_id in concerned_trips:
            log_likelihood_at_tau += log(individual_likelihoods[trip_id])

    return tau


def optim_two_dim_projected_gradient(
    first_run_position: int,
    second_run_position: int,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
    f2b_start: ndarray,
    local_tol: float = 0.01,
    max_local_iterations: int = 100,
) -> ndarray:

    f2b = array([f2b_start[i] for i in range(len(f2b_start))])
    first_run_code = data.runs[first_run_position]
    second_run_code = data.runs[second_run_position]
    concerned_trips_by_two_runs = list(
        set(
            data.concerned_trips_by_run[first_run_code]
            + data.concerned_trips_by_run[second_run_code]
        )
    )
    log_likelihood_by_local_iterations = []
    indiv_likelihood_and_auxiliary_egress_update_by_run(
        first_run_position,
        f2b,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )

    indiv_likelihood_and_auxiliary_egress_update_by_run(
        second_run_position,
        f2b,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )
    log_likelihood_by_local_iterations.append(0)
    for trip_id in concerned_trips_by_two_runs:
        log_likelihood_by_local_iterations[0] += log(individual_likelihoods[trip_id])

    diff_local_likelihood = log_likelihood_by_local_iterations[0]
    local_iteration = 0
    while (
        local_iteration < max_local_iterations
        and abs(diff_local_likelihood) > local_tol
    ):
        local_iteration += 1

        diff_first_run_position = differential_log_likelihood_global_by_component(
            first_run_position,
            f2b,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            access_auxiliary_variables,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )
        diff_second_run_position = differential_log_likelihood_global_by_component(
            second_run_position,
            f2b,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            access_auxiliary_variables,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )
        descent_direction = array(
            [diff_first_run_position, diff_second_run_position]
        ) / linalg.norm(array([diff_first_run_position, diff_second_run_position]))

        tau = find_step_backtracking_line_search(
            first_run_position,
            second_run_position,
            f2b,
            descent_direction,
            log_likelihood_by_local_iterations[local_iteration - 1],
            concerned_trips_by_two_runs,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            access_auxiliary_variables,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )
        descent_vector = array([0.0 for i in range(len(data.runs))])
        descent_vector[first_run_position] = descent_direction[0]
        descent_vector[second_run_position] = descent_direction[1]
        f2b = projection_on_unit_square_two_components(
            f2b + tau * descent_vector,
            first_run_position,
            second_run_position,
        )

        indiv_likelihood_and_auxiliary_egress_update_by_run(
            first_run_position,
            f2b,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )

        indiv_likelihood_and_auxiliary_egress_update_by_run(
            second_run_position,
            f2b,
            access_individual_likelihoods,
            egress_individual_likelihoods,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )

        log_likelihood_by_local_iterations.append(0)
        for trip_id in concerned_trips_by_two_runs:
            log_likelihood_by_local_iterations[local_iteration] += log(
                individual_likelihoods[trip_id]
            )

        diff_local_likelihood = (
            log_likelihood_by_local_iterations[-1]
            - log_likelihood_by_local_iterations[-2]
        )

    if iteration == max_local_iterations:
        print(
            f" The projected gradient algorithm didn't converge for component {first_run_position}, {second_run_position}."
        )
    return f2b


if __name__ == "__main__":

    # Parameters and data.
    # --------------------------------------------------------------------------------

    origin_station = "NAT"
    destination_stations = ["LYO", "CHL", "AUB", "ETO", "DEF"]

    date = "04/02/2020"

    write_output = True

    with open(f"f2b/parameters/parameters_{origin_station}.yml") as file:
        parameters = safe_load(file)

    data = Data(date, origin_station, destination_stations)
    tol = 0.1
    max_global_iterations = 10

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

    f2b_probabilities_initial = [0.0 for _ in range(len(data.runs))]

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

    # Iterative optimization with projected gradient search by components couple.
    # --------------------------------------------------------------------------------

    start_time = time()
    f2b = [f2b_probabilities_initial[i] for i in range(len(f2b_probabilities_initial))]
    iteration = 0
    diff_likelihood = likelihood_by_iterations[0]
    while iteration < max_global_iterations and abs(diff_likelihood) > tol:
        iteration += 1
        for i in tqdm(range(len(f2b) // 2)):
            first_run_position = 2 * i
            second_run_position = 2 * i + 1
            f2b_updated = optim_two_dim_projected_gradient(
                first_run_position,
                second_run_position,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                access_auxiliary_variables,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
                f2b,
            )

            f2b = [f2b_updated[i] for i in range(len(f2b_updated))]

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
    f2b_estimated = [x for x in f2b]
    print(f"Optimization time: {time()-start_time}s.")
    print(likelihood_by_iterations)

    # Output.
    # --------------------------------------------------------------------------------
    individual_log_likelihood_gradients = {}

    gradient_list = gradient_log_likelihood_global(
        f2b_estimated,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_log_likelihood_gradients,
        data,
    )

    if write_output:
        with open(
            "f2b/output/f2b_results_grid_search_" + origin_station + ".csv", "w"
        ) as output_file_f2b:
            writer_f2b = writer(output_file_f2b)
            writer_f2b.writerow(f2b_estimated)
