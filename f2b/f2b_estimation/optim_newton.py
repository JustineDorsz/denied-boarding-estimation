"""Estimation of fail-to-board probabilities from AFC and AVL data.
Log-likelihood optimization with grid search for each couple of components, 
with projected gradient descent.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-05"

from csv import writer
from time import time

from numpy import array, array_equal, linalg, matmul, ndarray
from yaml import safe_load

from data import Data
from likelihood_recursive_blocks import (
    compute_access_individual_likelihoods,
    compute_egress_individual_likelihoods,
    gradient_log_likelihood_global,
    hessian_log_likelihood_global,
    log_likelihood_global_with_egress_auxiliary_variables_updates,
)


def projection_on_unit_hypercube(vector: ndarray) -> ndarray:
    """Project a vector on unit hypercube [0,1]^n, with n the vector length."""
    projection = array([vector[i] for i in range(len(vector))])

    for component in range(len(vector)):
        if vector[component] < 0:
            projection[component] = 0
        elif vector[component] >= 1:
            projection[component] = 0.999

    return projection


def find_step_backtracking_line_search(
    f2b: ndarray,
    descent_direction: ndarray,
    descent_vector: ndarray,
    log_likelihood_reference: float,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
    c_armijo: float = 0.5,
    Niter: int = 10,
) -> float:

    tau = 1

    f2b_updated_tau = projection_on_unit_hypercube(
        f2b + tau * descent_vector,
    )

    if array_equal(
        f2b_updated_tau,
        f2b,
    ):
        # When the descent direction points out the admissible set in all directions,
        # no descent update, one returns arbitrary step.
        return tau

    log_likelihood_tau = log_likelihood_global_with_egress_auxiliary_variables_updates(
        f2b_updated_tau,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )

    iteration = 0
    while (
        iteration < Niter
        and log_likelihood_tau
        < log_likelihood_reference + c_armijo * tau * linalg.norm(descent_direction)
    ):
        iteration += 1
        tau /= 10
        f2b_updated_tau = projection_on_unit_hypercube(
            f2b + tau * descent_vector,
        )
        log_likelihood_tau = (
            log_likelihood_global_with_egress_auxiliary_variables_updates(
                f2b_updated_tau,
                access_individual_likelihoods,
                egress_individual_likelihoods,
                egress_auxiliary_variables,
                individual_likelihoods,
                data,
            )
        )

    return tau


def projected_newton_update(
    f2b_current: ndarray,
    iteration: int,
    log_likelihood_reference: float,
    access_individual_likelihoods: dict,
    egress_individual_likelihoods: dict,
    access_auxiliary_variables: dict,
    egress_auxiliary_variables: dict,
    individual_likelihoods: dict,
    data: Data,
):

    individual_log_likelihood_gradients = {}
    gradient_log_likelihood_current = gradient_log_likelihood_global(
        f2b_current,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_log_likelihood_gradients,
        data,
    )
    trip_of_interest = 0
    hessian_log_likelihood_current = hessian_log_likelihood_global(
        f2b_current,
        egress_individual_likelihoods,
        access_auxiliary_variables,
        egress_auxiliary_variables,
        individual_likelihoods,
        individual_log_likelihood_gradients,
        trip_of_interest,
        data,
    )
    try:
        hessian_inv = linalg.inv(hessian_log_likelihood_current)
    except linalg.LinAlgError:
        print(f"Hessian matrix not invertible at iteration {iteration}.")
    descent_direction = matmul(hessian_inv, gradient_log_likelihood_current)
    descent_vector = descent_direction / linalg.norm(descent_direction)

    tau = find_step_backtracking_line_search(
        f2b_current,
        descent_direction,
        descent_vector,
        log_likelihood_reference,
        access_individual_likelihoods,
        egress_individual_likelihoods,
        egress_auxiliary_variables,
        individual_likelihoods,
        data,
    )

    f2b_new = projection_on_unit_hypercube(f2b_current + tau * descent_vector)

    return f2b_new


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
    tol = 0.01
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

    # Iterative optimization with projected Newton update.
    # --------------------------------------------------------------------------------

    start_time = time()
    f2b = array(
        [f2b_probabilities_initial[i] for i in range(len(f2b_probabilities_initial))]
    )
    iteration = 0
    diff_likelihood = likelihood_by_iterations[0]
    while iteration < max_global_iterations and abs(diff_likelihood) > tol:
        iteration += 1
        f2b_updated = projected_newton_update(
            f2b,
            iteration,
            likelihood_by_iterations[iteration - 1],
            access_individual_likelihoods,
            egress_individual_likelihoods,
            access_auxiliary_variables,
            egress_auxiliary_variables,
            individual_likelihoods,
            data,
        )

        f2b = array([f2b_updated[i] for i in range(len(f2b_updated))])

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

    # Output.
    # --------------------------------------------------------------------------------

    if write_output:
        with open(
            "f2b/output/f2b_results_newton_" + origin_station + ".csv", "w"
        ) as output_file_f2b:
            writer_f2b = writer(output_file_f2b)
            writer_f2b.writerow(f2b_estimated)
