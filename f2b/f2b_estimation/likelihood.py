"""
Estimation of fail-to-board probabilities (also referred to as delayed boarding 
probabilities) by train run from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-11"

from math import exp, log
from time import time

import scipy.stats
from f2b.f2b_estimation.data import Data
from numpy import array, ndarray
from tqdm import tqdm
from yaml import safe_load


class Likelihood:
    """
    Likelihood contributions by trip.

    Attributes:
        - data(Data): info of trips and runs
        - origin_station(int): code of the origin station of the trips
        - f2b_probas(ndarray): fail to board probabilities by runs, stored by runs
        position according to chronological order
        - access_individual_contributions(dict): likelihood factor for access time,
        for each trip and its associated runs, by tuple key (trip_id, run_code)
        - egress_individual_contributions(dict): likelihood factor for egress time,
        for each trip and its associated runs, by tuple key (trip_id, run_code)
        - access_auxiliary_variables(dict): recursive access auxiliary variables,
        depending on f2b, by (trip_id, run_code)
        - egress_auxiliary_variables(dict): recursive egress auxiliary variables,
        depending on f2b, by (trip_id, run_code)
        - individual_log_likelihoods(dict): complete log likelihood by trip_id
        - individual_log_likelihood_derivatives(dict): complete individual log
        likelihood derivative, by (trip_id, run_code)
        - distributions_parameters(dict): distribution name and parameters by station code,
        modeling stations access and egress time

    Methods:
        - update_f2b_probas(f2b_probas: ndarray) -> None
        Set new value of f2b_probas. Update values affected by the new_probas of
        egress_auxiliary_variables, individual_log_likelihoods,
        access_auxiliary_variables and individual_log_likelihood_derivative.
        - get_global_log_likelihood() -> float
        Sum and return individual log likelihoods for all trips.
        - get_log_likelihood_derivative_by_run(self, run_code: str) -> float
        Sum individual log likelihood derivative for one run_code.
        - get_global_log_likelihood_gradient() -> ndarray
        Sum individual log likelihood derivatives by run_code and return log
        likelihood gradient vector.
        - get_global_log_likelihood_hessian() -> ndarray
        Compute and return log likelihood hessian matrix.
    """

    def __init__(self, data: Data, f2b_probas: array):
        """Initialize Likelihood attributes. Compute access and egress individual
        contributions independant from fail-to-board probability according to
        distributions modeling stations access and egress time. Store in
        access_individual_contributions and in egress_individual_contributions.
        Initialize f2b_probas and compute access_auxiliary_variables,
        egress_auxiliary_variables, individual_log_likelihoods
        and individual_log_likelihood_derivative.

        Args:
            - data(Data): info of trips and runs
            - f2b_probas(array): fail-to-board probabilities by run

        """

        self.data = data
        self.origin_station = data.origin_station

        # distributions parameters for each station
        with open(f"parameters/parameters_{self.data.origin_station}.yml") as file:
            distributions_parameters = safe_load(file)

        self.distribution_parameters = distributions_parameters

        self.egress_individual_contributions = {}
        self.access_individual_contributions = {}

        print(
            "(Offline) Compute access and egress individual likelihood contributions..."
        )
        for trip_id in tqdm(self.data.trips):
            # access term
            station_access_distrib_name = self.distribution_parameters[
                self.data.origin_station
            ]["distribution"]
            station_access_duration_distribution = eval(
                "scipy.stats." + station_access_distrib_name
            )
            station_access_duration_distribution_params = self.distribution_parameters[
                self.data.origin_station
            ]["parameters"]

            # access term: depend on asusmed boarded run
            for run_code in self.data.trips[trip_id].associated_runs:
                access_duration_upper_bound = (
                    self.data.runs[run_code].departure_times[self.data.origin_station]
                    - self.data.trips[trip_id].access_time
                ).total_seconds()
                access_duration_lower_bound = 0

                # if run_code not the first run of the day
                if self.data.runs[run_code].previous_run:
                    previous_run_code = self.data.runs[run_code].previous_run
                    previous_run_departure_time = self.data.runs[
                        previous_run_code
                    ].departure_times[self.data.origin_station]

                    # max in case station access after the departure of previous_run
                    access_duration_lower_bound = max(
                        0,
                        (
                            previous_run_departure_time
                            - self.data.trips[trip_id].access_time
                        ).total_seconds(),
                    )

                access_proba_difference = station_access_duration_distribution.cdf(
                    access_duration_upper_bound,
                    **station_access_duration_distribution_params,
                ) - station_access_duration_distribution.cdf(
                    access_duration_lower_bound,
                    **station_access_duration_distribution_params,
                )

                self.access_individual_contributions[
                    trip_id, run_code
                ] = access_proba_difference

            # egress term: distribution info
            station_egress_distrib_name = self.distribution_parameters[
                self.data.trips[trip_id].egress_station
            ]["distribution"]
            station_egress_duration_distribution = eval(
                "scipy.stats." + station_egress_distrib_name
            )
            station_egress_duration_distribution_params = self.distribution_parameters[
                self.data.trips[trip_id].egress_station
            ]["parameters"]

            # egress term: depends on assumed boarded run
            for run_code in self.data.trips[trip_id].associated_runs:
                egress_duration = (
                    self.data.trips[trip_id].egress_time
                    - self.data.runs[run_code].arrival_times[
                        self.data.trips[trip_id].egress_station
                    ]
                ).total_seconds()
                egress_proba = station_egress_duration_distribution.pdf(
                    egress_duration, **station_egress_duration_distribution_params
                )
                self.egress_individual_contributions[trip_id, run_code] = egress_proba

        self.__initialize_f2b_probas(f2b_probas)

    def _update_attributes_by_trip_id(self, trip_id: int) -> None:
        """Update individual_log_likelihoods, egress_auxiliary_variables,
        access_auxiliary_variables and individual_log_likelihood_derivatives
        associated to a given trip."""
        individual_likelihood = 0
        start = time()
        # update egress auxiliary variables and likelihoods terms
        for position, run_code in reversed(
            list(enumerate(self.data.trips[trip_id].associated_runs))
        ):
            run_position = self.data.runs_chronological_order.index(run_code)
            self.egress_auxiliary_variables[
                trip_id, run_code
            ] = self.egress_individual_contributions[trip_id, run_code] * (
                1 - self.f2b_probas[run_position]
            )

            #  If not last possible run, add the term of recursive dependance.
            if position != len(self.data.trips[trip_id].associated_runs) - 1:
                next_run_code = self.data.trips[trip_id].associated_runs[position + 1]
                self.egress_auxiliary_variables[trip_id, run_code] += (
                    self.f2b_probas[run_position]
                    * self.egress_auxiliary_variables[trip_id, next_run_code]
                )

            individual_likelihood += (
                self.access_individual_contributions[trip_id, run_code]
                * self.egress_auxiliary_variables[trip_id, run_code]
            )
        self.individual_log_likelihoods[trip_id] = log(individual_likelihood)

        # update access auxiliary variables and derivative terms
        for position, run_code in enumerate(self.data.trips[trip_id].associated_runs):

            self.access_auxiliary_variables[
                trip_id, run_code
            ] = self.access_individual_contributions[trip_id, run_code]

            #  If not first possible run, add the term of recursive dependance.
            if position != 0:
                previous_run_code = self.data.trips[trip_id].associated_runs[
                    position - 1
                ]
                previous_run_position = self.data.runs_chronological_order.index(
                    previous_run_code
                )
                self.access_auxiliary_variables[trip_id, run_code] += (
                    self.f2b_probas[previous_run_position]
                    * self.access_auxiliary_variables[trip_id, previous_run_code]
                )

            #  directionnal diff of individual likelihood along run_code.
            if position != len(self.data.trips[trip_id].associated_runs) - 1:
                next_run_code = self.data.trips[trip_id].associated_runs[position + 1]
                egress_contribution = self.egress_auxiliary_variables[
                    trip_id, next_run_code
                ]
            else:
                egress_contribution = 0

            self.individual_log_likelihood_derivatives[trip_id, run_code] = (
                self.access_auxiliary_variables[trip_id, run_code]
                * (
                    egress_contribution
                    - self.egress_individual_contributions[trip_id, run_code]
                )
                / exp(self.individual_log_likelihoods[trip_id])
            )

    def __initialize_f2b_probas(self, f2b_probas: ndarray) -> None:
        """Set f2b_probas attribute. Compute egress_auxiliary_variables and
        individual_log_likelihoods with recursive relation on associated runs
        of each trip in reversed chronological order. Compute
        access_auxiliary_variables and individual_log_likelihood_derivatives with
        recursive relation on associated runs of each trip in chronological order.

        Args:
            - f2b_probas(ndarray): fail-to-board probabilities of each run in chronological order.
        """

        self.f2b_probas = f2b_probas
        self.access_auxiliary_variables = {}
        self.egress_auxiliary_variables = {}
        self.individual_log_likelihoods = {}
        self.individual_log_likelihood_derivatives = {}

        for trip_id in self.data.trips:
            self._update_attributes_by_trip_id(trip_id)

    def update_f2b_probas(self, f2b_probas: ndarray) -> None:
        """Set new value of f2b_probas. Update values affected by the new_probas of
        egress_auxiliary_variables, individual_log_likelihoods,
        access_auxiliary_variables and individual_log_likelihood_derivatives.

        Args:
            - f2b_probas(ndarray): fail-to-board probabilities of each run in chronological order.
        """

        # run_codes whose fail-to-board probability is updated
        run_codes_updates = []
        for run_position, (old_f2b_proba, new_f2b_proba) in enumerate(
            zip(self.f2b_probas, f2b_probas)
        ):
            if old_f2b_proba != new_f2b_proba:
                run_codes_updates.append(
                    self.data.runs_chronological_order[run_position]
                )

        self.f2b_probas = f2b_probas
        for run_code in run_codes_updates:
            # loop over trips affected by new fail-to-board probabilities
            for trip_id in self.data.runs[run_code].associated_trips:
                self._update_attributes_by_trip_id(trip_id)

    def get_global_log_likelihood(self) -> float:
        """Compute the sum of individual log likelihoods for all trips.

        Return
            (float): total log likelihoods
        """

        global_log_likelihood = sum(self.individual_log_likelihoods.values())
        return global_log_likelihood

    def get_log_likelihood_derivative_by_run(self, run_code: str) -> float:
        """Get directionnal derivative of the log-likelihood according to a given
        run by summing individual log likelihood derivatives of the associated trips.

        Args:
            - run_code(str): code of the run

        Return:
            (float): directionnal derivative of the log likelihood according to the run
        """
        derivative = 0
        for trip_id in self.data.runs[run_code].associated_trips:
            derivative += self.individual_log_likelihood_derivatives[trip_id, run_code]
        return derivative

    def get_global_log_likelihood_gradient(self) -> ndarray:
        """Sum individual log likelihood derivatives by run_code and return total
        log likelihood gradient vector.

        Return:
            (ndarray): total log likelihood gradient vector. Each component corresponds
            to a run, in chronological order.
        """

        gradient_log_likelihood = array([0.0 for _ in range(self.data.runs_number)])
        for run_position, run_code in enumerate(self.data.runs_chronological_order):
            for trip_id in self.data.runs[run_code].associated_trips:
                gradient_log_likelihood[
                    run_position
                ] += self.individual_log_likelihood_derivatives[trip_id, run_code]
        return gradient_log_likelihood

    def get_global_log_likelihood_hessian(self) -> ndarray:
        """Compute log likelihood hessian matrix with recursive relations
        involving access_auxiliary_variables and egress_auxiliary_variables.

        Return:
            (ndarray): log likelihood hessian matrix. The rows and columns correspond
            to the runs in chronological order.
        """

        hessian_log_likelihood = array(
            [
                [0.0 for _ in range(self.data.runs_number)]
                for _ in range(self.data.runs_number)
            ]
        )

        for trip_id in self.data.trips:
            for first_position, first_run_code in enumerate(
                self.data.trips[trip_id].associated_runs
            ):
                first_run_position = self.data.runs_chronological_order.index(
                    first_run_code
                )
                hessian_log_likelihood[first_run_position, first_run_position] += (
                    -self.individual_log_likelihood_derivatives[trip_id, first_run_code]
                    ** 2
                )
                auxiliary_proba = 1

                for second_run_code in self.data.trips[trip_id].associated_runs[
                    first_position + 1 :
                ]:
                    if second_run_code != self.data.trips[trip_id].associated_runs[-1]:
                        next_run_code = self.data.trips[trip_id].associated_runs[
                            first_position + 2
                        ]
                        egress_contribution = (
                            self.egress_auxiliary_variables[trip_id, next_run_code]
                            - self.egress_individual_contributions[
                                trip_id, second_run_code
                            ]
                        )
                    else:
                        egress_contribution = -self.egress_individual_contributions[
                            trip_id, second_run_code
                        ]

                    second_run_position = self.data.runs_chronological_order.index(
                        second_run_code
                    )
                    hessian_log_likelihood[first_run_position, second_run_position] += (
                        egress_contribution
                        * auxiliary_proba
                        * self.access_auxiliary_variables[trip_id, first_run_code]
                        / exp(self.individual_log_likelihoods[trip_id])
                        - self.individual_log_likelihood_derivatives[
                            trip_id, first_run_code
                        ]
                        * self.individual_log_likelihood_derivatives[
                            trip_id, second_run_code
                        ]
                    )
                    hessian_log_likelihood[
                        second_run_position, first_run_position
                    ] = hessian_log_likelihood[first_run_position, second_run_position]
                    auxiliary_proba *= self.f2b_probas[second_run_position]

        return hessian_log_likelihood
