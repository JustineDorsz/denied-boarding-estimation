"""
**MISSING HEADER
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-12"

from abc import ABC, abstractmethod
from csv import writer
from time import time

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood import Likelihood
from matplotlib import pyplot
from numpy import array, ndarray
from tqdm import tqdm


class MaximumLikelihoodEstimation(ABC):
    """Maximum likelihood estimation with iterative optimization methods.

    This abstract class is a blueprint for different optimization methods.
    The abstract method _search_maximum() called by run_iterative_optimization()
    must be implemented by child classes.

    Attributes:
        - data(Data): observations, info of trips and runs
        - likelihood(Likelihood): log-likelihood terms and log-likelihood gradient
        terms for each trip
        - initial_f2b_probas(array): initial values of fail-to-board probabilities
        by run
        - global_tolerance(float): stopping criterion between two successive
        log-likelihoods values in global iterative optimization algorithm
        - max_iteration(int): number of maximum iterations allowed in global
        iterative optimization algorithm
        - log_likelihood_by_iteration(list): list of log-likelihood values by
        iterations
        - estimated_f2b_probas(array): estimated values of fail-to-board probabilities
        by run after optimization


    Methods:
        - run_iterative_optimization(): iterative optimization algorithm
        - write_estimated_f2b_probas(name:str, path:str): write fail-to-board
        probabilities as .csv file
        - plot_estimated_f2b_probas(): plot fail-to-board probabilities
    """

    def __init__(
        self,
        likelihood: Likelihood,
        initial_f2b_probas: ndarray,
        global_tolerance: float,
        max_iteration: int,
    ):
        self.likelihood = likelihood
        self.data = likelihood.data
        self.initial_f2b_probas = initial_f2b_probas
        self.global_tolerance = global_tolerance
        self.max_iteration = max_iteration
        self.log_likelihood_by_iteration = [likelihood.get_global_log_likelihood()]
        super().__init__()

    @abstractmethod
    def _search_maximum(self, iteration: int) -> None:
        pass

    def run_iterative_optimization(self) -> None:
        """Run global iterative likelihood optimization, update fail-to-board
        probabilities at each iteration by search_maximum() method.
        The algorithm stops whether global_tolerance is greater than two successive
        log-likelihoods values or iteration number greater than max_iteration.
        """
        print("--------------------------------------------------------------------\n")
        print("\n")
        print("                        Iterative optimization                      \n")
        print("\n")
        print("--------------------------------------------------------------------\n")
        iteration = 0
        self.estimated_f2b_probas = self.initial_f2b_probas
        diff_likelihood = self.log_likelihood_by_iteration[0]
        while (
            iteration < self.max_iteration
            and abs(diff_likelihood) > self.global_tolerance
        ):
            iteration += 1
            self._search_maximum(iteration)
            self.likelihood.update_f2b_probas(self.estimated_f2b_probas)
            self.log_likelihood_by_iteration.append(
                self.likelihood.get_global_log_likelihood()
            )
            diff_likelihood = (
                self.log_likelihood_by_iteration[-1]
                - self.log_likelihood_by_iteration[-2]
            )
        print("--------------------------------------------------------------------\n")
        print("\n")
        print(f"                     End after {iteration} iterations.              \n")
        print("\n")
        print("--------------------------------------------------------------------\n")

    def write_f2b_probas(self, name: str, path: str = "output/f2b_results/") -> None:
        """Write fail-to-board probabilities as .csv file.

        Args:
            - name(str): file name (created if does not exist, overwritten otherwise)
            - path(str): file path, default "f2b/output/f2b_results/"
        """
        with open(path + name + ".csv", "w") as output_file_f2b:
            writer_f2b = writer(output_file_f2b)
            writer_f2b.writerow(self.estimated_f2b_probas)

    def plot_f2b_probas(self) -> None:
        """Plot fail-to-board probabilities."""
        pyplot.subplots(figsize=(10, 5))
        pyplot.plot(self.estimated_f2b_probas)
        pyplot.show()


class LineSearch(MaximumLikelihoodEstimation):
    """Maximum Likelihood Estimation with one dimensionnal line search.

    The maximum likelihood is computed with an iterative optimization algorithm.
    At every iteration, fail-to-board probabilites are updated independently one
    after another in chronological order with a one dimensionnal maximum line search.
    The line search consists in a binary search of a zero of the corresponding
    directionnal derivative by component, starting from 0 as the reference probability
    value.

    Inherit from MaximumLikelihoodEstimation.

    Attributes:
        - data(Data): observations, info of trips and runs
        - likelihood(Likelihood): log-likelihood terms and log-likelihood gradient
        terms for each trip
        - initial_f2b_probas(array): initial values of fail-to-board probabilities
        by run
        - global_tolerance(float): stopping criterion between two successive
        log-likelihoods values in global iterative optimization algorithm
        - max_iteration(int): number of maximum iterations allowed in global
        iterative optimization algorithm
        - local_tolerance(float): stopping criterion between two successive
        directionnal derivatives values in local binary search
        - max_local_binary_iteration(int): number of iterations allowed in one
        dimensionnal maximum binary search
        - log_likelihood_by_iteration(list): list of log-likelihood values by
        iterations
        - estimated_f2b_probas(array): estimated values of fail-to-board probabilities
        by run after optimization


    Methods:
        - run_iterative_optimization(): iterative optimization algorithm
        - write_estimated_f2b_probas(name:str, path:str): write fail-to-board
        probabilities as .csv file
        - plot_estimated_f2b_probas(): plot fail-to-board probabilities
    """

    def __init__(
        self,
        likelihood: Likelihood,
        initial_f2b_probas: ndarray,
        global_tolerance: float,
        max_iteration: int,
        local_tolerance: float,
        max_local_binary_iteration: int,
    ):
        self.max_local_binary_iteration = max_local_binary_iteration
        self.local_tolerance = local_tolerance
        super().__init__(
            likelihood, initial_f2b_probas, global_tolerance, max_iteration
        )

    def _search_maximum(self, iteration: int) -> None:
        """Update fail-to-board probabilities of each run in
        chronological order with one dimensionnal maximum line search.
        Performs a binary search of a zero (not always monotonic) of
        the directionnal derivative by component.

        Args:
            -iteration(int): current iteration number in the global
            iterative optimization algorithm
        """

        print(f"Iteration {iteration}")
        print("Estimation of each component with line search...")
        for run_position in tqdm(range(self.data.runs_number)):
            run_code = self.data.runs_chronological_order[run_position]
            f2b_left = array([x for x in self.estimated_f2b_probas])
            f2b_right = array([x for x in self.estimated_f2b_probas])
            f2b_left[run_position] = 0
            # the value 1 is out of the likelihood bounds
            f2b_right[run_position] = 0.999

            self.likelihood.update_f2b_probas(f2b_left)
            derivative_left = self.likelihood.get_log_likelihood_derivative_by_run(
                run_code
            )

            self.likelihood.update_f2b_probas(f2b_right)
            derivative_right = self.likelihood.get_log_likelihood_derivative_by_run(
                run_code
            )

            if derivative_left * derivative_right >= 0:
                # We assume the derivative does not have a zero for this component.
                # We let the f2b proba at zero (initial value).
                continue

            local_derivative = [derivative_left]
            local_iteration = 0
            local_diff_derivative = local_derivative[0]
            while (
                local_iteration < self.max_local_binary_iteration
                and abs(local_diff_derivative) > self.local_tolerance
            ):
                local_iteration += 1
                f2b_middle = array(
                    [
                        (f2b_left[i] + f2b_right[i]) / 2.0
                        for i in range(self.data.runs_number)
                    ]
                )
                self.likelihood.update_f2b_probas(f2b_middle)
                derivative_middle = (
                    self.likelihood.get_log_likelihood_derivative_by_run(run_code)
                )

                if derivative_left * derivative_middle < 0:
                    f2b_right = array([x for x in f2b_middle])
                    derivative_right = derivative_middle
                elif derivative_right * derivative_middle < 0:
                    f2b_left = array([x for x in f2b_middle])
                    derivative_left = derivative_middle
                local_derivative.append(derivative_middle)
                local_diff_derivative = local_derivative[-1] - local_derivative[-2]
            self.estimated_f2b_probas = array([x for x in f2b_middle])


class LineSearchBackward(MaximumLikelihoodEstimation):
    ...


class GridSearch(MaximumLikelihoodEstimation):
    ...


class Powell(MaximumLikelihoodEstimation):
    ...


class ProjectedNewton(MaximumLikelihoodEstimation):
    ...


if __name__ == "__main__":
    origin_station = "NAT"
    destination_stations = ["LYO", "CHL", "AUB", "ETO", "DEF"]
    date = "04/02/2020"
    direction = 2

    data = Data(date, origin_station, destination_stations, direction)
    f2b_probas = array([0.0 for _ in range(data.runs_number)])

    likelihood = Likelihood(data, f2b_probas)
    MLE = LineSearch(
        likelihood=likelihood,
        initial_f2b_probas=f2b_probas,
        global_tolerance=0.1,
        max_iteration=8,
        local_tolerance=0.01,
        max_local_binary_iteration=20,
    )
    MLE.run_iterative_optimization()
    print(MLE.log_likelihood_by_iteration)
    # MLE.write_f2b_probas(f"line_search_{origin_station}")
