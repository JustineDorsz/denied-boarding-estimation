"""
Estimation of fail-to-board probabilities (also referred to as delayed boarding 
probabilities) by train run from AFC and AVL data.
"""

__authors__ = "Justine Dorsz"
__date__ = "2022-10-12"

from abc import ABC, abstractmethod
from csv import writer

from f2b.f2b_estimation.data import Data
from f2b.f2b_estimation.likelihood import Likelihood
from matplotlib import pyplot
from numpy import amax, amin, array, array_equal, linalg, matmul, mean, ndarray
from scipy import optimize
from tqdm import tqdm
from yaml import dump


def projection_on_unit_hypercube(vector: ndarray) -> ndarray:
    """Project a vector on the unit hypercube [0,1]^n."""
    projection = array([vector[i] for i in range(len(vector))])
    for i in range(len(vector)):
        if vector[i] < 0:
            projection[i] = 0
        elif vector[i] >= 1:
            projection[i] = 0.999

    return projection


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
        - run_iterative_optimization(): iterative log-likelihood optimization algorithm
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
        """Run global iterative log likelihood optimization, update fail-to-board
        probabilities at each iteration by _search_maximum() method.
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
        print(f"                   End after {iteration} iterations.               \n")
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


class LineSearchForward(MaximumLikelihoodEstimation):
    """Maximum Likelihood Estimation with one dimensionnal line search componentwise.

    The maximum likelihood is computed with an iterative optimization algorithm.
    At every iteration, fail-to-board probabilites are updated independently one
    after another in chronological order with a one dimensionnal maximum line search.
    The line search consists in a binary search of a zero of the corresponding
    directionnal derivative by component, starting from 0 as the reference probability
    value.

    Inherits from MaximumLikelihoodEstimation.

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
        probabilities as .csv file, and parameters of the estimation as yaml file
        at same location
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

    def _binary_update_by_component(self, run_code: int) -> None:
        """Estimate the best fail-to-board probability of a given run with binary
        search of a zero of the directionnal derivative of the log-likelihood.
        The fail-to-board probabilities of the other runs are unchanged.

        Args:
            -run_code(int): code of the run whose probability is estimated"""

        run_position = self.data.runs_chronological_order.index(run_code)
        f2b_left = array([x for x in self.estimated_f2b_probas])
        f2b_right = array([x for x in self.estimated_f2b_probas])
        f2b_left[run_position] = 0
        # the value 1 is out of the likelihood bounds
        f2b_right[run_position] = 0.999

        self.likelihood.update_f2b_probas(f2b_left)
        derivative_left = self.likelihood.get_log_likelihood_derivative_by_run(run_code)

        self.likelihood.update_f2b_probas(f2b_right)
        derivative_right = self.likelihood.get_log_likelihood_derivative_by_run(
            run_code
        )

        if derivative_left * derivative_right >= 0:
            # We assume the derivative does not have a zero for this component.
            # We let the f2b proba at zero (initial value).
            return

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
            derivative_middle = self.likelihood.get_log_likelihood_derivative_by_run(
                run_code
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

    def _search_maximum(self, iteration: int) -> None:
        """Update fail-to-board probabilities of each run in chronological order with
         one dimensionnal log-likelihood maximum line search. Performs a binary search
        of a zero (not always monotonic) of the directionnal derivative by component.

        Args:
            -iteration(int): current iteration number in the global
            iterative optimization algorithm
        """

        print(f"Iteration {iteration}")
        print("Estimation of each component with line search...")
        for run_code in tqdm(self.data.runs_chronological_order):
            self._binary_update_by_component(run_code)

    def write_f2b_probas(self, name: str, path: str = "output/f2b_results/") -> None:
        """Write optimization parameters as yaml file, and fail-to-board probabilities
        as .csv file.

        Args:
            - name(str): file name of both parameters and fail-to-board probabilities
            (created if does not exist, overwritten otherwise)
            - path(str): files path, default "f2b/output/f2b_results/"
        """
        optimization_parameters = {
            "global_tolerance": self.global_tolerance,
            "max_iteration": self.max_iteration,
            "local_tolerance": self.local_tolerance,
            "max_local_binary_iteration": self.max_local_binary_iteration,
        }
        with open(path + name + ".yaml", "w") as f:
            dump(optimization_parameters, f, sort_keys=False, default_flow_style=False)
            super().write_f2b_probas(name, path)


class LineSearchForwardBackward(LineSearchForward):
    """Maximum Likelihood Estimation with one dimensionnal line search componentwise,
    in chronological and reversed chronological order.

    The maximum likelihood is computed with an iterative optimization algorithm.
    At every iteration, fail-to-board probabilites are updated independently one
    after another with a one dimensionnal maximum line search. The update is performed
    in reversed chronological order every other iteration.
    The line search consists in a binary search of a zero of the corresponding
    directionnal derivative by component, starting from 0 as the reference probability
    value.

    Inherits from LineSearchForward.

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
        probabilities as .csv file, and parameters of the estimation as yaml file
        at same location
        - plot_estimated_f2b_probas(): plot fail-to-board probabilities
    """

    def _search_maximum(self, iteration: int) -> None:
        """Update fail-to-board probabilities of each run with binary search of
        likelihood maximum. The runs are successively treated in chronological order
        and then in reversed chronological order.

        Args:
            -iteration(int): current iteration number in the global
            iterative optimization algorithm
        """

        print(f"Iteration {iteration}")
        if iteration % 2 == 1:
            runs_in_estimation_order = self.data.runs_chronological_order
            print(
                "Estimation of each component in chronological order with line search..."
            )

        else:
            runs_in_estimation_order = reversed(self.data.runs_chronological_order)
            print(
                "Estimation of each component in reversed chronological order with line search..."
            )

        for run_code in tqdm(runs_in_estimation_order):
            super()._binary_update_by_component(run_code)


class GridSearch(MaximumLikelihoodEstimation):
    """Maximum Likelihood Estimation with gradient descent by pairs of fail-to-board
    probabilities associated to successive runs.

    The maximum likelihood is computed with an iterative optimization algorithm.
    At every iteration, fail-to-board probabilites are updated by pairs corresponding
    to successive runs in chronological order, with a bi-dimensional projected
    gradient descent method.

    Inherits from MaximumLikelihoodEstimation.

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
        - max_local_descent_iteration(int): number of iterations allowed for a
        bi-dimensional projected gradient descent
        - max_step_search_iter(int): number of iterations allowed for descent step
        search
        - armijo_constant(int): constant in Armijo condition for sufficiently small
        descent step
        - log_likelihood_by_iteration(list): list of log-likelihood values by
        iterations
        - estimated_f2b_probas(array): estimated values of fail-to-board probabilities
        by run after optimization


    Methods:
        - run_iterative_optimization(): iterative optimization algorithm
        - write_estimated_f2b_probas(name:str, path:str): write fail-to-board
        probabilities as .csv file, and parameters of the estimation as yaml file
        at same location
        - plot_estimated_f2b_probas(): plot fail-to-board probabilities
    """

    def __init__(
        self,
        likelihood: Likelihood,
        initial_f2b_probas: ndarray,
        global_tolerance: float,
        max_iteration: int,
        local_tolerance: float,
        max_local_descent_iteration: int,
        max_step_search_iter: int = 30,
        armijo_constant: float = 0.01,
    ):
        self.max_local_descent_iteration = max_local_descent_iteration
        self.local_tolerance = local_tolerance
        self.max_step_search_iter = max_step_search_iter
        self.armijo_constant = armijo_constant
        super().__init__(
            likelihood, initial_f2b_probas, global_tolerance, max_iteration
        )

    def _find_step_backtracking_line_search(
        self,
        first_run_position: int,
        second_run_position: int,
        descent_direction: ndarray,
    ) -> float:
        """Compute admissible descent step satisfying Armijo condition with iterative
        backtracking line search. The descent is related to a given pair of component
        of the fail-to-board probabilities vector, in a given descent direction.
        The candidate step equals 1 initally, and is divided by 10 until Armijo
        inequality holds, or the number of iterations reached max_step_search_iter.

        Args:
            - first_run_position(int): code of the first component concerned by the
            descent in the fail-to-board probabilities vector
            - second_run_position(int): code of the second component concerned by the
            descent in the fail-to-board probabilities vector
            - descent_direction(ndarray): array of the two components of the descent
            direction

        Return:
            float: descent step
        """

        tau = 1

        # Vector with non zero values only at descent pair components,
        # for size compatibility with f2b_probas
        descent_vector = array([0.0 for _ in range(self.data.runs_number)])
        descent_vector[first_run_position] = descent_direction[0]
        descent_vector[second_run_position] = descent_direction[1]

        f2b_updated_tau = projection_on_unit_hypercube(
            self.estimated_f2b_probas
            + tau * descent_vector / linalg.norm(descent_direction),
        )

        if array_equal(
            f2b_updated_tau,
            self.estimated_f2b_probas,
        ):
            # When the descent direction points out the admissible set
            # and f2b[first_run_position, second_run_position] is already on the
            # boundary, there is no descent update. One returns arbitrary step.
            return tau

        log_likelihood_reference = self.likelihood.get_global_log_likelihood()
        self.likelihood.update_f2b_probas(f2b_updated_tau)
        log_likelihood_at_tau = self.likelihood.get_global_log_likelihood()

        step_iteration = 0
        while (
            step_iteration < self.max_step_search_iter
            and log_likelihood_at_tau
            < log_likelihood_reference
            + self.armijo_constant * tau * linalg.norm(descent_direction)
        ):
            step_iteration += 1
            tau /= 10
            f2b_updated_tau = projection_on_unit_hypercube(
                self.estimated_f2b_probas
                + tau * descent_vector / linalg.norm(descent_direction),
            )
            self.likelihood.update_f2b_probas(f2b_updated_tau)
            log_likelihood_at_tau = self.likelihood.get_global_log_likelihood()
        return tau

    def _projected_gradient_update(
        self, first_run_position: int, second_run_position: int
    ) -> None:
        """Update of fail-to-board probabilities for a pair of successive runs to
        maximize their likelihood's contributions. The optimization is performed
        iteratively with a projected gradient descent method. An admissible descent
        step is computed at each iteration with the method
        _find_step_backtracking_line_search().
        The likelihood's attributes are updated with the estimated fail-to-board
        probabilities.

        Args:
            - first_run_position(int): code of the first component concerned by the
            descent in the fail-to-board probabilities vector
            - second_run_position(int): code of the second component concerned by the
            descent in the fail-to-board probabilities vector

        """

        first_run_code = self.data.runs_chronological_order[first_run_position]
        second_run_code = self.data.runs_chronological_order[second_run_position]
        log_likelihood_by_local_it = [self.likelihood.get_global_log_likelihood()]
        local_iteration = 0
        local_diff_log_likelihood = log_likelihood_by_local_it[0]

        while (
            local_iteration < self.max_local_descent_iteration
            and abs(local_diff_log_likelihood) > self.local_tolerance
        ):
            local_iteration += 1
            descent_direction = array(
                [
                    self.likelihood.get_log_likelihood_derivative_by_run(
                        first_run_code
                    ),
                    self.likelihood.get_log_likelihood_derivative_by_run(
                        second_run_code
                    ),
                ]
            )
            # Take the boundary into account:
            # if the proba is zero and the derivative non positive, then
            # ignore the influence and set the derivative at zero.
            if (
                self.estimated_f2b_probas[first_run_position] == 0
                and descent_direction[0] <= 0
            ):
                descent_direction[0] = 0.0

            if (
                self.estimated_f2b_probas[second_run_position] == 0
                and descent_direction[1] <= 0
            ):
                descent_direction[1] = 0.0
            if descent_direction[0] == 0 and descent_direction[1] == 0:
                return

            tau = self._find_step_backtracking_line_search(
                first_run_position, second_run_position, descent_direction
            )

            # Vector with non zero values only at descent pair components,
            # for size compatibility with f2b_probas
            descent_vector = array([0.0 for i in range(len(data.runs))])
            descent_vector[first_run_position] = descent_direction[0] / linalg.norm(
                descent_direction
            )
            descent_vector[second_run_position] = descent_direction[1] / linalg.norm(
                descent_direction
            )

            self.estimated_f2b_probas = projection_on_unit_hypercube(
                self.estimated_f2b_probas + tau * descent_vector,
            )
            self.likelihood.update_f2b_probas(self.estimated_f2b_probas)
            log_likelihood_by_local_it.append(
                self.likelihood.get_global_log_likelihood()
            )
            local_diff_log_likelihood = (
                log_likelihood_by_local_it[-1] - log_likelihood_by_local_it[-2]
            )
        if local_iteration == self.max_local_descent_iteration:
            print(
                f" The projected gradient algorithm didn't converge for component {first_run_position}, {second_run_position}."
            )

    def _search_maximum(self, iteration: int) -> None:
        """Update all fail-to-board probabilities by pairs of components corresponding
        to successive runs, in chronological order. Each pair of components is
        estimated with a two-dimensional projected gradient descent.

        Args:
            - iteration(int): number of current iteration in global optimization algorithm
        """
        for i in tqdm(range(data.runs_number // 2)):
            first_run_position = 2 * i
            second_run_position = 2 * i + 1
            self._projected_gradient_update(first_run_position, second_run_position)

    def write_f2b_probas(self, name: str, path: str = "output/f2b_results/") -> None:
        """Write optimization parameters as yaml file, and fail-to-board probabilities
        as .csv file.

        Args:
            - name(str): file name of both parameters and fail-to-board probabilities
            (created if does not exist, overwritten otherwise)
            - path(str): files path, default "f2b/output/f2b_results/"
        """
        optimization_parameters = {
            "global_tolerance": self.global_tolerance,
            "max_iteration": self.max_iteration,
            "local_tolerance": self.local_tolerance,
            "max_local_descent_iteration": self.max_local_descent_iteration,
            "armijo_constant": self.armijo_constant,
        }
        with open(path + name + ".yaml", "w") as f:
            dump(optimization_parameters, f, sort_keys=False, default_flow_style=False)
            super().write_f2b_probas(name, path)


class ProjectedNewton(MaximumLikelihoodEstimation):
    """Maximum Likelihood Estimation with a global multi-dimensionnal Newton descent.

    The maximum likelihood is computed with a Newton method completed with a projection
    step of the updated fail-to-board probabilities vector on the unit hypercube.

    Inherits from MaximumLikelihoodEstimation.

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
        probabilities as .csv file, and parameters of the estimation as yaml file
        at same location
        - plot_estimated_f2b_probas(): plot fail-to-board probabilities
    """

    def _search_maximum(self, iteration: int) -> None:
        """Update estimated fail-to-board probabilities with Newton descent to find
         a zero of the gradient. The new fail-to-board probabilities vector is
         projected on unit hypercube before update.

        The hessian matrix of the log-likelihood (involved in the update formula) is
        adapted to take into account the boundary: the columns and rows corresponding
         to fail-to-board probabilities equal to zero and a directionnal derivative
         pointing outside the admissible set are set to zero.

        Args:
            -iteration(int): current iteration number in the global
            iterative optimization algorithm
        """

        gradient_log_likelihood_current = (
            self.likelihood.get_global_log_likelihood_gradient()
        )
        hessian_log_likelihood = self.likelihood.get_global_log_likelihood_hessian()

        # We restrict the hessian to runs satisfying both conditions:
        #   - at least one associated trip,
        #   - if their current estimated fail-to-board proba is zero:
        # directionnal derivatives pointing towards the interior of [0,1].

        runs_with_information = []
        for (run_position, run_code) in enumerate(self.data.runs_chronological_order):
            run_with_information = False
            if self.data.runs[run_code].associated_trips:
                if self.estimated_f2b_probas[run_position] > 0:
                    run_with_information = True

                else:
                    if gradient_log_likelihood_current[run_position] > 0:
                        run_with_information = True

            if run_with_information:
                runs_with_information.append(run_position)

            else:
                hessian_log_likelihood[:, run_position] = 0
                hessian_log_likelihood[run_position, :] = 0

        hessian_inv = linalg.pinv(hessian_log_likelihood)
        descent_vector = -matmul(hessian_inv, gradient_log_likelihood_current)

        self.estimated_f2b_probas = projection_on_unit_hypercube(
            self.estimated_f2b_probas + descent_vector
        )


class Scipy:
    """Maximum Likelihood Estimation with Scipy's minimization function.

    Inherits from MaximumLikelihoodEstimation.
    The main issue of global minimization algorithm is the handling of mixed optimality
    constraints (both boundary and interior points). The difference in magnitude of the
     differential information associated to each parameters is an obstacle for minimum
     finding.

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
    """

    def __init__(
        self,
        likelihood: Likelihood,
        initial_f2b_probas: ndarray,
        global_tolerance: float,
    ):
        self.likelihood = likelihood
        self.data = likelihood.data
        self.initial_f2b_probas = initial_f2b_probas
        self.global_tolerance = global_tolerance
        self.log_likelihood_by_iteration = [likelihood.get_global_log_likelihood()]

    def _minus_log_likelihood_global(self, f2b_probas: ndarray, iteration: ndarray):
        """
        Evaluate and return minus global log-likelihood at a given fail-to-board
         probability vector.

        Args:
            -f2b_probas(ndarray): vector of fail-to-board probabilities at which the
            log-likelihood is evaluated

        Return
            (float) log-likelihood value
        """

        self.estimated_f2b_probas = f2b_probas
        iteration[0] += 1
        print(iteration[0])
        print(f"current f2b vector minimum: {amin(f2b_probas)}")
        print(f"current f2b vector maximum: {amax(f2b_probas)}")
        print("\n")
        self.likelihood.update_f2b_probas(self.estimated_f2b_probas)
        self.log_likelihood_by_iteration.append(
            self.likelihood.get_global_log_likelihood()
        )
        return -self.log_likelihood_by_iteration[-1]

    def _gradient_minus_log_likelihood_global(
        self, f2b_probas: ndarray, iteration: ndarray
    ):
        """
        Evaluate and return the gradient vector of minus the global log-likelihood at
         a given fail-to-board probability vector.

        Args:
            -f2b_probas(ndarray): vector of fail-to-board probabilities at which the
            gradient is evaluated

        Return
            (ndarray) vector with the components of the gradient
        """

        self.likelihood.update_f2b_probas(f2b_probas)
        return -self.likelihood.get_global_log_likelihood_gradient()

    def _hessian_minus_log_likelihood_global(
        self, f2b_probas: ndarray, iteration: ndarray
    ):
        """
        Evaluate and return the hessian matrix of minus the global log-likelihood at a
        given fail-to-board probability vector.

        Args:
            -f2b_probas(ndarray): vector of fail-to-board probabilities at which the
            hessian is evaluated

        Return
            (ndarray) matrix with the components of the hessian
        """
        self.likelihood.update_f2b_probas(f2b_probas)
        return -self.likelihood.get_global_log_likelihood_hessian()

    def run_iterative_optimization(self) -> None:
        """Run global iterative log likelihood optimization with Scipy."""

        print("--------------------------------------------------------------------\n")
        print("\n")
        print("                        Iterative optimization                      \n")
        print("\n")
        print("--------------------------------------------------------------------\n")
        iteration = array([0])
        self.f2b_estimated = optimize.minimize(
            self._minus_log_likelihood_global,
            self.initial_f2b_probas,
            args=(iteration),
            method="Powell",
            tol=self.global_tolerance,
            jac=self._gradient_log_likelihood_global,
            hess=self._hessian_log_likelihood_global,
            bounds=[(0, 0.999) for i in range(len(self.data.runs))],
        ).x

        print("--------------------------------------------------------------------\n")
        print("\n")
        print(f"                   End after {iteration} iterations.               \n")
        print("\n")
        print("--------------------------------------------------------------------\n")
