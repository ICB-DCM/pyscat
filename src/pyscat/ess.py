"""Enhanced Scatter Search.

See papers on eSS :footcite:p:`EgeaBal2009,EgeaMar2010`,
CeSS :footcite:p:`VillaverdeEge2012`, and saCeSS :footcite:p:`PenasGon2017`.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pypesto.optimize
from pypesto import OptimizerResult, Problem
from pypesto.history import MemoryHistory

from .function_evaluator import FunctionEvaluator, create_function_evaluator
from .refset import RefSet

logger = logging.getLogger(__name__)

__all__ = ["ESSOptimizer", "ESSExitFlag"]


class ESSExitFlag(int, enum.Enum):
    """Scatter search exit flags.

    Exit flags used by :class:`ESSOptimizer` and
    :class:`SacessOptimizer`.
    """

    #: ESS did not run/finish yet
    DID_NOT_RUN = 0
    #: Exited after reaching the maximum number of iterations
    MAX_ITER = -1
    #: Exited after exhausting function evaluation budget
    MAX_EVAL = -2
    #: Exited after exhausting wall-time budget
    MAX_TIME = -3
    #: Termination because of other reasons than exit criteria
    ERROR = -99


class OptimizerFactory(Protocol):
    def __call__(
        self, max_eval: float, max_walltime_s: float
    ) -> pypesto.optimize.Optimizer:
        """Create a new optimizer instance.

        :param max_eval:
            Maximum number of objective functions allowed.
        :param max_walltime_s:
            Maximum walltime in seconds.
        """
        ...


class ESSOptimizer:
    """Enhanced Scatter Search (eSS) global optimization.

    Scatter search is a meta-heuristic for global optimization. A set of points
    (the reference set, RefSet) is iteratively adapted to explore the parameter
    space and to follow promising directions.

    This implementation is based on :footcite:p:`EgeaBal2009,EgeaMar2010`,
    but does not implement any constraint handling beyond box constraints.

    The basic steps of ESS are:

    * Initialization: Generate a diverse set of points (RefSet) in the
      parameter space.
    * Recombination: Generate new points by recombining the RefSet points.
    * Improvement: Improve the RefSet by replacing points with better ones.

    The steps are repeated until a stopping criterion is met.

    ESS is gradient-free, unless a gradient-based local optimizer is used
    (``local_optimizer``).


    :ivar history:
        History of the best values/parameters found so far.
        (Monotonously decreasing objective values.)


    Hyperparameters
    ---------------

    Various hyperparameters control the behavior of eSS.
    Initialization is controlled by ``dim_refset`` and ``n_diverse``.
    Local optimizations are controlled by ``local_optimizer``, ``local_n1``,
    ``local_n2``, and ``balance``.

    Exit criteria
    -------------

    The optimization stops if any of the following criteria are met:

    * The maximum number of iterations is reached (``max_iter``).
    * The maximum number of objective function evaluations is reached
      (``max_eval``).
    * The maximum wall-time is reached (``max_walltime_s``).

    One of these criteria needs to be provided.
    Note that the wall-time and function evaluation criteria are not checked
    after every single function evaluation, and thus, the actual number of
    function evaluations may slightly exceed the given value.

    Parallelization
    ---------------

    Objective function evaluations inside :class:`ESSOptimizer` can be
    parallelized using multiprocessing or multithreading by passing a value
    >1 for ``n_procs`` or ``n_threads``, respectively.

    .. seealso::

       :class:`SacessOptimizer`

    .. footbibliography::
    """

    def __init__(
        self,
        *,
        max_iter: int = None,
        dim_refset: int = None,
        local_n1: int = 1,
        local_n2: int = 10,
        balance: float = 0.5,
        local_optimizer: pypesto.optimize.Optimizer
        | OptimizerFactory
        | None = None,
        max_eval=None,
        n_diverse: int = None,
        n_procs=None,
        n_threads=None,
        max_walltime_s=None,
        result_includes_refset: bool = False,
        result_includes_local_solutions: bool = True,
    ):
        r"""Initialize.

        For plausible values of hyperparameters,
        see :footcite:t:`VillaverdeEge2012`.

        :param dim_refset:
            Size of the RefSet. Note that in every iteration at least
            ``dim_refset**2 - dim_refset`` function evaluations will occur.
        :param max_iter:
            Maximum number of eSS iterations.
        :param local_n1:
            Minimum number of iterations before first local search.
            Ignored if ``local_optimizer=None``.
        :param local_n2:
            Minimum number of iterations between consecutive local
            searches. Maximally one local search per performed in each
            iteration. Ignored if ``local_optimizer=None``.
        :param local_optimizer:
            Local optimizer for refinement, or a callable that creates an
            :class:`pypesto.optimize.Optimizer` or ``None`` to skip local
            searches.
            In case of a callable, it will be called with the keyword arguments
            `max_walltime_s` and `max_eval`, which should be passed to the
            optimizer (if supported) to honor the overall budget.
            See :class:`SacessFidesFactory` for an example.
        :param n_diverse:
            Number of samples to choose from to construct the initial RefSet
        :param max_eval:
            Maximum number of objective functions allowed. This criterion is
            only checked once per iteration, not after every objective
            evaluation, so the actual number of function evaluations may exceed
            this value.
        :param max_walltime_s:
            Maximum walltime in seconds. Will only be checked between local
            optimizations and other simulations, and thus, may be exceeded by
            the duration of a local search.
        :param balance:
            Quality vs. diversity balancing factor with
            :math:`0 \leq balance \leq 1`; ``0`` = only quality,
            ``1`` = only diversity.
            Affects the choice of starting points for local searches. I.e.,
            whether local optimization should focus on improving the best
            solutions found so far (quality), or on exploring new regions of
            the parameter space (diversity).
            Ignored if ``local_optimizer=None``.
        :param n_procs:
            Number of parallel processes to use for parallel function
            evaluation. Mutually exclusive with `n_threads`.
        :param n_threads:
            Number of parallel threads to use for parallel function evaluation.
            Mutually exclusive with `n_procs`.
        :param result_includes_refset:
            Whether the :meth:`minimize` result should include the final
            RefSet.
        :param result_includes_local_solutions:
            Whether the :meth:`minimize` result should include the local search
            results (if any).
        """
        if max_eval is None and max_walltime_s is None and max_iter is None:
            # in this case, we'd run forever
            raise ValueError(
                "Either `max_iter`, `max_eval` or `max_walltime_s` "
                "have to be provided."
            )
        if max_eval is None:
            max_eval = np.inf
        if max_walltime_s is None:
            max_walltime_s = np.inf
        if max_iter is None:
            max_iter = np.inf

        # Hyperparameters
        self.local_n1: int = local_n1
        self.local_n2: int = local_n2
        self.max_iter: int = max_iter
        self.max_eval: int = max_eval
        self.dim_refset: int = dim_refset
        self.local_optimizer = local_optimizer
        self.n_diverse: int = n_diverse
        if n_procs is not None and n_threads is not None:
            raise ValueError(
                "`n_procs` and `n_threads` are mutually exclusive."
            )
        self.n_procs: int | None = n_procs
        self.n_threads: int | None = n_threads
        self.balance: float = balance
        # After how many iterations a stagnated solution is to be replaced by
        #  a random one. Default value taken from [EgeaMar2010]_
        self.n_change: int = 20
        # Only perform local search from best solution
        self.local_only_best_sol: bool = False
        self.max_walltime_s = max_walltime_s
        self._initialize()
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}-{id(self)}"
        )
        self._result_includes_refset = result_includes_refset
        self._result_includes_local_solutions = result_includes_local_solutions

    def _initialize(self):
        """(Re-)Initialize."""
        # RefSet
        self.refset: RefSet | None = None
        # Overall best parameters found so far
        self.x_best: np.ndarray | None = None
        # Overall best function value found so far
        self.fx_best: float = np.inf
        # Results from local searches (only those with finite fval)
        # (there is potential to save memory here by only keeping the
        # parameters in memory and not the full result)
        self.local_solutions: list[OptimizerResult] = []
        # Index of current iteration
        self.n_iter: int = 0
        # ESS iteration at which the last local search took place
        # (only local searches with a finite result are counted)
        self.last_local_search_niter: int = 0
        # Whether self.x_best has changed in the current iteration
        self.x_best_has_changed: bool = False
        self.exit_flag: ESSExitFlag = ESSExitFlag.DID_NOT_RUN
        self.evaluator: FunctionEvaluator | None = None
        self._start_time: float | None = None
        self.history: MemoryHistory = MemoryHistory()

    def _initialize_minimize(
        self,
        problem: Problem = None,
        refset: RefSet | None = None,
        start_time: float | None = None,
    ):
        """(Re-)initialize for optimizations.

        Create initial refset, start timer, ... .
        """
        self._initialize()
        self._start_time = (
            start_time if start_time is not None else time.time()
        )

        if (refset is None and problem is None) or (
            refset is not None and problem is not None
        ):
            raise ValueError(
                "Exactly one of `problem` or `refset` has to be provided."
            )

        problem = problem if problem else refset.evaluator.problem

        if problem.x_guesses.shape[0]:
            # We'll use problem.startpoint_method to sample random points
            #  later on. Depending on the startpoint method, this will return
            #  the provided guesses, meaning that we'll always get the same
            #  points. This means, we won't explore the parameter space, or
            #  potentially even get stuck if the provided guesses are not
            #  evaluable.
            raise ValueError(
                "Providing startpoints in `problem.x_guesses` "
                f"is not supported by {self.__class__.__name__}. "
                "Unset `problem.x_guesses`."
            )

        _check_valid_bounds(problem)

        # generate initial RefSet if not provided
        if refset is None:
            if self.dim_refset is None:
                raise ValueError(
                    "Either refset or dim_refset have to be provided."
                )
            self.evaluator = create_function_evaluator(
                problem,
                n_threads=self.n_threads,
                n_procs=self.n_procs,
            )

            # Initial RefSet generation
            # [EgeaMar2010]_ 2.1
            self.refset = RefSet.from_random(
                dim=self.dim_refset,
                n_diverse=self.n_diverse or 10 * problem.dim,
                evaluator=self.evaluator,
            )
        else:
            self.refset = refset

        self.evaluator = self.refset.evaluator
        self.x_best = np.full(
            shape=(self.evaluator.problem.dim,), fill_value=np.nan
        )
        # initialize global best from initial refset
        for x, fx in zip(self.refset.x, self.refset.fx, strict=False):
            self._maybe_update_global_best(x, fx)

        self._recombination_strategy = DefaultRecombination()
        self._intensification_strategy = GoBeyondStrategy()

    def minimize(
        self,
        problem: Problem = None,
        refset: RefSet | None = None,
    ) -> pypesto.Result:
        """Minimize the given objective.

        :param problem:
            Problem to run ESS on.
        :param refset:
            The initial RefSet or ``None`` to auto-generate.
        """
        self._initialize_minimize(problem=problem, refset=refset)

        # [PenasGon2017]_ Algorithm 1
        while self._keep_going():
            self._do_iteration()

        self._report_final()
        self.history.finalize(exitflag=self.exit_flag.name)
        return self._create_result()

    def _do_iteration(self):
        """Perform an ESS iteration."""
        self.x_best_has_changed = False

        self.refset.sort()
        self._report_iteration()
        self.refset.prune_too_close()

        # Apply combination method to update the RefSet
        x_best_children, fx_best_children = (
            self._recombination_strategy.combine_solutions(
                self.refset, self.evaluator, should_continue=self._keep_going
            )
        )

        # Intensification strategy to further improve the new combinations
        self._intensification_strategy.execute(
            x_best_children,
            fx_best_children,
            self.refset,
            self.evaluator,
            should_continue=self._keep_going,
        )
        for i in range(self.refset.dim):
            # update overall best after intensification?
            self._maybe_update_global_best(
                x_best_children[i], fx_best_children[i]
            )

        # Maybe perform a local search
        if self.local_optimizer is not None and self._keep_going():
            self._do_local_search(x_best_children, fx_best_children)

        # Replace RefSet members by best children where an improvement
        #  was made. replace stuck members by random points.
        for i in range(self.refset.dim):
            if fx_best_children[i] < self.refset.fx[i]:
                self.refset.update(i, x_best_children[i], fx_best_children[i])
            else:
                self.refset.n_stuck[i] += 1
                if self.refset.n_stuck[i] > self.n_change:
                    self.refset.replace_by_random(i)

        self.n_iter += 1

    def _create_result(self) -> pypesto.Result:
        """Create the result object.

        Currently, this returns the overall best value and the final RefSet.
        """
        common_result_fields = {
            "exitflag": self.exit_flag,
            # meaningful? this is the overall time, and identical for all
            #  reported points
            "time": time.time() - self._start_time,
            "n_fval": self.evaluator.n_eval,
            "optimizer": str(self),
        }
        i_result = 0
        result = pypesto.Result(problem=self.evaluator.problem)

        # save global best
        optimizer_result = pypesto.OptimizerResult(
            id=str(i_result),
            x=self.x_best,
            fval=self.fx_best,
            message="Global best",
            **common_result_fields,
        )
        optimizer_result.update_to_full(result.problem)
        result.optimize_result.append(optimizer_result)

        if self._result_includes_local_solutions:
            # save local solutions
            for i, optimizer_result in enumerate(self.local_solutions):
                i_result += 1
                optimizer_result.id = f"Local solution {i}"
                result.optimize_result.append(optimizer_result)

        if self._result_includes_refset:
            # save refset
            for i in range(self.refset.dim):
                i_result += 1
                result.optimize_result.append(
                    pypesto.OptimizerResult(
                        id=str(i_result),
                        x=self.refset.x[i],
                        fval=self.refset.fx[i],
                        message=f"RefSet[{i}]",
                        **common_result_fields,
                    )
                )
                result.optimize_result[-1].update_to_full(result.problem)

        return result

    def _keep_going(self) -> bool:
        """Check exit criteria.

        :returns: ``True`` if not of the exit criteria is met,
            ``False`` otherwise.
        """
        # TODO DW which further stopping criteria: gtol, fatol, frtol?

        if self.n_iter >= self.max_iter:
            self.exit_flag = ESSExitFlag.MAX_ITER
            return False

        if self._get_remaining_eval() <= 0:
            self.exit_flag = ESSExitFlag.MAX_EVAL
            return False

        if self._get_remaining_time() <= 0:
            self.exit_flag = ESSExitFlag.MAX_TIME
            return False

        return True

    def _get_remaining_time(self):
        """Get remaining wall time in seconds."""
        if self.max_walltime_s is None:
            return np.inf
        return self.max_walltime_s - (time.time() - self._start_time)

    def _get_remaining_eval(self):
        """Get remaining function evaluations."""
        if self.max_eval is None:
            return np.inf
        return self.max_eval - self.evaluator.n_eval

    def _do_local_search(
        self, x_best_children: np.ndarray, fx_best_children: np.ndarray
    ) -> None:
        """
        Perform local searches to refine the next generation.

        See [PenasGon2017]_ Algorithm 2.
        """
        if self.local_only_best_sol and self.x_best_has_changed:
            self.logger.debug("Local search only from best point.")
            local_search_x0_fx0_candidates = ((self.x_best, self.fx_best),)
        # first local search?
        elif self.n_iter == self.local_n1:
            self.logger.debug(
                f"First local search from best point due to "
                f"local_n1={self.local_n1}."
            )
            local_search_x0_fx0_candidates = ((self.x_best, self.fx_best),)
        elif (
            self.n_iter >= self.local_n1
            and self.n_iter - self.last_local_search_niter >= self.local_n2
        ):
            priority_order = self.prioritize_local_search_candidates(
                x_best_children,
                fx_best_children,
                self.local_solutions,
                self.balance,
            )
            local_search_x0_fx0_candidates = (
                (x_best_children[i], fx_best_children[i])
                for i in priority_order
            )
        else:
            return

        # actual local search
        # repeat until a finite value is found,
        #  or we don't have any startpoints left
        for (
            local_search_x0,
            local_search_fx0,
        ) in local_search_x0_fx0_candidates:
            optimizer_result = self._local_minimize(
                x0=local_search_x0, fx0=local_search_fx0
            )
            if np.isfinite(optimizer_result.fval):
                self.local_solutions.append(optimizer_result)

                self._maybe_update_global_best(
                    optimizer_result.x[optimizer_result.free_indices],
                    optimizer_result.fval,
                )
                break
        else:
            self.logger.debug(
                "Local search: No finite value found in any local search."
            )
            return

        self.last_local_search_niter = self.n_iter

    @staticmethod
    def prioritize_local_search_candidates(
        x_best_children: np.ndarray,
        fx_best_children: np.ndarray,
        local_solutions: Sequence[OptimizerResult],
        balance: float,
    ) -> np.ndarray:
        """
        Compute an index order for local-search start points that balances
        solution quality and diversity.

        The priority combines a quality ranking (better objective values are
        preferred) and a diversity ranking (candidates further from known local
        optima are preferred). The final priority is a weighted combination of
        the two ranks.

        See [PenasGon2017]_ Algorithm 2 L12-L18.

        :param x_best_children: Array of candidate parameter vectors with shape
            ``(n_candidates, problem_dim)``.
        :param fx_best_children: Array of objective values for the candidates
            with shape ``(n_candidates,)``.
        :param local_solutions: Sequence of existing local ``OptimizerResult``s
            used to compute distances for diversity. May be empty.
        :param balance: Balancing factor in ``[0, 1]``. ``0`` -> prioritize
            quality only, ``1`` -> prioritize diversity only.
        :returns: Array of indices into the candidate arrays ordered by
            decreasing priority (i.e., first index = highest priority).
        """
        # rank by fval, smaller is better
        quality_rank = fx_best_children.argsort().argsort()
        # compute minimal distance between the best children and all local
        #  optima found so far
        min_distances = (
            np.fromiter(
                (
                    min(
                        np.linalg.norm(
                            y_i
                            - optimizer_result.x[optimizer_result.free_indices]
                        )
                        for optimizer_result in local_solutions
                    )
                    for y_i in x_best_children
                ),
                dtype=np.float64,
                count=len(x_best_children),
            )
            if len(local_solutions)
            else np.zeros(len(x_best_children))
        )
        # sort by furthest distance to existing local optima
        diversity_rank = min_distances.argsort()[::-1].argsort()
        # compute priority, balancing quality and diversity
        #  (smaller value = higher priority)
        priority = (1 - balance) * quality_rank + balance * diversity_rank
        return np.argsort(priority)

    def _local_minimize(self, x0: np.ndarray, fx0: float) -> OptimizerResult:
        """Perform a local search from the given startpoint."""
        max_walltime_s = self._get_remaining_time()
        max_eval = self._get_remaining_eval()
        # If we are out of budget, return a dummy result.
        # This prevents issues with optimizers that fail if there is no budget
        # (E.g., Ipopt).
        if max_walltime_s < 1 or max_eval < 1:
            msg = "No time or function evaluations left for local search."
            self.logger.info(msg)
            return OptimizerResult(
                id="0",
                x=x0,
                fval=np.inf,
                message=msg,
                n_fval=0,
                n_grad=0,
                time=0,
                history=None,
            )

        # create optimizer instance if necessary
        if isinstance(self.local_optimizer, pypesto.optimize.Optimizer):
            optimizer = self.local_optimizer
            # added in pypesto 0.5.8
            if (
                hasattr(optimizer, "supports_maxeval")
                and optimizer.supports_maxeval()
            ):
                optimizer.set_maxeval(max_eval)
            if (
                hasattr(optimizer, "supports_maxtime")
                and optimizer.supports_maxtime()
            ):
                optimizer.set_maxtime(max_walltime_s)
        else:
            optimizer = self.local_optimizer(
                max_eval=max_eval,
                max_walltime_s=max_walltime_s,
            )

        # actual local search
        optimizer_result: OptimizerResult = optimizer.minimize(
            problem=self.evaluator.problem,
            x0=x0,
            id="0",
        )

        # add function evaluations during the local search to our function
        #  evaluation counter (NOTE: depending on the setup, we might neglect
        #  gradient evaluations).
        self.evaluator.n_eval += optimizer_result.n_fval

        self.logger.info(
            f"Local search: {fx0} -> {optimizer_result.fval} "
            f"took {optimizer_result.time:.3g}s, finished with "
            f"{optimizer_result.exitflag}: {optimizer_result.message}"
        )
        return optimizer_result

    def _maybe_update_global_best(self, x, fx):
        """Update the global best value if the provided value is better."""
        if fx < self.fx_best:
            self.x_best[:] = x
            self.fx_best = fx
            self.x_best_has_changed = True
            self.history.update(
                self.x_best.copy(),
                (0,),
                pypesto.C.MODE_FUN,
                {pypesto.C.FVAL: self.fx_best},
            )

    def _report_iteration(self):
        """Log the current iteration."""
        if self.n_iter == 0:
            self.logger.info("iter | best | nf | refset         | nlocal")

        with np.printoptions(
            edgeitems=5,
            threshold=8,
            linewidth=100000,
            formatter={"float": lambda x: f"{x:.3g}"},
        ):
            self.logger.info(
                f"{self.n_iter:4} | {self.fx_best:+.2E} | "
                f"{self.evaluator.n_eval} "
                f"| {self.refset.fx} | {len(self.local_solutions)}"
            )

    def _report_final(self):
        """Log scatter search summary."""
        with np.printoptions(
            edgeitems=5,
            threshold=10,
            linewidth=100000,
            formatter={"float": lambda x: f"{x:.3g}"},
        ):
            self.logger.info(
                f"-- Final eSS fval after {self.n_iter} iterations, "
                f"{self.evaluator.n_eval} function evaluations: "
                f"{self.fx_best}. "
                f"Exit flag: {self.exit_flag.name}. "
                f"Num local solutions: {len(self.local_solutions)}."
            )
            self.logger.debug(f"Final refset: {np.sort(self.refset.fx)} ")


class RecombinationStrategy(Protocol):
    def combine_solutions(
        self,
        refset: RefSet,
        evaluator: FunctionEvaluator,
        should_continue: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Combine solutions from the RefSet to create the next generation.

        :return: (y, fy) arrays for the next generation
            (shape: refset.dim x problem.dim).
        """


class DefaultRecombination:
    """
    Default ESS recombination scheme.

    Biased hyper-rectangle sampling.
    See [EgeaBal2009]_ Section 3.2 for details.
    """

    def combine(
        self, refset: RefSet, evaluator: FunctionEvaluator, i: int, j: int
    ) -> np.ndarray:
        """Combine RefSet members ``i`` and ``j``.

        Samples a new point from a biased hyper-rectangle derived from the
        given parents, favoring the direction of the better parent.

        :param refset:
            The current sorted RefSet, sorted by quality.
        :param evaluator:
            Function evaluator.
        :param i:
            Index of first RefSet member for recombination.
        :param j:
            Index of second RefSet member for recombination.

        :return: A new parameter vector.
        """
        c1, c2 = self.get_hyper_rect(refset, evaluator, i, j)
        return np.random.uniform(low=c1, high=c2, size=evaluator.problem.dim)

    @staticmethod
    def get_hyper_rect(
        refset: RefSet, evaluator: FunctionEvaluator, i: int, j: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get biased hyper-rectangle based on RefSet members ``i`` and ``j``.
        Assumes that the RefSet is sorted by quality.

        :param refset:
            The current sorted RefSet, sorted by quality.
        :param evaluator:
            Function evaluator.
        :param i:
            Index of first RefSet member for recombination
        :param j:
            Index of second RefSet member for recombination

        :return: A tuple (c1, c2) with the lower and upper corner of the
            hyper-rectangle.
        """
        if i == j:
            raise ValueError("i == j")

        x = refset.x

        d = (x[j] - x[i]) / 2.0
        # i < j implies f(x_i) < f(x_j) for the sorted RefSet
        alpha = 1 if i < j else -1
        # beta is a relative rank-based distance between the two parents
        #  0 <= beta <= 1
        beta = (abs(j - i) - 1) / (refset.dim - 2)
        # new hyper-rectangle, biased towards the better parent
        c1 = x[i] - d * (1 + alpha * beta)
        c2 = x[i] + d * (1 - alpha * beta)

        # this will not always yield admissible points -> clip to bounds
        ub, lb = evaluator.problem.ub, evaluator.problem.lb
        c1 = np.fmax(np.fmin(c1, ub), lb)
        c2 = np.fmax(np.fmin(c2, ub), lb)

        return c1, c2

    def combine_solutions(
        self,
        refset: RefSet,
        evaluator: FunctionEvaluator,
        should_continue: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine solutions and evaluate.

        Creates the next generation from the RefSet by pair-wise combination
        of all RefSet members. Creates ``RefSet.dim ** 2 - RefSet.dim`` new
        parameter vectors, tests them, and keeps the best child of each parent.

        :returns:
            * y:
                The next generation of parameter vectors
                (`dim_refset` x `dim_problem`).
            * fy:
                The objective values corresponding to the parameters in `y`.
        """
        dim = refset.dim
        p_dim = evaluator.problem.dim
        # arrays for the next generation
        y = np.zeros((dim, p_dim))
        fy = np.full(dim, np.inf)

        for i in range(dim):
            # build children from i combined with all other j != i
            # (i.e., `dim_refset**2 - dim_refset` children in total)
            xs_new = np.vstack(
                tuple(
                    self.combine(refset, evaluator, i, j)
                    for j in range(dim)
                    if j != i
                )
            )
            fxs_new = evaluator.multiple(xs_new)
            best_idx = int(np.argmin(fxs_new))
            fy[i] = fxs_new[best_idx]
            y[i] = xs_new[best_idx]

            if should_continue is not None and not should_continue():
                break

        return y, fy


class IntensificationStrategy(Protocol):
    def execute(
        self,
        x_best_children: np.ndarray,
        fx_best_children: np.ndarray,
        refset: RefSet,
        evaluator: FunctionEvaluator,
        should_continue: Callable[[], bool] | None = None,
    ) -> None:
        """
        Update arrays `x_best_children`, `fx_best_children`
        for the next generation.

        :param x_best_children:
            Next generation parameter vectors
            (shape: refset.dim x problem.dim).
            Will be updated in-place.
        :param fx_best_children:
            Next generation objective values (shape: refset.dim).
            Will be updated in-place.
        :param refset:
            Current RefSet.
        :param evaluator:
            Function evaluator.
        :param should_continue:
            Callable that returns whether the algorithm should continue.
            If ``None``, the algorithm is assumed to always continue.
        """


class GoBeyondStrategy:
    """
    Go-beyond intensification strategy.

    If a child is better than its parent, intensify search in that
    direction until no further improvement is made.

    See [Egea2009]_ algorithm 1 + section 3.4
    """

    @staticmethod
    def execute(
        x_best_children: np.ndarray,
        fx_best_children: np.ndarray,
        refset: RefSet,
        evaluator: FunctionEvaluator,
        should_continue: Callable[[], bool] | None = None,
    ) -> None:
        """Apply go-beyond strategy."""
        for i in range(refset.dim):
            if fx_best_children[i] >= refset.fx[i]:
                # Offspring is not better than parent
                continue

            # offspring is better than parent
            x_parent = refset.x[i].copy()
            fx_parent = refset.fx[i]
            x_child = x_best_children[i].copy()
            fx_child = fx_best_children[i]
            improvement = 1
            # Multiplier used in determining the hyper-rectangle from which to
            # sample children. Will be increased in case of 2 consecutive
            # improvements.
            # (corresponds to 1/\Lambda in [Egea2009]_ algorithm 1)
            go_beyond_factor = 1
            while fx_child < fx_parent:
                # update best child
                x_best_children[i] = x_child
                fx_best_children[i] = fx_child

                # create new solution, child becomes parent
                # hyper-rectangle for sampling child
                box_lb = x_child - (x_parent - x_child) * go_beyond_factor
                box_ub = x_child
                # clip to bounds
                ub, lb = evaluator.problem.ub, evaluator.problem.lb
                box_lb = np.fmax(np.fmin(box_lb, ub), lb)
                box_ub = np.fmax(np.fmin(box_ub, ub), lb)
                # sample parameters
                x_new = np.random.uniform(low=box_lb, high=box_ub)
                x_parent = x_child
                fx_parent = fx_child
                x_child = x_new
                fx_child = evaluator.single(x_child)

                improvement += 1
                if improvement == 2:
                    go_beyond_factor *= 2
                    improvement = 0

            if should_continue is not None and not should_continue():
                break


def _check_valid_bounds(
    problem: Problem,
) -> None:
    """Check that the problem has valid bounds.

    :param problem:
        Problem to check.
    :raises ValueError:
        If bounds are invalid.
    """
    if problem.lb is None or problem.ub is None:
        raise ValueError(
            "Optimizer requires box constraints (lower and upper bounds), "
            "but None were given."
        )
    if np.any(problem.ub <= problem.lb):
        raise ValueError(
            "Invalid bounds: upper bound must be larger than lower bound "
            "for all parameters."
        )
    if np.any(np.isinf(problem.lb)) or np.any(np.isinf(problem.ub)):
        raise ValueError(
            "Invalid bounds: lower and upper bounds must be finite."
        )
