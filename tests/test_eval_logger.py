import logging
import pickle
from typing import Any

import numpy as np
import numpy.testing as npt
import pypesto.optimize

from pyscat import ESSOptimizer, SacessOptimizer
from pyscat.eval_logger import EvalLogger, ThresholdSelector, TopKSelector
from pyscat.examples import problem_schwefel
from pyscat.function_evaluator import (
    FunctionEvaluator,
    FunctionEvaluatorMP,
    FunctionEvaluatorMT,
)


def test_eval_logger_with_function_evaluator():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        evaluator = FunctionEvaluator(problem)
        xs = [
            np.random.default_rng().uniform(problem.lb, problem.ub)
            for _ in range(5)
        ]
        fxs = evaluator.multiple(xs)

    # Check that all evaluations were logged
    npt.assert_equal(len(el.evals), 5)
    for (x_logged, fx_logged), x, fx in zip(el.evals, xs, fxs, strict=True):
        npt.assert_array_equal(x_logged, x)
        npt.assert_almost_equal(fx_logged, fx)


def test_eval_logger_with_function_evaluator_survives_pickling():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        problem = pickle.loads(pickle.dumps(problem))  # noqa S301
        evaluator = FunctionEvaluator(problem)
        xs = [
            np.random.default_rng().uniform(problem.lb, problem.ub)
            for _ in range(5)
        ]
        fxs = evaluator.multiple(xs)

    # Check that all evaluations were logged
    npt.assert_equal(len(el.evals), 5)
    for (x_logged, fx_logged), x, fx in zip(el.evals, xs, fxs, strict=True):
        npt.assert_array_equal(x_logged, x)
        npt.assert_almost_equal(fx_logged, fx)


def test_eval_logger_with_function_evaluator_mt():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        evaluator = FunctionEvaluatorMT(problem, 5)
        xs = [
            np.random.default_rng().uniform(problem.lb, problem.ub)
            for _ in range(5)
        ]
        fxs = evaluator.multiple(xs)

    # Check that all evaluations were logged
    #  -- the order may differ due to multithreading
    npt.assert_equal(len(el.evals), 5)
    logged_dict = {
        tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals
    }
    for x, fx in zip(xs, fxs, strict=True):
        npt.assert_array_equal(np.array(logged_dict[tuple(x)]), fx)


def test_eval_logger_with_function_evaluator_mp():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        evaluator = FunctionEvaluatorMP(problem, 5)
        xs = [
            np.random.default_rng().uniform(problem.lb, problem.ub)
            for _ in range(5)
        ]
        fxs = evaluator.multiple(xs)

    # Check that all evaluations were logged
    #  -- the order may differ due to multiprocessing
    npt.assert_equal(len(el.evals), 5)
    logged_dict = {
        tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals
    }
    for x, fx in zip(xs, fxs, strict=True):
        npt.assert_array_equal(np.array(logged_dict[tuple(x)]), fx)


def test_logger_ess():
    problem = problem_schwefel
    el = EvalLogger()

    with el.attach(problem):
        optimizer = ESSOptimizer(
            dim_refset=10,
            max_eval=100,
            local_optimizer=pypesto.optimize.ScipyOptimizer(),
        )
        res = optimizer.minimize(problem=problem)

    # Check that all evaluations were logged
    assert len(el.evals) == res.optimize_result[0].n_fval
    # Check that best point is among logged evaluations
    best_x, best_fx = res.optimize_result[0].x, res.optimize_result[0].fval
    logged_dict = {
        tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals
    }
    npt.assert_equal(logged_dict[tuple(best_x)], best_fx)


def test_logger_sacess():
    """Test logging all function evaluations during optimization."""
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        optimizer = SacessOptimizer(
            problem=problem,
            num_workers=4,
            max_walltime_s=2,
            sacess_loglevel=logging.WARNING,
        )
        res = optimizer.minimize()

    # Check that all evaluations were logged
    assert len(el.evals) == optimizer.n_eval_total
    # Check that best point is among logged evaluations
    best_x, best_fx = res.optimize_result[0].x, res.optimize_result[0].fval
    logged_dict = {
        tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals
    }
    npt.assert_equal(logged_dict[tuple(best_x)], best_fx)

    # ensure that all parameters are unique
    # (catch any issues related to identical RNG states across workers)
    logged_xs = [tuple(x_logged) for x_logged, fx_logged in el.evals]
    assert len(logged_xs) == len(set(logged_xs)), "Parameters are not unique"


class TestEvalLogger(EvalLogger):
    """
    EvalLogger variant that duplicates every logged evaluation into a separate
    archive list that is never consumed by selectors. Useful in tests
    where you need an immutable record of all evaluations.
    """

    __test__ = False

    def __init__(self, selector=None, _shared_evals=None, _archive=None):
        # let base class create _manager and _shared_evals when appropriate
        super().__init__(selector=selector, _shared_evals=_shared_evals)

        if _shared_evals is None:
            # EvalLogger created a Manager;
            #  create an archive list on the same manager
            self._archive = self._manager.list()
        else:
            # on unpickle the archive will be provided via __setstate__;
            # if not, fall back to a plain list (local-only)
            self._archive = _archive

    def log(self, x: Any, fx: float) -> None:
        """
        Append to the regular shared list
        and also to the immutable archive list.
        """
        super().log(x, fx)
        self._archive.append((x, fx))

    @property
    def evals_all(self):
        """Return all archived evaluations (never consumed)."""
        return list(self._archive)

    def __getstate__(self):
        return super().__getstate__() | {
            "_archive": self._archive,
        }

    def __setstate__(self, state):
        super().__setstate__(state)
        self._archive = state.get("_archive")


def test_logger_topk_ess():
    problem = problem_schwefel
    selector = TopKSelector(k=5, dim=problem.dim)
    el = TestEvalLogger(selector=selector)

    with el.attach(problem):
        optimizer = ESSOptimizer(
            dim_refset=10,
            max_eval=100,
            local_optimizer=pypesto.optimize.ScipyOptimizer(),
        )
        res = optimizer.minimize(problem=problem)

    # Check that all evaluations were logged
    assert len(el.evals_all) == res.optimize_result[0].n_fval

    topk = selector.snapshot()
    topk_fx = topk["fx"]
    npt.assert_equal(len(topk_fx), 5)
    # Check that top-k are sorted
    for i in range(4):
        assert topk_fx[i] <= topk_fx[i + 1]
    # Check that top-k are indeed the best k
    fxs_logged = [fx_logged for x_logged, fx_logged in el.evals_all]
    fxs_logged.sort()
    for i in range(5):
        npt.assert_equal(topk_fx[i], fxs_logged[i])


def test_logger_threshold_sacess():
    problem = problem_schwefel
    selector = ThresholdSelector(mode="abs", threshold=1e-2, dim=problem.dim)
    el = TestEvalLogger(selector=selector)

    with el.attach(problem):
        optimizer = SacessOptimizer(
            problem=problem,
            num_workers=4,
            max_walltime_s=2,
            sacess_loglevel=logging.WARNING,
        )
        res = optimizer.minimize()

    # Check that all evaluations were logged
    assert len(el.evals_all) == optimizer.n_eval_total
    best_fx = res.optimize_result[0].fval
    all_fval = [fx_logged for x_logged, fx_logged in el.evals_all]
    assert best_fx == min(all_fval)

    num_expected = sum(1 for fx in all_fval if fx <= best_fx + 1e-2)
    snap = selector.snapshot()
    snap_fx = snap["fx"]
    npt.assert_equal(len(snap_fx), num_expected)

    # Check that all checkpointed evaluations are below threshold
    for fx in snap_fx:
        assert fx <= best_fx + 1e-2
