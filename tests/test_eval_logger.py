import logging

import pypesto.optimize
from pyscat.eval_logger import EvalLogger
from pyscat.function_evaluator import (
    FunctionEvaluator,
    FunctionEvaluatorMT,
    FunctionEvaluatorMP,
)
from pyscat.examples import problem_schwefel
import numpy as np
import numpy.testing as npt
import pickle
from pyscat import SacessOptimizer, ESSOptimizer


def test_eval_logger_with_function_evaluator():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        evaluator = FunctionEvaluator(problem)
        xs = [np.random.default_rng().uniform(problem.lb, problem.ub) for _ in range(5)]
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
        problem = pickle.loads(pickle.dumps(problem))
        evaluator = FunctionEvaluator(problem)
        xs = [np.random.default_rng().uniform(problem.lb, problem.ub) for _ in range(5)]
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
        xs = [np.random.default_rng().uniform(problem.lb, problem.ub) for _ in range(5)]
        fxs = evaluator.multiple(xs)

        # Check that all evaluations were logged -- the order may differ due to multithreading
        npt.assert_equal(len(el.evals), 5)
        logged_dict = {tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals}
        for x, fx in zip(xs, fxs, strict=True):
            npt.assert_array_equal(np.array(logged_dict[tuple(x)]), fx)


def test_eval_logger_with_function_evaluator_mp():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        evaluator = FunctionEvaluatorMP(problem, 5)
        xs = [np.random.default_rng().uniform(problem.lb, problem.ub) for _ in range(5)]
        fxs = evaluator.multiple(xs)

    # Check that all evaluations were logged -- the order may differ due to multiprocessing
    npt.assert_equal(len(el.evals), 5)
    logged_dict = {tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals}
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
    logged_dict = {tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals}
    npt.assert_equal(logged_dict[tuple(best_x)], best_fx)


def test_logger_sacess():
    problem = problem_schwefel

    el = EvalLogger()

    with el.attach(problem):
        optimizer = SacessOptimizer(
            num_workers=4, max_walltime_s=2, sacess_loglevel=logging.WARNING
        )
        res = optimizer.minimize(problem=problem)

    # Check that all evaluations were logged
    assert len(el.evals) == sum(
        worker_result.n_eval for worker_result in optimizer.worker_results
    )
    # Check that best point is among logged evaluations
    best_x, best_fx = res.optimize_result[0].x, res.optimize_result[0].fval
    logged_dict = {tuple(x_logged): fx_logged for x_logged, fx_logged in el.evals}
    npt.assert_equal(logged_dict[tuple(best_x)], best_fx)
