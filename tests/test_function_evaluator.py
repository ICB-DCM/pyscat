import numpy as np
import pytest

from pyscat.function_evaluator import (
    FunctionEvaluator,
    FunctionEvaluatorMP,
    FunctionEvaluatorMT,
    create_function_evaluator,
)


def objective_sum(x):
    return float(np.sum(x))


def simple_startpoint_method(n_starts=1, problem=None):
    return np.zeros((n_starts, problem.dim))


class SimpleProblem:
    """Minimal problem-like object for tests."""

    def __init__(self, objective, startpoint_method, dim: int = 2):
        self.objective = objective
        self.startpoint_method = startpoint_method
        self.dim = dim


def test_counters(subtests):
    problem = SimpleProblem(
        objective=objective_sum,
        startpoint_method=simple_startpoint_method,
        dim=2,
    )

    for fe in (
        FunctionEvaluator(problem=problem),
        FunctionEvaluatorMT(problem=problem, n_threads=1),
        FunctionEvaluatorMT(problem=problem, n_threads=4),
        FunctionEvaluatorMP(problem=problem, n_procs=1),
        FunctionEvaluatorMP(problem=problem, n_procs=4),
    ):
        with subtests.test(fe=fe):
            assert fe.n_eval == 0

            val = fe.single(np.array([1.0, 2.0]))
            assert pytest.approx(val) == 3.0
            assert fe.n_eval == 1
            fe.single(np.array([1.0, 2.0]))
            assert fe.n_eval == 2

            fe.reset_counter()
            assert fe.n_eval == 0

            xs = [
                np.array([1.0, 0.0]),
                np.array([0.5, 0.5]),
                np.array([2.0, 2.0]),
            ]
            fxs = fe.multiple(xs)
            assert fxs.shape == (3,)
            assert fe.n_eval == len(xs)

            fe.multiple(xs)
            assert fe.n_eval == 2 * len(xs)


def test_random_retries_until_finite():
    # create a startpoint method that returns all-NaN initially,
    #  then valid points
    call_count = 0

    def startpoint_method(n_starts=1, problem=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return np.full((n_starts, 2), np.nan)
        return np.tile(np.array([1.0, 2.0]), (n_starts, 1))

    problem = SimpleProblem(
        objective=objective_sum, startpoint_method=startpoint_method, dim=2
    )
    fe = FunctionEvaluator(problem=problem)

    xs, fxs = fe.multiple_random(3)
    assert xs.shape == (3, 2)
    assert np.isfinite(fxs).all()
    assert fe.n_eval == 6  # 3 from first invalid, 3 from second valid call

    fe.reset_counter()
    call_count = 0
    x, fx = fe.single_random()
    assert x.shape == (2,)
    assert np.isfinite(fx)
    assert fe.n_eval == 2


def test_create_function_evaluator_choices_and_error():
    problem = SimpleProblem(
        objective=objective_sum,
        startpoint_method=simple_startpoint_method,
        dim=2,
    )

    with pytest.raises(ValueError, match="Only one of"):
        create_function_evaluator(problem=problem, n_procs=2, n_threads=2)

    fe_mp = create_function_evaluator(problem=problem, n_procs=1)
    assert isinstance(fe_mp, FunctionEvaluatorMP)

    fe_mt = create_function_evaluator(problem=problem, n_threads=1)
    assert isinstance(fe_mt, FunctionEvaluatorMT)

    fe_default = create_function_evaluator(problem=problem)
    assert isinstance(fe_default, FunctionEvaluatorMT)
