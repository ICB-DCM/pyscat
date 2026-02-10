import tempfile

import numpy as np
import numpy.testing as npt
import pypesto
import pytest
from pypesto.history import Hdf5History, HistoryOptions, MemoryHistory
from pypesto.optimize import FidesOptimizer

from pyscat import (
    ESSOptimizer,
)
from pyscat.examples import problem_info
from pyscat.function_evaluator import FunctionEvaluatorMP
from pyscat.refset import RefSet


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "problem_info", problem_info.values(), ids=problem_info.keys()
)
def test_ess_finds_minimum(problem_info):
    expected_best_fx = problem_info["global_best"]
    problem = problem_info["problem"]

    ess = ESSOptimizer(max_walltime_s=5, dim_refset=20)
    res = ess.minimize(problem)
    best_fx = res.optimize_result[0].fval

    assert abs(best_fx - expected_best_fx) < 1e-4, (
        f"Expected best fx ~ {expected_best_fx}, got {best_fx}"
    )
    assert best_fx >= expected_best_fx, (
        f"Best fx {best_fx} is less than expected minimum {expected_best_fx}"
    )

    global_best_hist = res.optimize_result[0].history
    assert isinstance(global_best_hist, MemoryHistory)
    assert global_best_hist.get_fval_trace()[-1] == best_fx


def test_ess_result_contains_refset(rosen_problem):
    problem = rosen_problem
    ess = ESSOptimizer(
        max_walltime_s=2, dim_refset=10, result_includes_refset=True
    )
    res = ess.minimize(problem)

    assert len(res.optimize_result) == 11

    ess = ESSOptimizer(
        max_walltime_s=2, dim_refset=10, result_includes_refset=False
    )
    res = ess.minimize(problem)

    assert len(res.optimize_result) == 1


def test_result_contains_local_solutions(rosen_problem):
    problem = rosen_problem
    ess = ESSOptimizer(
        max_walltime_s=2,
        dim_refset=10,
        local_optimizer=FidesOptimizer(),
        result_includes_local_solutions=True,
        result_includes_refset=False,
    )
    res = ess.minimize(problem)
    assert len(ess.local_solutions) > 0
    assert len(res.optimize_result) == 1 + len(ess.local_solutions)

    ess = ESSOptimizer(
        max_walltime_s=2,
        dim_refset=10,
        local_optimizer=FidesOptimizer(),
        result_includes_local_solutions=False,
        result_includes_refset=False,
    )
    res = ess.minimize(problem)
    assert len(ess.local_solutions) > 0
    assert len(res.optimize_result) == 1


def test_ess_multiprocess(rosen_problem):
    from fides.constants import Options as FidesOptions

    problem = rosen_problem
    # augment objective with parameter prior to check it's copyable
    #  https://github.com/ICB-DCM/pyPESTO/issues/1465
    #  https://github.com/ICB-DCM/pyPESTO/pull/1467
    problem.objective = pypesto.objective.AggregatedObjective(
        [
            problem.objective,
            pypesto.objective.NegLogParameterPriors(
                [
                    pypesto.objective.get_parameter_prior_dict(
                        0, "uniform", [0, 1], "lin"
                    )
                ]
            ),
        ]
    )
    problem.startpoint_method = pypesto.startpoint.UniformStartpoints()

    ess = ESSOptimizer(
        max_iter=20,
        # also test passing a callable as local_optimizer
        local_optimizer=lambda max_walltime_s, **kwargs: FidesOptimizer(
            options={FidesOptions.MAXTIME: max_walltime_s}
        ),
    )
    refset = RefSet.from_random(
        dim=10,
        evaluator=FunctionEvaluatorMP(
            problem=problem,
            n_procs=4,
        ),
        n_diverse=100,
    )
    res = ess.minimize(
        refset=refset,
    )
    print("ESS result: ", res.summary())


def test_prioritize_local_search_candidates():
    x_candidates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 3.0]])
    fx_candidates = np.array([2.0, 1.0, 3.0, 0.0])
    local_solutions = [pypesto.result.OptimizerResult(x=np.array([1.0, 0.0]))]

    # balance = 0 -> ranking by fx only
    # no local solutions
    order = ESSOptimizer.prioritize_local_search_candidates(
        x_candidates, fx_candidates, local_solutions=[], balance=0.0
    )
    expected = np.array([3, 1, 0, 2])
    npt.assert_array_equal(order, expected)
    # with local solutions
    order = ESSOptimizer.prioritize_local_search_candidates(
        x_candidates,
        fx_candidates,
        local_solutions=local_solutions,
        balance=0.0,
    )
    npt.assert_array_equal(order, expected)

    # balance = 1 -> ranking by distance to local solutions only
    order = ESSOptimizer.prioritize_local_search_candidates(
        x_candidates,
        fx_candidates,
        local_solutions=local_solutions,
        balance=1.0,
    )
    # distances: [1, 0, sqrt(2), sqrt(9 + 4)]
    expected = np.array([3, 2, 0, 1])
    npt.assert_array_equal(order, expected)


@pytest.mark.parametrize(
    "n_procs,n_threads", [(None, None), (2, None), (None, 2)]
)
def test_ess_is_deterministic(rosen_problem, n_procs, n_threads):
    """Test that ESSOptimizer is deterministic given the same random seed."""
    problem = rosen_problem

    np.random.seed(0)
    ess1 = ESSOptimizer(
        max_iter=10, dim_refset=15, n_procs=n_procs, n_threads=n_threads
    )
    res1 = ess1.minimize(problem)
    best_x1 = res1.optimize_result[0].x
    best_fx1 = res1.optimize_result[0].fval

    np.random.seed(0)
    ess2 = ESSOptimizer(
        max_iter=10, dim_refset=15, n_procs=n_procs, n_threads=n_threads
    )
    res2 = ess2.minimize(problem)
    best_x2 = res2.optimize_result[0].x
    best_fx2 = res2.optimize_result[0].fval

    npt.assert_array_equal(best_x1, best_x2)
    npt.assert_array_equal(best_fx1, best_fx2)
    npt.assert_array_equal(ess1.refset.x, ess2.refset.x)
    assert ess1.evaluator.n_eval == ess2.evaluator.n_eval


def test_failure_on_invalid_bounds(rosen_problem):
    lb, ub = rosen_problem.lb.copy(), rosen_problem.ub.copy()
    problem = rosen_problem
    ess = ESSOptimizer(max_iter=5, dim_refset=10)

    problem.lb_full[-1] = float("inf")
    with pytest.raises(ValueError, match="bound"):
        ess.minimize(problem)

    problem.lb_full, problem.ub_full = lb.copy(), ub.copy()
    problem.ub_full[-1] = float("-inf")
    with pytest.raises(ValueError, match="bound"):
        ess.minimize(problem)

    problem.lb_full, problem.ub_full = lb.copy(), lb.copy()
    with pytest.raises(ValueError, match="bound"):
        ess.minimize(problem)

    problem.lb_full, problem.ub_full = ub.copy(), lb.copy()
    with pytest.raises(ValueError, match="bounds"):
        ess.minimize(problem)


def test_fail_on_x_guesses(rosen_problem):
    problem = rosen_problem
    problem.set_x_guesses(np.array([[0.5, 0.5], [1.5, 1.5]]))
    with pytest.raises(ValueError, match="x_guesses"):
        ess = ESSOptimizer(max_iter=5, dim_refset=10, max_walltime_s=1)
        ess.minimize(problem)

    problem.set_x_guesses(np.empty((0, problem.dim_full)))
    ess.minimize(problem)


def test_ess_history(rosen_problem):
    """Test passing custom history to ESSOptimizer."""
    problem = rosen_problem
    ess = ESSOptimizer(max_iter=5, dim_refset=10, max_walltime_s=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        history = Hdf5History(
            file=tmpdir + "/history.hdf5",
            id="0",
            options=HistoryOptions(trace_record=True, trace_save_iter=1),
        )
        res = ess.minimize(problem, history=history)
        assert len(history)
        npt.assert_allclose(
            history.get_x_trace()[-1], res.optimize_result[0].x
        )
