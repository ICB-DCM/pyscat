from pyscat import (
    ESSOptimizer,
)
from pyscat.function_evaluator import FunctionEvaluatorMP
from pyscat.refset import RefSet
from pypesto.optimize import FidesOptimizer
import pypesto
from pyscat.examples import problem_info
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("problem_info", problem_info.values(), ids=problem_info.keys())
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


def test_ess_multiprocess(problem):
    from fides.constants import Options as FidesOptions

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
    refset = RefSet(
        dim=10,
        evaluator=FunctionEvaluatorMP(
            problem=problem,
            n_procs=4,
        ),
    )
    refset.initialize_random(10 * refset.dim)
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
        x_candidates, fx_candidates, local_solutions=local_solutions, balance=0.0
    )
    npt.assert_array_equal(order, expected)

    # balance = 1 -> ranking by distance to local solutions only
    order = ESSOptimizer.prioritize_local_search_candidates(
        x_candidates, fx_candidates, local_solutions=local_solutions, balance=1.0
    )
    # distances: [1, 0, sqrt(2), sqrt(9 + 4)]
    expected = np.array([3, 2, 0, 1])
    npt.assert_array_equal(order, expected)
