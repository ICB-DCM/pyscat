import logging

import numpy as np
import pypesto
import pytest
import scipy

from pyscat import (
    SacessOptimizer,
    SacessOptions,
    get_default_ess_options,
)
from pyscat.examples import problem_info


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "problem_info", problem_info.values(), ids=problem_info.keys()
)
def test_sacess_finds_minimum(problem_info):
    expected_best_fx = problem_info["global_best"]
    problem = problem_info["problem"]

    ess_init_args = get_default_ess_options(
        num_workers=8, dim=problem.dim, local_optimizer=False
    )
    ess = SacessOptimizer(
        problem=problem, max_walltime_s=6, ess_init_args=ess_init_args
    )
    res = ess.minimize()
    best_fx = res.optimize_result[0].fval

    assert abs(best_fx - expected_best_fx) < 1e-4, (
        f"Expected best fx ~ {expected_best_fx}, got {best_fx}"
    )
    assert best_fx >= expected_best_fx, (
        f"Best fx {best_fx} is less than expected minimum {expected_best_fx}"
    )


def test_sacess_adaptation(capsys, rosen_problem):
    """Test that adaptation step of the saCeSS optimizer succeeds."""
    problem = rosen_problem
    ess = SacessOptimizer(
        problem=problem,
        num_workers=2,
        max_walltime_s=2,
        sacess_loglevel=logging.DEBUG,
        ess_loglevel=logging.DEBUG,
        options=SacessOptions(
            # trigger frequent adaptation
            # - don't do that in production
            adaptation_min_evals=0,
            adaptation_sent_offset=0,
            adaptation_sent_coeff=0,
        ),
    )
    ess.set_local_optimizer(None)
    ess.minimize()
    assert "Updated settings on worker" in capsys.readouterr().err


class FunctionOrError:
    """Callable that raises an error every nth invocation."""

    def __init__(self, fun, error_frequency=100):
        self.counter = 0
        self.error_frequency = error_frequency
        self.fun = fun

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.error_frequency == 0:
            raise RuntimeError("Intentional error.")
        return self.fun(*args, **kwargs)


def test_sacess_worker_error(capsys):
    """
    Check that SacessOptimizer does not hang if an error occurs on a worker.
    """
    objective = pypesto.objective.Objective(
        fun=FunctionOrError(scipy.optimize.rosen),
        grad=scipy.optimize.rosen_der,
    )
    problem = pypesto.Problem(
        objective=objective, lb=0 * np.ones((1, 2)), ub=1 * np.ones((1, 2))
    )
    sacess = SacessOptimizer(
        problem=problem,
        num_workers=2,
        max_walltime_s=2,
        sacess_loglevel=logging.DEBUG,
        ess_loglevel=logging.DEBUG,
    )
    res = sacess.minimize()
    assert isinstance(res, pypesto.Result)
    assert "Intentional error." in capsys.readouterr().err


def test_failure_on_invalid_bounds(rosen_problem):
    lb, ub = rosen_problem.lb.copy(), rosen_problem.ub.copy()
    problem = rosen_problem

    problem.lb_full[-1] = float("inf")
    with pytest.raises(ValueError, match="bound"):
        SacessOptimizer(problem=problem, num_workers=2, max_walltime_s=1)

    problem.lb_full, problem.ub_full = lb.copy(), ub.copy()
    problem.ub_full[-1] = float("-inf")
    with pytest.raises(ValueError, match="bound"):
        SacessOptimizer(problem=problem, num_workers=2, max_walltime_s=1)

    problem.lb_full, problem.ub_full = lb.copy(), lb.copy()
    with pytest.raises(ValueError, match="bound"):
        SacessOptimizer(problem=problem, num_workers=2, max_walltime_s=1)

    problem.lb_full, problem.ub_full = ub.copy(), lb.copy()
    with pytest.raises(ValueError, match="bounds"):
        SacessOptimizer(problem=problem, num_workers=2, max_walltime_s=1)


def test_fail_on_x_guesses(rosen_problem):
    problem = rosen_problem
    problem.set_x_guesses(np.array([[0.5, 0.5], [1.5, 1.5]]))
    with pytest.raises(ValueError, match="x_guesses"):
        SacessOptimizer(problem=problem, num_workers=2, max_walltime_s=1)
