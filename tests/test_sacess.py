import logging
import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pypesto
import pytest
import scipy
from pypesto.history import Hdf5History
from pypesto.store import write_result

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
        problem=problem,
        max_walltime_s=60 / min(8, os.cpu_count()),
        ess_init_args=ess_init_args,
    )
    res = ess.minimize()
    best_fx = res.optimize_result[0].fval

    # close to the approximate global minimum
    assert abs(best_fx - expected_best_fx) < 1e-3, (
        f"Expected best fx ~ {expected_best_fx}, got {best_fx}"
    )
    assert best_fx >= expected_best_fx, (
        f"Best fx {best_fx} is less than expected minimum {expected_best_fx}"
    )

    # check history consistency
    hist = res.optimize_result[0].history
    assert isinstance(hist, pypesto.history.MemoryHistory)
    assert hist.get_fval_trace()[-1] == best_fx, (
        "Best fx in history does not match best fx in optimize_result."
    )
    assert np.all(np.diff(hist.get_fval_trace()) <= 0.0), (
        "Fval trace in history is not monotonically decreasing."
    )
    assert np.all(np.diff(hist.get_time_trace()) >= 0.0), (
        "Time trace in history is not monotonically increasing."
    )
    for x, fx in zip(hist.get_x_trace(), hist.get_fval_trace(), strict=True):
        npt.assert_almost_equal(
            problem.objective(x),
            fx,
            err_msg="History contains inconsistent x and fval traces.",
        )


def test_sacess_result_can_be_stored(rosen_problem, tmp_path):
    """Test that the result of SacessOptimizer can be stored and loaded."""
    problem = rosen_problem
    save_path = Path(tmp_path) / "sacess_result.h5"

    ess = SacessOptimizer(
        problem=problem,
        num_workers=2,
        max_walltime_s=2,
    )
    res = ess.minimize()

    write_result(
        result=res,
        filename=save_path,
        problem=True,
        optimize=True,
        profile=False,
        sample=False,
    )
    Hdf5History.from_history(other=ess.histories[0], file=save_path, id_="bla")

    Hdf5History.from_history(
        other=res.optimize_result[0].history,
        file=save_path,
        id_=res.optimize_result[0].id,
    )


def test_sacess_adaptation(capsys, rosen_problem):
    """Test that adaptation step of the saCeSS optimizer succeeds."""
    problem = rosen_problem

    # based on number of sent solutions
    ess = SacessOptimizer(
        problem=problem,
        num_workers=2,
        max_walltime_s=4,
        sacess_loglevel=logging.DEBUG,
        ess_loglevel=logging.DEBUG,
        options=SacessOptions(
            # trigger frequent adaptation
            # - don't do that in production
            adaptation_min_evals=10**10,
            adaptation_sent_offset=0,
            adaptation_sent_coeff=0,
        ),
    )
    ess.set_local_optimizer(None)
    ess.minimize()
    assert "Updated settings on worker" in capsys.readouterr().err

    # based on number of evaluations since last sent solution
    ess.options.adaptation_min_evals = 0
    ess.options.adaptation_sent_offset = 10**10
    ess.minimize()
    assert "Updated settings on worker" in capsys.readouterr().err


class FunctionOrError:
    """Callable that raises an error every nth invocation."""

    def __init__(self, fun, error_period=100):
        self.counter = 0
        # raise an error every `error_period` calls
        self.error_period = error_period
        self.fun = fun

    def __call__(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.error_period == 0:
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
        max_walltime_s=8,
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


def test_autosave(rosen_problem, tmp_path):
    """Test autosave functionality of SacessOptimizer."""
    problem = rosen_problem
    save_path = Path(tmp_path)
    ess = SacessOptimizer(
        problem=problem,
        num_workers=2,
        max_walltime_s=2,
        autosave_dir=save_path,
    )
    ess.minimize()
    assert sorted(save_path.iterdir()) == [
        ess.get_autosave_path(save_path, i) for i in range(ess.num_workers)
    ], "Missing autosave files"

    for i in range(ess.num_workers):
        autosave_path = ess.get_autosave_path(save_path, i)
        result = pypesto.store.read_result(
            autosave_path, problem=True, optimize=True, with_history=True
        )
        assert len(result.optimize_result) == 1 + ess.worker_results[i].n_local
        # if time exceeded between the last improvement and the last autosave,
        #  the best solution might not be included in the autosave file
        assert result.optimize_result[0].fval >= ess.worker_results[i].fx
        hist = result.optimize_result[0].history
        assert isinstance(hist, pypesto.history.HistoryBase)
        assert hist.get_fval_trace()[-1] <= result.optimize_result[0].fval
