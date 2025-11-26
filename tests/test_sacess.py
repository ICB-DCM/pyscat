from pyscat import (
    get_default_ess_options,
    SacessOptimizer,
    SacessOptions,
)
import logging
import pypesto
import numpy as np
import scipy


def test_sacess_adaptation(capsys, problem):
    """Test that adaptation step of the SACESS optimizer succeeds."""
    ess_init_args = get_default_ess_options(
        num_workers=2, dim=problem.dim, local_optimizer=False
    )
    ess = SacessOptimizer(
        max_walltime_s=2,
        sacess_loglevel=logging.DEBUG,
        ess_loglevel=logging.DEBUG,
        ess_init_args=ess_init_args,
        options=SacessOptions(
            # trigger frequent adaptation
            # - don't do that in production
            adaptation_min_evals=0,
            adaptation_sent_offset=0,
            adaptation_sent_coeff=0,
        ),
    )
    ess.minimize(problem)
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
    """Check that SacessOptimizer does not hang if an error occurs on a worker."""
    objective = pypesto.objective.Objective(
        fun=FunctionOrError(scipy.optimize.rosen), grad=scipy.optimize.rosen_der
    )
    problem = pypesto.Problem(
        objective=objective, lb=0 * np.ones((1, 2)), ub=1 * np.ones((1, 2))
    )
    sacess = SacessOptimizer(
        num_workers=2,
        max_walltime_s=2,
        sacess_loglevel=logging.DEBUG,
        ess_loglevel=logging.DEBUG,
    )
    res = sacess.minimize(problem)
    assert isinstance(res, pypesto.Result)
    assert "Intentional error." in capsys.readouterr().err
