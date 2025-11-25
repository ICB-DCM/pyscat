from pyscat import (
    SacessCmaFactory,
    SacessIpoptFactory,
    SacessFidesFactory,
    ESSOptimizer,
    get_default_ess_options,
    ESSExitFlag,
    SacessOptimizer,
    SacessOptions,
    FunctionEvaluatorMP,
    RefSet,
)
import logging
from pypesto.optimize import FidesOptimizer
import pytest
import pypesto
import numpy as np
import scipy


@pytest.fixture
def problem() -> pypesto.Problem:
    objective = pypesto.objective.Objective(
        fun=scipy.optimize.rosen, grad=scipy.optimize.rosen_der
    )
    problem = pypesto.Problem(
        objective=objective, lb=0 * np.ones((1, 2)), ub=1 * np.ones((1, 2))
    )
    return problem


@pytest.mark.parametrize("ess_type", ["ess", "sacess"])
@pytest.mark.parametrize(
    "local_optimizer",
    [
        None,
        FidesOptimizer(),
        SacessFidesFactory(),
        SacessCmaFactory(),
        SacessIpoptFactory(),
    ],
)
@pytest.mark.flaky(reruns=3)
def test_ess(problem, local_optimizer, ess_type, request):
    if ess_type == "ess":
        ess = ESSOptimizer(
            dim_refset=10,
            max_iter=20,
            local_optimizer=local_optimizer,
            local_n1=15,
            local_n2=5,
            n_threads=2,
            balance=0.5,
        )
    elif ess_type == "sacess":
        if "cr" in request.node.callspec.id or "integrated" in request.node.callspec.id:
            # Not pickleable - incompatible with CESS
            pytest.skip()
        # SACESS with 12 processes
        #  We use a higher number than reasonable to be more likely to trigger
        #  any potential race conditions (gh-1204)
        ess_init_args = get_default_ess_options(num_workers=12, dim=problem.dim)
        for x in ess_init_args:
            x["local_optimizer"] = local_optimizer
        ess = SacessOptimizer(
            max_walltime_s=4,
            sacess_loglevel=logging.DEBUG,
            ess_loglevel=logging.WARNING,
            ess_init_args=ess_init_args,
            options=SacessOptions(
                adaptation_min_evals=500,
                adaptation_sent_offset=10,
                adaptation_sent_coeff=5,
            ),
        )

    else:
        raise ValueError(f"Unsupported ESS type {ess_type}.")

    res = ess.minimize(
        problem=problem,
    )
    assert ess.exit_flag in (ESSExitFlag.MAX_TIME, ESSExitFlag.MAX_ITER)
    print("ESS result: ", res.summary())

    # best values roughly: rosen 7.592e-10
    if local_optimizer:
        assert res.optimize_result[0].fval < 1e-4
    assert res.optimize_result[0].fval < 1


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


def test_ess_refset_repr():
    assert RefSet(10, None).__repr__() == "RefSet(dim=10)"
    assert (
        RefSet(10, None, x=np.zeros(10), fx=np.arange(10)).__repr__()
        == "RefSet(dim=10, fx=[0 ... 9])"
    )


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
