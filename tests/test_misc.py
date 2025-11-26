from pyscat import (
    SacessCmaFactory,
    SacessIpoptFactory,
    SacessFidesFactory,
    ESSOptimizer,
    get_default_ess_options,
    ESSExitFlag,
    SacessOptimizer,
    SacessOptions,
)
import logging
from pypesto.optimize import FidesOptimizer
import pytest


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
def test_ess(problem, local_optimizer, ess_type):
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
