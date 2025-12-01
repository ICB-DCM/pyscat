import logging

import pytest
from pypesto.optimize import FidesOptimizer

from pyscat import (
    ESSExitFlag,
    ESSOptimizer,
    SacessCmaFactory,
    SacessFidesFactory,
    SacessIpoptFactory,
    SacessOptimizer,
    SacessOptions,
)


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
def test_ess(rosen_problem, local_optimizer, ess_type):
    problem = rosen_problem
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
        res = ess.minimize(
            problem=problem,
        )

    elif ess_type == "sacess":
        # saCeSS with 12 processes
        #  We use a higher number than reasonable to be more likely to trigger
        #  any potential race conditions (gh-pypesto/1204)
        ess = SacessOptimizer(
            problem=problem,
            num_workers=12,
            max_walltime_s=4,
            sacess_loglevel=logging.DEBUG,
            ess_loglevel=logging.WARNING,
            options=SacessOptions(
                adaptation_min_evals=500,
                adaptation_sent_offset=10,
                adaptation_sent_coeff=5,
            ),
        )
        ess.set_local_optimizer(local_optimizer)
        res = ess.minimize()
    else:
        raise ValueError(f"Unsupported ESS type {ess_type}.")

    assert ess.exit_flag in (ESSExitFlag.MAX_TIME, ESSExitFlag.MAX_ITER)
    print("ESS result: ", res.summary())

    # best values roughly: rosen 7.592e-10
    if local_optimizer:
        assert res.optimize_result[0].fval < 1e-4
    assert res.optimize_result[0].fval < 1
