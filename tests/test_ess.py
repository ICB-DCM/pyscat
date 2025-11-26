from pyscat import (
    ESSOptimizer,
)
from pyscat.function_evaluator import FunctionEvaluatorMP
from pyscat.refset import RefSet
from pypesto.optimize import FidesOptimizer
import pypesto


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
