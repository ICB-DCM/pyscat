from __future__ import annotations

import numpy as np
import pypesto
import pytest
import scipy


@pytest.fixture
def rosen_problem() -> pypesto.Problem:
    objective = pypesto.objective.Objective(
        fun=scipy.optimize.rosen, grad=scipy.optimize.rosen_der
    )
    problem = pypesto.Problem(
        objective=objective, lb=0 * np.ones((1, 2)), ub=1 * np.ones((1, 2))
    )
    return problem
