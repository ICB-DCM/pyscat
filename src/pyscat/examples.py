"""Examples of optimization problems."""

from __future__ import annotations

from typing import Any

import numpy as np
from pypesto import Objective, Problem


def shubert(x: np.ndarray) -> float:
    # https://www.sfu.ca/~ssurjano/shubert.html
    sum1 = np.sum(
        [i * np.cos((i + 1) * x[0] + i) for i in range(1, 6)], axis=0
    )
    sum2 = np.sum(
        [i * np.cos((i + 1) * x[1] + i) for i in range(1, 6)], axis=0
    )
    return sum1 * sum2


problem_shubert = Problem(
    objective=Objective(fun=shubert, grad=None),
    lb=np.array([4, 4]),
    ub=np.array([7, 7]),
)


def schwefel(x: np.ndarray, d: int = 2) -> float:
    # https://www.sfu.ca/~ssurjano/schwef.html
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))


problem_schwefel = Problem(
    objective=Objective(fun=schwefel, grad=None),
    lb=np.array([-500, -500]),
    ub=np.array([500, 500]),
)


def langermann(x: np.ndarray, m: int = 5) -> float:
    # https://www.sfu.ca/~ssurjano/langer.html
    c = np.array([1, 2, 5, 2, 3])
    a = np.array([[2, 5, 2, 1, 7], [5, 2, 1, 4, 9]])

    return np.sum(
        [
            c[i]
            * np.exp(-np.sum((x - a[:, i]) ** 2) / np.pi)
            * np.cos(np.pi * np.sum((x - a[:, i]) ** 2))
            for i in range(m)
        ]
    )


problem_langermann = Problem(
    objective=Objective(fun=langermann, grad=None),
    lb=np.array([0, 0]),
    ub=np.array([10, 10]),
)

problem_info: dict[str, dict[str, Any]] = {
    "Shubert": {
        "global_best": -186.73090883102392,
        "problem": problem_shubert,
    },
    "Schwefel": {
        "global_best": 0,
        "problem": problem_schwefel,
    },
    "Langermann": {
        "global_best": -4.2220737923052925,
        "problem": problem_langermann,
    },
}
for problem_id in problem_info:
    problem_info[problem_id]["name"] = problem_id


def xyz(
    problem: Problem, nx: int = 200, ny: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create x,y,z array for 3D plots for the objective function."""
    x = np.linspace(problem.lb[0], problem.ub[0], nx)
    y = np.linspace(problem.lb[1], problem.ub[1], ny)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.objective(np.array([X[i, j], Y[i, j]]))

    return X, Y, Z


def plot_problem(problem: Problem, title=None):
    """Visualize the given problem/objective."""
    import matplotlib.pyplot as plt

    X, Y, Z = xyz(problem)

    fig = plt.figure(figsize=(16, 6))
    if title is not None:
        fig.suptitle(title)

    # 3D plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, Z, cmap="viridis")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$fval$")

    # contour plot
    ax2 = fig.add_subplot(122)
    c = ax2.contourf(X, Y, Z, cmap="viridis")
    plt.colorbar(c, ax=ax2)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")

    plt.show()
