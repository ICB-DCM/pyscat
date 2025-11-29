"""Plotting functions for PyScat."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pypesto


def plot_ess_history(history: pypesto.HistoryBase):
    """Plot the history of the eSS optimization."""
    plt.step(
        history.get_time_trace(),
        history.get_fval_trace(),
        where="post",
    )
    plt.scatter(
        history.get_time_trace(),
        history.get_fval_trace(),
        marker=".",
        label="eSS iterations",
    )
    plt.xlabel("time (s)")
    plt.ylabel("fval")
    plt.title("Best fval during each iteration")
    plt.legend()


def monotonic_history(
    histories: list[pypesto.history.HistoryBase],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the overall monotonically decreasing history from multiple histories.

    :param histories:
        List of histories to merge.
        The histories are expected to have time and function value traces,
        and that the function values are monotonically decreasing within each
        history.

    :return:
        t:
            Time points of overall history.
        fx:
            Objective values of overall history.
    """
    if len(histories) == 0:
        warnings.warn("No histories to process.", stacklevel=2)
        return np.array([], dtype="float"), np.array([], dtype="float")

    for history in histories:
        fvals = history.get_fval_trace()
        if not np.all(fvals[:-1] >= fvals[1:]):
            raise ValueError(
                "Each history is expected to have monotonically "
                "decreasing function values."
            )

    # merge results
    t = np.hstack([history.get_time_trace() for history in histories])
    fx = np.hstack([history.get_fval_trace() for history in histories])
    time_order = np.argsort(t)
    t = t[time_order]
    fx = fx[time_order]

    # get the monotonously decreasing sequence
    monotone = np.where(fx == np.fmin.accumulate(fx))[0]
    t_mono, fx_mono = t[monotone], fx[monotone]

    # remove duplicates
    # convert to np.recarray for multi-level sorting
    # so that we can use np.unique later to remove duplicates
    records = np.rec.fromarrays([t_mono, fx_mono], names="t,fx")
    records.sort(order=["t", "fx"])
    t_mono, unique_idx = np.unique(records.t, return_index=True, sorted=True)
    fx_mono = records.fx[unique_idx]

    # extend from last decrease to last timepoint
    if t_mono[-1] != (t_max := t.max()):
        t_mono = np.append(t_mono, [t_max])
        fx_mono = np.append(fx_mono, [fx_mono.min()])

    return t_mono, fx_mono


def plot_sacess_history(
    histories: list[pypesto.history.HistoryBase],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot `SacessOptimizer` history.

    Plot the history of the best objective values for each
    :class:`SacessOptimizer`
    worker over computation time as step splot.

    :param histories:
        List of histories from different workers as obtained from
        :attr:`SacessOptimizer.histories`.
    :param ax:
        Axes object to use.

    :return: The plot axes. `ax` or a new axes if `ax` was `None`.
    """
    ax = ax or plt.subplot()
    if len(histories) == 0:
        warnings.warn("No histories to plot.", stacklevel=2)

    # plot overall minimum
    t_overall, fx_overall = monotonic_history(histories)
    ax.step(
        t_overall,
        fx_overall,
        linestyle="dotted",
        color="grey",
        where="post",
        label="overall",
        alpha=0.8,
    )

    # plot steps of individual workers
    for worker_idx, history in enumerate(histories):
        x, y = history.get_time_trace(), history.get_fval_trace()
        if len(x) == 0:
            warnings.warn(f"No trace for worker #{worker_idx}.", stacklevel=2)
            continue
        # extend from last decrease to last timepoint
        x = np.append(x, [np.max(t_overall)])
        y = np.append(y, [np.min(y)])
        lines = ax.step(
            x, y, ".-", where="post", label=f"worker {worker_idx}", alpha=0.8
        )
        # Plot last point without marker, unless we actually had
        # an improvement there.
        # The time point of the overall last improvement is appended to
        # all histories, even if redundant,
        # so we can just skip the marker for the last point.
        for line in lines:
            line.set_markevery([True] * (len(x) - 1) + [False])

    ax.legend()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fval")
    ax.set_title("SacessOptimizer convergence")
    return ax
