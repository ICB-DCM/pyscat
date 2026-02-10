"""Various utility functions."""

from __future__ import annotations

import warnings

import numpy as np
import pypesto


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
    if tuple(map(int, np.__version__.split("."))) >= (2, 3, 0):
        # `sorted` argument was added in numpy 2.3.0
        t_mono, unique_idx = np.unique(
            records.t, return_index=True, sorted=True
        )
    else:
        # pre 2.3.0, `unique` output was sorted by default
        t_mono, unique_idx = np.unique(records.t, return_index=True)
    fx_mono = records.fx[unique_idx]

    # extend from last decrease to last timepoint
    if len(t_mono) and t_mono[-1] != (t_max := t.max()):
        t_mono = np.append(t_mono, [t_max])
        fx_mono = np.append(fx_mono, [fx_mono.min()])

    return t_mono, fx_mono
