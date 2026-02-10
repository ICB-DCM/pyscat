"""Various utility functions."""

from __future__ import annotations

import warnings

import numpy as np
import pypesto


def merge_monotonic_histories(
    histories: list[pypesto.history.HistoryBase],
    strict: bool = True,
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

    res = merge_monotonic_traces(
        time_traces=[np.asarray(h.get_time_trace()) for h in histories],
        fx_traces=[np.asarray(h.get_fval_trace()) for h in histories],
        strict=strict,
    )
    return res["time"], res["fx"]


def merge_monotonic_traces(  # noqa: C901
    time_traces: list[np.ndarray],
    fx_traces: list[np.ndarray],
    strict: bool = True,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Get the overall traces with monotonically decreasing function values from
    multiple time / function value / ... traces.

    :param time_traces:
        List of time traces to merge.
    :param fx_traces:
        List of function value traces to merge.
    :param strict:
        If True, enforce strict monotonicity (no equal values).
    :param kwargs: Additional arguments to be sorted and pruned according
        to time and function value.
    :return: Dictionary with keys "time", "fx", and additional keys
        from `kwargs`, each containing an array of values.
    """
    if len(time_traces) != len(fx_traces):
        raise ValueError(
            "Length of time_traces and fx_traces must be the same."
        )

    for k, v in kwargs.items():
        if len(v) != len(time_traces):
            raise ValueError(
                f"Length of kwarg '{k}' does not match "
                "length of time_traces / fx_traces."
            )

    if not time_traces:
        return {
            "time": np.array([]),
            "fx": np.array([]),
            **{k: np.array([]) for k in kwargs},
        }

    all_times = np.hstack(time_traces)
    all_fvals = np.hstack(fx_traces)

    sort_idxs = np.argsort(all_times)
    all_times = all_times[sort_idxs]
    all_fvals = all_fvals[sort_idxs]
    monotone_idxs = np.where(all_fvals == np.fmin.accumulate(all_fvals))[0]

    all_times = all_times[monotone_idxs]
    all_fvals = all_fvals[monotone_idxs]

    # drop duplicate time points, keep best function value for each time point
    # convert to np.recarray for multi-level sorting
    # so that we can use np.unique later to remove duplicates
    records = np.rec.fromarrays([all_times, all_fvals], names="t,fx")
    order = records.argsort(order=["t", "fx"])
    records = records[order]
    if tuple(map(int, np.__version__.split("."))) >= (2, 3, 0):
        # `sorted` argument was added in numpy 2.3.0
        time_mono, unique_idx = np.unique(
            records.t, return_index=True, sorted=True
        )
    else:
        # pre 2.3.0, `unique` output was sorted by default
        time_mono, unique_idx = np.unique(records.t, return_index=True)
    fx_mono = records.fx[unique_idx]

    if strict and len(fx_mono):
        # make strictly monotonic
        strict_mono_idx = [0]
        for i in range(1, len(fx_mono)):
            if fx_mono[i] < fx_mono[strict_mono_idx[-1]]:
                strict_mono_idx.append(i)
        time_mono = time_mono[strict_mono_idx]
        fx_mono = fx_mono[strict_mono_idx]
    else:
        strict_mono_idx = slice(None)

    res = {
        "time": time_mono,
        "fx": fx_mono,
    }

    if not kwargs:
        return res

    # Track which trace each element came from + original index.
    # This is done to defer concatenation of additional arguments until after
    # sorting and monotonicity filtering to avoid unnecessary concatenation
    # of potentially large arrays.
    # This is in particular relevant if the values come from h5py datasets,
    # where this allows to only read the relevant entries from disk
    # instead of concatenating the entire dataset into memory.
    trace_idxs = np.concatenate(
        [np.full(len(t), i) for i, t in enumerate(time_traces)]
    )
    original_indices = np.concatenate([np.arange(len(t)) for t in time_traces])
    # would could as well avoid constructing the full arrays
    # and directly compute the relevant indices for each trace,
    # but this seems simpler
    mask = sort_idxs[monotone_idxs][unique_idx][strict_mono_idx]
    trace_idxs = trace_idxs[mask]
    original_indices = original_indices[mask]

    for k, v_list in kwargs.items():
        res[k] = np.array(
            [
                v_list[i_trace][i_in_trace]
                for i_trace, i_in_trace in zip(
                    trace_idxs, original_indices, strict=True
                )
            ]
        )

    return res
