"""Plotting functions for PyScat."""

from __future__ import annotations
import matplotlib.pyplot as plt
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
