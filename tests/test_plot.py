import numpy as np
from pypesto.history.memory import MemoryHistory
from pyscat.plot import monotonic_history


def test_monotonic_history():
    def create_history(t, fx):
        from pypesto.C import FVAL, TIME

        history = MemoryHistory()
        history._trace[TIME] = t
        history._trace[FVAL] = fx
        assert (history.get_time_trace() == t).all()
        assert (history.get_fval_trace() == fx).all()
        return history

    t = np.arange(5, dtype=float)
    history1 = create_history(t, -t)
    t_mono, fx_mono = monotonic_history([history1, history1])
    assert t_mono.tolist() == history1.get_time_trace()
    assert fx_mono.tolist() == history1.get_fval_trace()

    history2 = create_history(t, -2 * t)
    for histories in (
        [history1, history2],
        [history2, history1],
        [history1, history2, history1],
        [history2, history1, history2],
    ):
        t_mono, fx_mono = monotonic_history(histories)
        assert t_mono.tolist() == history2.get_time_trace()
        assert fx_mono.tolist() == history2.get_fval_trace()

    t = np.arange(0.5, 6.5, dtype=float)
    history3 = create_history(t, 2 - 2 * t)
    for histories in (
        [history1, history3],
        [history3, history1],
        [history1, history3, history1],
    ):
        t_mono, fx_mono = monotonic_history(histories)
        assert t_mono.tolist() == [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5]
        assert fx_mono.tolist() == [
            -0.0,
            -1.0,
            -1.0,
            -2.0,
            -3.0,
            -3.0,
            -5.0,
            -7.0,
            -9.0,
        ]
