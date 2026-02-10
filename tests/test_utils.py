import numpy as np
import pytest
from pypesto.history.memory import MemoryHistory

from pyscat.utils import merge_monotonic_histories, merge_monotonic_traces


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
    t_mono, fx_mono = merge_monotonic_histories([history1, history1])
    assert t_mono.tolist() == history1.get_time_trace()
    assert fx_mono.tolist() == history1.get_fval_trace()

    history2 = create_history(t, -2 * t)
    for histories in (
        [history1, history2],
        [history2, history1],
        [history1, history2, history1],
        [history2, history1, history2],
    ):
        t_mono, fx_mono = merge_monotonic_histories(histories)
        assert t_mono.tolist() == history2.get_time_trace()
        assert fx_mono.tolist() == history2.get_fval_trace()

    t = np.arange(0.5, 6.5, dtype=float)
    history3 = create_history(t, 2 - 2 * t)
    for histories in (
        [history1, history3],
        [history3, history1],
        [history1, history3, history1],
    ):
        t_mono, fx_mono = merge_monotonic_histories(histories, strict=False)
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

    history4 = create_history(np.array([]), np.array([]))
    t_mono, fx_mono = merge_monotonic_histories([history4])
    assert t_mono.size == 0
    assert fx_mono.size == 0


@pytest.mark.parametrize("strict", [True, False])
def test_merge_monotonic_traces_basic(strict):
    """Test basic merging of monotonic traces."""
    t1 = np.array([1.0, 2.0, 3.0])
    fx1 = np.array([-1.0, -2.0, -3.0])
    t2 = np.array([1.5, 2.5])
    fx2 = np.array([-1.5, -3.5])

    result = merge_monotonic_traces([t1, t2], [fx1, fx2], strict=strict)

    assert "time" in result
    assert "fx" in result
    assert result["time"].tolist() == [1.0, 1.5, 2.0, 2.5]
    assert result["fx"].tolist() == [-1.0, -1.5, -2.0, -3.5]


@pytest.mark.parametrize("strict", [True, False])
def test_merge_monotonic_traces_with_kwargs(strict):
    """Test merging traces with additional keyword arguments."""
    t1 = np.array([1.0, 2.0, 3.0])
    fx1 = np.array([-1.0, -2.0, -3.0])
    other1 = np.array([10, 20, 30])

    t2 = np.array([1.5, 2.5])
    fx2 = np.array([-1.5, -3.5])
    other2 = np.array([15, 35])

    result = merge_monotonic_traces(
        [t1, t2], [fx1, fx2], other=[other1, other2], strict=strict
    )

    assert "time" in result
    assert "fx" in result
    assert "other" in result
    assert result["time"].tolist() == [
        1.0,
        1.5,
        2.0,
        2.5,
    ]
    assert result["fx"].tolist() == [-1.0, -1.5, -2.0, -3.5]
    assert result["other"].tolist() == [10, 15, 20, 35]


def test_merge_monotonic_traces():
    times = [[0, 1, 2, 3], [0, 1, 2, 3]]
    fvals = [[5, 4, 3, 2], [6, 5, 4, 3]]
    traj = merge_monotonic_traces(
        times,
        fvals,
    )
    assert traj["time"].tolist() == [0, 1, 2, 3]
    assert traj["fx"].tolist() == [5, 4, 3, 2]

    traj = merge_monotonic_traces([[5, 4, 3, 2]], [[1, 1, 1, 1]], strict=False)
    assert traj["time"].tolist() == [2, 3, 4, 5]
    assert traj["fx"].tolist() == [1, 1, 1, 1]

    traj = merge_monotonic_traces(
        [[2, 3, 4, 5], [1, 3, 17]], [[1, 1, 1, 1], [1, 1, 0]], strict=True
    )
    assert traj["time"].tolist() == [1, 17]
    assert traj["fx"].tolist() == [1, 0]

    for strict in [True, False]:
        res = merge_monotonic_traces([], [], strict=strict)
        assert list(res.keys()) == ["time", "fx"]
        assert res["time"].tolist() == []
        assert res["fx"].tolist() == []

    traj = merge_monotonic_traces(
        [[3, 2, 1]],
        [[2, 3, 4]],
        x=[
            np.array(
                [
                    [3, 3],
                    [
                        2,
                        2,
                    ],
                    [1, 1],
                ]
            )
        ],
        strict=True,
    )
    assert traj["time"].tolist() == [1, 2, 3]
    assert traj["fx"].tolist() == [4, 3, 2]
    assert (traj["x"] == np.array([[1, 1], [2, 2], [3, 3]])).all()
