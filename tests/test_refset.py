# python
import numpy as np
import pytest

from pyscat.refset import RefSet


class FakeProblem:
    def __init__(self, dim, lb=None, ub=None):
        self.dim = dim
        self.lb = np.zeros(dim) if lb is None else np.asarray(lb)
        self.ub = np.ones(dim) if ub is None else np.asarray(ub)


class FakeEvaluator:
    """Deterministic fake evaluator for tests.

    - `multiple_random(n)` returns `n` samples and their quadratic fvals.
    - `single_random()` returns one sample and its fval.
    """

    def __init__(self, problem: FakeProblem, seed: int = 0):
        self.problem = problem
        self.rng = np.random.default_rng(seed)
        self.max_failures = 10**10

    def multiple_random(self, n: int):
        x = self.rng.uniform(
            self.problem.lb, self.problem.ub, (n, self.problem.dim)
        )
        fx = np.sum(x**2, axis=1)  # simple convex objective
        return x, fx

    def single_random(self):
        x = self.rng.uniform(
            self.problem.lb, self.problem.ub, self.problem.dim
        )
        fx = float(np.sum(x**2))
        return x, fx


def test_constructor_invalid_combinations():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob)

    with pytest.raises(ValueError, match="at least 3"):
        RefSet(evaluator=ev, fx=np.array([1, 2]), x=np.array([[0, 0], [1, 1]]))


def test_repr_with_and_without_fx():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob)

    # with fx provided
    assert (
        repr(RefSet(evaluator=ev, x=np.zeros(10), fx=np.arange(10)))
        == "RefSet(dim=10, fx=[0 ... 9])"
    )


def test_initialize_from_array_and_random():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob, seed=1)

    x_diverse, fx_diverse = ev.multiple_random(10)
    rs = RefSet.from_diverse(
        dim=6, x_diverse=x_diverse, fx_diverse=fx_diverse, evaluator=ev
    )
    assert rs.dim == 6
    assert rs.x.shape == (rs.dim, prob.dim)
    assert rs.fx.shape == (rs.dim,)

    # first half must be the best `num_best` items from fx_diverse
    order = np.argsort(fx_diverse)
    x_diverse = x_diverse[order]
    fx_diverse = fx_diverse[order]

    num_best = int(rs.dim / 2)
    assert np.allclose(rs.fx[:num_best], fx_diverse[:num_best])
    assert np.allclose(rs.x[:num_best], x_diverse[:num_best])


def test_initialize_from_array_errors():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob)

    x_diverse, fx_diverse = ev.multiple_random(3)  # too few points
    with pytest.raises(
        ValueError,
        match="Cannot create RefSet with dimension 4 from only 3 points",
    ):
        RefSet.from_diverse(
            x_diverse=x_diverse, fx_diverse=fx_diverse, dim=4, evaluator=ev
        )

    # mismatched lengths
    x_diverse, fx_diverse = ev.multiple_random(5)
    with pytest.raises(
        ValueError,
        match="Lengths of `x_diverse` and `fx_diverse` do not match",
    ):
        RefSet.from_diverse(
            x_diverse=x_diverse,
            fx_diverse=fx_diverse[:-1],
            dim=4,
            evaluator=ev,
        )


def test_sort_and_attributes():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob, seed=2)
    x, fx = ev.multiple_random(6)
    rs = RefSet(evaluator=ev, x=x.copy(), fx=fx.copy())
    rs.add_attribute("score", fx.copy())

    rs.sort()
    # fx must be sorted ascending
    assert np.all(rs.fx[:-1] <= rs.fx[1:])
    # attribute must be permuted in same order
    assert np.allclose(rs.fx, rs.attributes["score"])


def test_add_attribute_length_mismatch():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob)
    rs = RefSet.from_random(n_diverse=10, evaluator=ev, dim=4)
    with pytest.raises(ValueError, match="Attribute length does not match"):
        rs.add_attribute("bad", np.zeros(3))


def test_update_and_replace_by_random():
    prob = FakeProblem(dim=2)
    ev = FakeEvaluator(prob, seed=3)
    # initialize with random points
    rs = RefSet.from_random(dim=4, evaluator=ev, n_diverse=8)

    # update
    new_x = np.full(prob.dim, 0.42)
    new_fx = 0.1
    rs.update(1, new_x, new_fx)
    assert np.allclose(rs.x[1], new_x)
    assert rs.fx[1] == pytest.approx(new_fx)
    assert rs.n_stuck[1] == 0

    # replace by random
    rs.replace_by_random(1)
    assert not np.allclose(rs.x[1], new_x) and rs.fx[1] != new_fx
    assert rs.n_stuck[1] == 0


def test_prune_too_close_replaces_close_points():
    prob = FakeProblem(dim=2, lb=np.array([0.0, 0.0]), ub=np.array([1.0, 1.0]))
    ev = FakeEvaluator(prob, seed=4)
    # build refset such that two members are nearly identical
    x = np.array(
        [
            [0.1, 0.1],
            # close to first
            [0.10000001, 0.10000001],
            [0.9, 0.9],
        ]
    )
    fx = np.array([0.5, 0.50000001, 0.02])
    rs = RefSet(evaluator=ev, x=x.copy(), fx=fx.copy())
    # first and second points are below this threshold
    rs.proximity_threshold = 1e-6
    rs.prune_too_close()
    # after pruning, the refset has to be sorted, the second point replaced,
    #   and the first and third points still present
    assert all(
        np.not_equal(np.array([0.10000001, 0.10000001]), xx).all()
        for xx in rs.x
    )
    assert any(np.equal(np.array([0.1, 0.1]), xx).all() for xx in rs.x)
    assert any(np.equal(np.array([0.9, 0.9]), xx).all() for xx in rs.x)
    assert np.all(rs.fx[:-1] <= rs.fx[1:])


def test_resize_shrink_and_grow_and_attributes_preserved():
    prob = FakeProblem(dim=3)
    ev = FakeEvaluator(prob, seed=5)
    # start with dim=6
    rs = RefSet.from_random(dim=6, evaluator=ev, n_diverse=12)
    rs.add_attribute("counter", np.arange(6).astype(float))
    # shrink
    rs.resize(3)
    assert rs.dim == 3
    assert rs.x.shape == (3, prob.dim)
    assert rs.fx.shape == (3,)
    assert rs.attributes["counter"].shape == (3,)
    # grow again
    rs.resize(8)
    assert rs.dim == 8
    assert rs.x.shape == (8, prob.dim)
    assert rs.fx.shape == (8,)
    # attribute must be extended and contain zeros for new entries
    assert rs.attributes["counter"].shape == (8,)
    assert np.all(rs.attributes["counter"][3:] == 0)
