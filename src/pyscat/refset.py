"""ReferenceSet functionality for scatter search."""

from __future__ import annotations

from typing import Any

import numpy as np

from .function_evaluator import FunctionEvaluator, TooManyFailuresError

__all__ = ["RefSet"]


class RefSet:
    """Scatter search reference set.

    :ivar evaluator: Function evaluator
    :ivar x: Parameters in the reference set
    :ivar fx: Function values at the parameters in the reference set
    :ivar n_stuck:
        Counts the number of times a refset member did not lead to an
        improvement in the objective (length: ``dim``).
    """

    def __init__(
        self,
        *,
        x: np.ndarray,
        fx: np.ndarray,
        evaluator: FunctionEvaluator,
    ):
        """Construct.

        :param evaluator:
            Function evaluator
        :param x:
            Initial RefSet parameters (shape: (dim, problem.dim)).
        :param fx:
            Function values corresponding to entries in x (shape: (dim,)).
        """
        if x.shape[0] < 3:
            raise ValueError("RefSet dimension has to be at least 3.")
        self.evaluator = evaluator
        # \epsilon in [PenasGon2017]_
        self.proximity_threshold = 1e-3

        self.x = x
        self.fx = fx

        self.n_stuck = np.zeros(shape=[self.dim], dtype=int)
        self.attributes: dict[Any, np.ndarray] = {}

    @staticmethod
    def from_parameters(
        x: np.ndarray,
        evaluator: FunctionEvaluator,
    ) -> RefSet:
        """Create a RefSet from the given parameters."""
        fx = evaluator.multiple(x)
        return RefSet(evaluator=evaluator, x=x, fx=fx)

    @property
    def dim(self) -> int:
        """Number of points in the RefSet."""
        return self.x.shape[0]

    def __repr__(self) -> str:
        fx = f", fx=[{np.min(self.fx)} ... {np.max(self.fx)}]"
        return f"RefSet(dim={self.dim}{fx})"

    def sort(self) -> None:
        """Sort RefSet by quality."""
        order = np.argsort(self.fx)
        self.fx = self.fx[order]
        self.x = self.x[order]
        self.n_stuck = self.n_stuck[order]
        for attribute_name, attribute_values in self.attributes.items():
            self.attributes[attribute_name] = attribute_values[order]

    @staticmethod
    def from_random(
        *, dim: int, n_diverse: int, evaluator: FunctionEvaluator
    ) -> RefSet:
        """Create an initial reference set from random parameters.

        Sample ``n_diverse`` random points, populate half of the RefSet using
        the best solutions and fill the rest with random points.
        """
        # sample n_diverse points
        x_diverse, fx_diverse = evaluator.multiple_random(n_diverse)
        return RefSet.from_diverse(
            x_diverse=x_diverse,
            fx_diverse=fx_diverse,
            evaluator=evaluator,
            dim=dim,
        )

    @staticmethod
    def from_diverse(
        *,
        dim: int,
        x_diverse: np.ndarray,
        fx_diverse: np.ndarray,
        evaluator: FunctionEvaluator,
    ) -> RefSet:
        """Create an initial reference set using the provided points.

        Populate half of the RefSet using the best given solutions and fill the
        rest with a random selection from the remaining points.
        """
        if len(x_diverse) != len(fx_diverse):
            raise ValueError(
                "Lengths of `x_diverse` and `fx_diverse` do not match."
            )
        if dim > len(x_diverse):
            raise ValueError(
                "Cannot create RefSet with dimension "
                f"{dim} from only {len(x_diverse)} points."
            )

        fx = np.full(shape=(dim,), fill_value=np.inf)
        x = np.full(shape=(dim, evaluator.problem.dim), fill_value=np.nan)

        # create initial refset with 50% best values
        num_best = int(dim / 2)
        order = np.argsort(fx_diverse)
        x[:num_best] = x_diverse[order[:num_best]]
        fx[:num_best] = fx_diverse[order[:num_best]]

        # ... and 50% random points
        random_idxs = np.random.choice(
            order[num_best:], size=dim - num_best, replace=False
        )
        x[num_best:] = x_diverse[random_idxs]
        fx[num_best:] = fx_diverse[random_idxs]

        return RefSet(evaluator=evaluator, x=x, fx=fx)

    def prune_too_close(self):
        """Prune too similar RefSet members.

        Replace a parameter vector if its maximum relative difference to a
        better parameter vector is below the given threshold.

        Assumes RefSet is sorted, and ensures the RefSet remains sorted.
        """
        # Compare [PenasGon2007]
        #  Note that the main text states that distance between the two points
        #  is normalized to the bounds of the search space. However,
        #  Algorithm 1, line 9 normalizes to x_j instead. The accompanying
        #  code does normalize to max(abs(x_i), abs(x_j)).
        # Normalizing to the bounds of the search space seems more reasonable.
        #  Otherwise, for a parameter with bounds [lb, ub],
        #  where (ub-lb)/ub < proximity_threshold, we would never find an
        #  admissible point.
        x = self.x
        ub, lb = self.evaluator.problem.ub, self.evaluator.problem.lb
        width = ub - lb

        def normalize(x):
            """Normalize parameter vector to the bounds of the search space."""
            return (x - lb) / width

        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # check proximity
                # zero-division may occur here
                n_failures = -1
                with np.errstate(divide="ignore", invalid="ignore"):
                    while (
                        np.max(np.abs(normalize(x[i]) - normalize(x[j])))
                        <= self.proximity_threshold
                    ):
                        # too close. replace x_j.
                        x[j], self.fx[j] = self.evaluator.single_random()
                        self.sort()

                        n_failures += 1
                        if n_failures > self.evaluator.max_failures:
                            # prevent infinite loop
                            raise TooManyFailuresError(
                                "Too many failures while trying to find a "
                                "sufficiently different point. Consider "
                                "increasing the proximity threshold or the "
                                "maximum number of failures."
                            )

    def update(self, i: int, x: np.ndarray, fx: float):
        """Update a RefSet entry."""
        self.x[i] = x
        self.fx[i] = fx
        self.n_stuck[i] = 0

    def replace_by_random(self, i: int):
        """Replace the RefSet member with the given index by a random point."""
        self.x[i], self.fx[i] = self.evaluator.single_random()
        self.n_stuck[i] = 0

    def add_attribute(self, name: str, values: np.ndarray):
        """
        Add an attribute array to the refset members.

        An attribute can be any 1D array of the same length as the refset.
        The added array will be sorted together with the refset members.
        """
        if len(values) != self.dim:
            raise ValueError("Attribute length does not match refset length.")
        self.attributes[name] = np.array(values)

    def resize(self, new_dim: int):
        """
        Resize the refset.

        If the dimension does not change, do nothing.
        If size is decreased, drop entries from the end (i.e., the worst
        values, assuming it is sorted). If size is increased, the new
        entries are filled with randomly sampled parameters and the refset is
        sorted.

        NOTE: Any attributes are just truncated or filled with zeros.
        """
        if new_dim == self.dim:
            return

        if new_dim < self.dim:
            # shrink
            self.fx = self.fx[:new_dim]
            self.x = self.x[:new_dim]
            self.n_stuck = self.n_stuck[:new_dim]
            for attribute_name, attribute_values in self.attributes.items():
                self.attributes[attribute_name] = attribute_values[:new_dim]
        else:
            # grow
            n_new = new_dim - self.dim
            new_x, new_fx = self.evaluator.multiple_random(n_new)
            self.fx = np.append(self.fx, new_fx)
            self.x = np.vstack((self.x, new_x))
            self.n_stuck = np.append(self.n_stuck, np.zeros(shape=(n_new,)))
            for attribute_name, attribute_values in self.attributes.items():
                self.attributes[attribute_name] = np.append(
                    attribute_values,
                    np.zeros(shape=n_new, dtype=attribute_values.dtype),
                )
            self.sort()
