"""
Functionality for logging function evaluations during optimization.

This is all very ugly, but I couldn't find a better way to track function
evaluations that works with multiple levels of parallelization,
deep-copying, and pickling.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import functools
from contextlib import contextmanager
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
from typing import Any, Callable
import os
import tempfile
import threading
import heapq
from typing import Optional, Sequence
import logging

import pypesto.ensemble

__all__ = ["EvalLogger", "TopKSelector", "ThresholdSelector"]

logger = logging.getLogger(__name__)


def _reconstruct_method_interceptor(
    obj: Any, method_name: str, handler
) -> MethodInterceptorProxy:
    """
    Module-level factory used by pickle to rebuild the MethodInterceptorProxy.
    """
    return MethodInterceptorProxy(obj, method_name, handler)


class MethodInterceptorProxy:
    own_attrs = ("_obj", "_method_name", "_handler")

    def __init__(self, obj: Any, method_name: str, handler: Callable):
        """
        Wrap `obj` and intercept calls to `method_name`.

        `handler` is called as handler(orig_callable, *args, **kwargs).
        """

        for attr in self.own_attrs:
            if hasattr(obj, attr):
                raise ValueError(f"Cannot proxy object with attribute {attr!r}")

        self._obj = obj
        self._method_name = method_name
        self._handler = handler

    @property
    def __class__(self):
        """
        Make the proxy appear as the wrapped object's class for
        isinstance() checks and similar uses.
        """
        return self._obj.__class__

    def __repr__(self):
        # We might want to forward __repr__ as well, but for debugging it's
        # useful to see that this is a proxy.
        return f"<Proxy of {repr(self._obj)}>"

    def __getattr__(self, name):
        # forward everything except the intercepted method
        attr = getattr(self._obj, name)
        if name == self._method_name and callable(attr):

            @functools.wraps(attr)
            def wrapped(*args, **kwargs):
                return self._handler(attr, *args, **kwargs)

            return wrapped
        return attr

    def __setattr__(self, name, value):
        # keep proxy internals on the proxy;
        # forward other sets to wrapped object
        if name in self.own_attrs:
            object.__setattr__(self, name, value)
        else:
            setattr(self._obj, name, value)

    def __call__(self, *args, **kwargs):
        # support calling the proxy itself (intercept __call__)
        if self._method_name == "__call__":
            orig = getattr(self._obj, "__call__", self._obj)
            return self._handler(orig, *args, **kwargs)

        # otherwise, if underlying object is callable and not intercepted,
        #  call it
        orig = getattr(self._obj, "__call__", None)
        if callable(orig):
            return orig(*args, **kwargs)
        raise TypeError(f"{type(self).__name__} object is not callable")

    def __reduce__(self):
        """
        Control pickling: return a top-level callable and args so pickle
        doesn't try to reconstruct the object based on the (overridden)
        `__class__`.
        """
        return (
            _reconstruct_method_interceptor,
            (self._obj, self._method_name, self._handler),
        )

    def __deepcopy__(self, memo):
        """
        Deepcopy the proxy but keep the *original* handler reference.        .
        """
        import copy

        if id(self) in memo:
            return memo[id(self)]
        obj_copy = copy.deepcopy(self._obj, memo)
        proxy_copy = MethodInterceptorProxy(obj_copy, self._method_name, self._handler)
        memo[id(self)] = proxy_copy
        return proxy_copy


@contextmanager
def temp_swap_attr(container: Any, attr_name: str, proxy: Any):
    """
    Temporarily replace `container.attr_name` with `proxy`
    and restore afterwards.
    """
    had_attr = hasattr(container, attr_name)
    old = getattr(container, attr_name, None)
    setattr(container, attr_name, proxy)

    yield

    if had_attr:
        setattr(container, attr_name, old)
    else:
        try:
            delattr(container, attr_name)
        except Exception:
            logger.exception("delattr failed in temp_swap_attr cleanup")
            pass


class EvalLogger:
    """
    Log function evaluations.

    **This is experimental and is likely to change in future releases.**
    """

    def __init__(
        self,
        selector: EvalSelectorBase = None,
        _shared_evals: ListProxy = None,
        _shared_lock: threading.Lock = None,
    ):
        """
        If no `shared_evals` is provided, create a private Manager and a manager.list()
        that will be used to store evaluations. The Manager object is kept
        on the instance for the process lifetime but is excluded from pickling.
        """
        if _shared_evals is None:
            self._manager = Manager()
            self._shared_evals = self._manager.list()
            self._shared_lock = self._manager.Lock()
        else:
            # when passed an existing manager list proxy (e.g. on unpickle)
            self._manager = None
            self._shared_evals = _shared_evals
            self._shared_lock = _shared_lock

        self.selector = selector

    @property
    def evals(self) -> list:
        return list(self._shared_evals)

    def log(self, x: np.ndarray, fx: float) -> None:
        """Log an evaluation.

        :param x: Parameter vector
        :param fx: Function value
        """
        if self.selector is not None and not self.selector.is_running:
            # start as late as possible to avoid issues with forking processes
            #  elsewhere
            self.selector.start_background_ingest(self._shared_evals, self._shared_lock)

        with self._shared_lock:
            self._shared_evals.append((x, fx))

    def _obj_call_wrapper(self, orig_bound, x, *args, **kwargs):
        """
        Instance method wrapper for `Objective.__call__`.

        Must be pickleable, so cannot be a closure.
        """
        # We might have to support different return types here in the future
        # (e.g., residuals). For now, assume scalar objective value.
        fx = orig_bound(x, *args, **kwargs)
        self.log(x, fx)
        return fx

    @contextmanager
    def attach(self, problem: pypesto.Problem):
        """Context manager to attach the `EvalLogger` to an `Objective`.

        :param problem: The problem that contains the objective whose
            evaluations are to be logged.
        """
        proxy = MethodInterceptorProxy(
            problem.objective, "__call__", self._obj_call_wrapper
        )
        with temp_swap_attr(problem, "objective", proxy):
            yield

        if self.selector is not None:
            self.selector.stop_background_ingest()
            # ingest any remaining entries
            self.selector.ingest_from_shared(self._shared_evals, self._shared_lock)

    def __getstate__(self):
        """
        Pickle everything except the Manager and selector.
        """
        return {"_shared_evals": self._shared_evals, "_shared_lock": self._shared_lock}

    def __setstate__(self, state):
        """Restore."""
        self._shared_evals = state.get("_shared_evals")
        self._shared_lock = state.get("_shared_lock")
        self._manager = None
        self.selector = None


class EvalSelectorBase:
    """
    Base class for objective evaluation selectors.

    This is intended for filtering and potentially checkpointing objective
    function evaluations in combination with :class:`EvalLogger`.

    Subclasses must implement :meth:`process` to process the incoming function
    evaluation records. ``EvalSelectorBase`` consumes entries from an
    :class:`EvalLogger` and pass them to `process`.

    :param dim: Problem dimension (length of parameter vector).
    :param path: Optional filesystem path used by subclasses.
    :param dtype: Numpy dtype used for internal numeric storage.
    """

    def __init__(self, dim: int, path: Path | str | None = None, dtype=float):
        """
        Initialize the selector.

        :param dim: Problem dimension (length of parameter vector).
        :param path: Optional filesystem path used by subclasses for persistence.
        :param dtype: Numpy dtype used for internal numeric storage.
        """
        # TODO: consider making that easier to use and inferring dim from first
        #  record? defer initialization until then.
        self.dim = int(dim)
        self.path: Path | None = Path(path) if path is not None else None
        self.dtype = dtype

        #: Threading lock protecting internal mutable state.
        self._lock = threading.Lock()
        #: Background ingestion thread.
        self._bg_thread: threading.Thread | None = None
        #: Event used to request background thread stop.
        self._stop_bg = threading.Event()

    def _normalize_x(self, x: Any) -> np.ndarray:
        """
        Accept list/tuple/np.ndarray and return a contiguous np.ndarray with
        dtype.
        """
        arr = np.asarray(x, dtype=self.dtype)
        if arr.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},), got {arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _key_from_array(arr: np.ndarray) -> tuple:
        """
        Create a hashable key from a parameter vector array.
        """
        return tuple(float(v) for v in arr)

    @property
    def is_running(self) -> bool:
        """Return True if background ingestion thread is running.

        False otherwise.
        """
        return self._bg_thread is not None

    def ingest_from_shared(
        self, shared_list: ListProxy, shared_lock: Optional[threading.Lock]
    ) -> int:
        """
        Consume entries from a shared list.

        :param shared_list: ``multiprocessing.Manager().list()``
            containing records to ingest.
        :param shared_lock: Lock protecting access to `shared_list`.
        :returns: Number of processed entries.
        """
        with shared_lock:
            snap = list(shared_list)
            n = len(snap)
            if n == 0:
                return 0
            shared_list[:] = []

        consumed = 0
        for item in snap:
            if item is None:
                continue
            try:
                x, fx = item
                self.process(x, float(fx))
            except Exception:
                # ignore malformed items / add errors
                pass
            consumed += 1
        return consumed

    def start_background_ingest(
        self, shared_list: ListProxy, shared_lock: threading.Lock, interval: float = 1.0
    ):
        """
        Start a daemon thread that periodically ingests from the shared list.

        :param shared_list: ``multiprocessing.Manager().list()``
            containing records to ingest.
        :param shared_lock: Lock protecting access to `shared_list`.
        :param interval: Time interval between ingests in seconds.
        """
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return

        self._stop_bg.clear()

        def worker():
            while not self._stop_bg.is_set():
                self.ingest_from_shared(shared_list, shared_lock)
                self._stop_bg.wait(interval)

        t = threading.Thread(target=worker, daemon=True)
        self._bg_thread = t
        t.start()

    def stop_background_ingest(self):
        """
        Stop the background ingestion thread if running.
        """
        if self._bg_thread is None:
            return

        self._stop_bg.set()
        self._bg_thread.join(timeout=2.0)
        self._bg_thread = None

    def process(self, x: Sequence[float], fx: float) -> None:
        """
        Process an objective evaluation.
        """
        # TODO: consider changing (x, fx) to a more extendable record type
        raise NotImplementedError()


class TopKSelector(EvalSelectorBase):
    """
    Maintain the K-best unique parameter vectors seen so far.
    """

    def __init__(
        self, *, k: int, dim: int, path: Path | str | None = None, dtype=float
    ):
        """
        Initialize.

        :param k: The number of best entries to keep.
            (Best means lowest function value.)
        :param dim: Problem dimension (length of parameter vector).
        :param path: Optional filesystem path used by subclasses.
        :param dtype: Numpy dtype used for internal numeric storage.
        """
        super().__init__(dim=dim, path=path, dtype=dtype)
        self._k = int(k)

        #: Buffer for parameter vectors.
        self._x = np.empty((self._k, self.dim), dtype=self.dtype)
        #: Buffer for function values.
        self._fx = np.empty(self._k, dtype=self.dtype)
        #: Buffer for pseudo-timestamps. Currently: insertion order.
        self._ts = np.empty(self._k, dtype=np.int64)
        #: Validity mask for slots. (Not all slots may be used yet.)
        self._valid = np.zeros(self._k, dtype=bool)

        #: Min-heap of stored entries to avoid full sorting on insert.
        #  Entries are tuples (-fx, ts, slot)
        self._heap: list[tuple] = []
        #: Pseudo-timestamp counter for entries. Used for stable sorting
        #  of heap entries with equal fx.
        self._counter = 0
        #: Next free slot index in _x, _fx, _ts buffers.
        self._next_slot = 0
        #: stored parameter vectors as hashable keys to enforce uniqueness
        self._seen: set[tuple] = set()

    def process(self, x: Sequence[float], fx: float) -> None:
        """
        Process an objective evaluation.

        Add (x, fx) to top-K if x is not already stored.
        """
        x_arr = self._normalize_x(x)
        key = self._key_from_array(x_arr)

        with self._lock:
            # reject duplicate parameter vectors
            if key in self._seen:
                return

            # determine slot to use
            if self._next_slot < self._k:
                # less than k entries stored: use next free slot
                slot = self._next_slot
                self._next_slot += 1
            else:
                # k entries already stored: evict the worst if new is better.
                # worst is at top of min-heap (as -fx)
                worst_fx = -self._heap[0][0]
                if fx >= worst_fx:
                    # worse than or equal to worst stored: reject
                    return
                # remove worst from heap and seen-set
                _, _, slot = heapq.heappop(self._heap)
                old_key = self._key_from_array(self._x[slot])
                self._seen.discard(old_key)

            # store new entry
            ts = self._counter
            self._counter += 1

            self._x[slot] = x_arr
            self._fx[slot] = fx
            self._ts[slot] = ts
            self._valid[slot] = True

            heapq.heappush(self._heap, (-float(fx), int(ts), int(slot)))
            self._seen.add(key)
            return

    def snapshot(self) -> dict[str, np.ndarray]:
        """Create a snapshot of the stored entries."""
        with self._lock:
            mask = self._valid
            order = [
                slot
                for _, _, slot in sorted(self._heap, key=lambda t: (-t[0], t[1]))
                if mask[slot]
            ]
            return {
                "x": np.ascontiguousarray(self._x[mask][order]),
                "fx": np.ascontiguousarray(self._fx[mask][order]),
                "ts": np.ascontiguousarray(self._ts[mask][order]),
            }

    # TODO: snapshotting frequency -- who controls? after how many ingests? seconds? ...?
    def save(self):
        """Save the stored entries as numpy ``.npz`` file."""
        if self.path is None:
            return

        snapshot = self.snapshot()

        # first save to a temp file to avoid corrupted files on crashes,
        #  then atomically rename
        dirpath = self.path.parent
        fd, tmpname = tempfile.mkstemp(prefix="._topk_", suffix=".npz", dir=dirpath)
        os.close(fd)
        try:
            np.savez(tmpname, **snapshot)
            os.replace(tmpname, self.path)
        except Exception:
            logger.exception("Failed to save snapshot to %s", self.path)

    def to_ensemble(self) -> pypesto.ensemble.Ensemble:
        """Create a :class:`pypesto.Ensemble` from the stored entries."""
        snapshot = self.snapshot()
        x = snapshot["x"]
        ensemble = pypesto.ensemble.Ensemble(x_vectors=x)
        return ensemble


class ThresholdSelector(EvalSelectorBase):
    """
    Maintain all unique parameter vectors below a certain threshold based
    on the best function value seen so far.
    """

    def __init__(
        self,
        *,
        dim: int,
        mode: str,
        threshold: float,
        path: Optional[str] = None,
        dtype=float,
        _k: int = 100,
        _chunk_size: int = 64,
    ):
        """
        Initialize.

        :param dim: Problem dimension (length of parameter vector).
        :param path: Optional filesystem path for snapshots.
        :param threshold: Threshold for accepting new entries.
            Interpreted according to `mode`.
        :param mode: 'abs' or 'rel' mode for thresholding.
            If 'abs', new entries are accepted if
            ``fx - best_fx <= threshold``.
            If 'rel', new entries are accepted if
            ``|(fx - best_fx) / best_fx | <= threshold``.
        :param dtype: Numpy dtype used for internal numeric storage.
        :param _k: Initial capacity.
        :param _chunk_size: Minimum grow chunk size.
        """
        super().__init__(dim=dim, path=path, dtype=dtype)

        if mode not in ("abs", "rel"):
            raise ValueError(f"Unknown threshold mode {mode!r}")

        self._threshold = float(threshold)
        self._mode = mode
        self._chunk_size = max(1, int(_chunk_size))
        #: Capacity of internal buffers (will grow as needed).
        self._cap = max(self._chunk_size, int(_k))
        #: Current number of stored entries.
        self._len = 0
        self._x = np.empty((self._cap, self.dim), dtype=self.dtype)
        self._fx = np.empty(self._cap, dtype=self.dtype)
        self._ts = np.empty(self._cap, dtype=np.int64)
        self._valid = np.zeros(self._cap, dtype=bool)
        #: Stored parameter vectors as hashable keys to enforce uniqueness.
        #  Maps parameter tuple -> slot index.
        self._seen: dict[tuple, int] = {}
        #: List of free slot indices.
        self._free_slots: list[int] = []
        #: Pseudo-timestamp counter for entries.
        self._counter = 0
        #: Best function value seen so far.
        self._best_fx: float | None = None

    def _grow_to(self, min_cap: int):
        """Grow internal buffers to at least min_cap."""
        # allocate new buffers
        new_cap = max(self._cap * 2, min_cap, self._cap + self._chunk_size)
        x_new = np.empty((new_cap, self.dim), dtype=self.dtype)
        fx_new = np.empty(new_cap, dtype=self.dtype)
        ts_new = np.empty(new_cap, dtype=np.int64)
        valid_new = np.zeros(new_cap, dtype=bool)

        # initialize with old data
        x_new[: self._cap] = self._x
        fx_new[: self._cap] = self._fx
        ts_new[: self._cap] = self._ts
        valid_new[: self._cap] = self._valid

        self._x, self._fx, self._ts = x_new, fx_new, ts_new
        self._valid = valid_new
        self._cap = new_cap

    def _meets_threshold(self, fx: float) -> bool:
        """Check if `fx` meets the threshold criterion."""
        best_fx = self._best_fx
        if best_fx is None:
            return True

        match self._mode:
            case "abs":
                return fx - best_fx <= self._threshold
            case "rel":
                # TODO: handle best_fx == 0 case?
                #  let's see if relative thresholding is actually useful
                return abs((fx - best_fx) / best_fx) <= self._threshold
            case _:
                raise ValueError(f"Unknown threshold mode {self._mode!r}")

    def _store(self, slot: int, x: np.ndarray, fx: float):
        """Store (x, fx) in the slot with the given index."""
        ts = self._counter
        self._counter += 1
        self._x[slot] = x
        self._fx[slot] = fx
        self._ts[slot] = ts
        self._valid[slot] = True
        self._len += 1

    def process(self, x: Sequence[float], fx: float) -> None:
        """
        Process an objective evaluation.

        Add (x, fx) if it meets the threshold and is not a duplicate.
        """
        x_arr = self._normalize_x(x)
        key = self._key_from_array(x_arr)

        with self._lock:
            if key in self._seen:
                return

            if not self._meets_threshold(fx):
                return

            # pick a slot
            if self._free_slots:
                # reuse a slot freed during pruning
                slot = self._free_slots.pop()
            else:
                if self._len >= self._cap:
                    # capacity exhausted -- grow and use next slot
                    old_cap = self._cap
                    self._grow_to(self._cap + 1)
                    slot = old_cap
                else:
                    # find next unused slot among existing capacity
                    slot = 0
                    while slot < self._cap and self._valid[slot]:
                        slot += 1
                    if slot >= self._cap:
                        raise AssertionError(
                            f"Slot {slot} exceeds capacity {self._cap}"
                        )

            # store
            self._store(slot, x_arr, fx)
            self._seen[key] = slot

            # update best and prune if necessary
            if self._best_fx is None or fx < self._best_fx:
                self._best_fx = fx
                self._prune()
            return

    def _prune(self):
        """
        Prune after fx_best changed.

        Remove stored entries that no longer meet the threshold .
        """
        # find entries to remove
        to_remove: list[tuple[tuple, int]] = []
        for key, slot in self._seen.items():
            if not self._meets_threshold(self._fx[slot]):
                to_remove.append((key, slot))

        # remove them
        for key, slot in to_remove:
            # mark slot free
            self._valid[slot] = False
            del self._seen[key]
            self._free_slots.append(slot)
            self._len -= 1

    def snapshot(self):
        """Create a snapshot of the stored entries."""
        with self._lock:
            mask = self._valid
            order = np.argsort(self._fx[mask])
            return {
                "x": np.ascontiguousarray(self._x[mask][order]),
                "fx": np.ascontiguousarray(self._fx[mask][order]),
                "ts": np.ascontiguousarray(self._ts[mask][order]),
            }

    def save(self):
        """Save the stored entries as numpy ``.npz`` file."""
        if self.path is None:
            return

        snapshot = self.snapshot()

        # first save to a temp file to avoid corrupted files on crashes,
        #  then atomically rename
        dirpath = self.path.parent
        fd, tmpname = tempfile.mkstemp(prefix="._thresh_", suffix=".npz", dir=dirpath)
        os.close(fd)
        try:
            np.savez(tmpname, **snapshot)
            os.replace(tmpname, self.path)
        except Exception:
            logger.exception("Failed to save snapshot to %s", self.path)

    def to_ensemble(self) -> pypesto.ensemble.Ensemble:
        """Create a :class:`pypesto.Ensemble` from the stored entries."""
        snapshot = self.snapshot()
        x = snapshot["x"]
        ensemble = pypesto.ensemble.Ensemble(x_vectors=x)
        return ensemble
