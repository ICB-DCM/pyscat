"""
Functionality for logging function evaluations during optimization.

This is all very ugly, but I couldn't find a better way to track function
evaluations that works with multiple levels of parallelization,
deep-copying, and pickling.
"""

from __future__ import annotations


import numpy as np
import functools
from contextlib import contextmanager
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
from typing import Any, Callable

import pypesto


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
            pass


class EvalLogger:
    """
    Log function evaluations.

    **This is experimental and is likely to change in future releases.**
    """

    def __init__(self, _shared_evals: ListProxy = None):
        """
        If no `shared_evals` is provided, create a private Manager and a manager.list()
        that will be used to store evaluations. The Manager object is kept
        on the instance for the process lifetime but is excluded from pickling.
        """
        if _shared_evals is None:
            self._manager = Manager()
            self._shared_evals = self._manager.list()
        else:
            # when passed an existing manager list proxy (e.g. on unpickle)
            self._manager = None
            self._shared_evals = _shared_evals

    @property
    def evals(self) -> list:
        return list(self._shared_evals)

    def log(self, x: np.ndarray, fx: float) -> None:
        """Log an evaluation.

        :param x: Parameter vector
        :param fx: Function value
        """
        # TODO: options for filtering:
        #   monotonic only, top N, likelihood ratio, ...
        #   https://github.com/ICB-DCM/pyscat/issues/13
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

    def __getstate__(self):
        """
        Only pickle the manager list proxy;
        exclude the Manager instance itself.
        """
        return {"_shared_evals": self._shared_evals}

    def __setstate__(self, state):
        """
        Restore the shared list proxy from state.
        """
        self._shared_evals = state.get("_shared_evals")
        self._manager = None
