from functools import wraps
import json
import time
from typing import Any, Dict


class _TracerScope:
    def __init__(self, tracer: "Tracer", name: str, attrs: Dict[str, Any]) -> None:
        self.tracer = tracer
        self.name = name
        self.attrs = attrs
    def begin(self) -> None:
        self.tracer._tick(self.name, "B", self.attrs)
    def end(self) -> None:
        self.tracer._tick(self.name, "E", {})
    def __enter__(self) -> None:
        self.begin()
    def __exit__(self, type, value, traceback) -> None:
        self.end()


class _ContextScope:
    def __init__(self, tracer: "Tracer", ctx: Dict[str, Any]) -> None:
        self.tracer = tracer
        self.ctx = ctx

    def __enter__(self) -> None:
        self.tracer._push_context(self.ctx)

    def __exit__(self, type, value, traceback) -> None:
        self.tracer._pop_context()

class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self.record = []
        self.cur = None
        self.contexts = []

    def _add_record(self, name: str, phase: str, delta: int, attrs: Dict[str, Any]) -> None:
        attrs["name"] = name
        attrs["ph"] = phase
        attrs["delta"] = delta
        self.record.append(attrs)

    def _tick(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        cur = time.time_ns()
        if self.cur is None:
            delta = 0
        else:
            delta = cur - self.cur
        self.cur = cur

        self._add_record(name, phase, delta, attrs)

    def tick(self, name: str, **attrs: Any) -> None:
        """Record an event."""
        self._tick(name, "i", attrs)

    def scope(self, name: str, **kwargs: Any) -> _TracerScope:
        """Time a scope of code."""
        return _TracerScope(self, name, kwargs)

    def duration(self, name: str, duration: int, **attrs: Any) -> None:
        """Record a duration, in nanoseconds. This should be called at the end of the duration."""
        cur = time.time_ns()
        delta = cur - self.cur - duration
        self.cur = cur

        self._add_record(name, "B", delta, attrs)
        self._add_record(name, "E", duration, {})

    def scoped(self, func):
        """Decorator to time a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.scope(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def context(self, **ctx) -> _ContextScope:
        """Pass parameters around in a context."""
        return _ContextScope(ctx)

    def _push_context(self, ctx) -> None:
        self.contexts.append(ctx)

    def _pop_context(self) -> None:
        self.contexts.pop()

    def get(self, q: str):
        """Query the current context."""
        return self.contexts[-1].get(q)

    def log(self, filename) -> None:
        with open(filename, "w", newline="") as file:
            json.dump(self.record, file, indent=2)


tracers = Tracer()
