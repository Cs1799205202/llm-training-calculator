from functools import wraps
import json
import time
import torch
from typing import Any, Dict, List, Optional


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
        self.record: List[Any] = []
        self.cur: int = None
        self.pending_initial_delta: int = None
        self.pending: List[Any] = None
        self.contexts = []

    def _calibrate(self) -> int:
        """Reset the clock and get delta."""
        cur = time.time_ns()
        if self.cur is None:
            delta = 0
        else:
            delta = cur - self.cur
        self.cur = cur
        return delta

    def _add_record(self, attrs: Dict[str, Any]) -> None:
        self.record.append(attrs)

    def _create_record(self, name: str, phase: str, delta: int, attrs: Dict[str, Any]) -> Any:
        attrs["name"] = name
        attrs["ph"] = phase
        attrs["delta"] = delta
        return attrs

    def _add_pending(self, attrs: Dict[str, Any]) -> None:
        self.pending.append(attrs)

    def _add_cuda_event(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        attrs["cuda_event"] = event
        # Do not know the duration yet, so let delta be 0
        self._add_pending(self._create_record(name, phase, 0, attrs))

    def iteration_begin(self) -> None:
        """Start tracing an iteration."""
        delta = self._calibrate()
        self.pending_initial_delta = delta
        self.pending = []
        # Mark the beginning of the iteration
        self._add_cuda_event("iteration", "B", {})

    def is_tracing(self) -> bool:
        return self.pending is not None

    def iteration_end(self) -> None:
        """End tracing an iteration. Note that this performs synchronization."""
        # Mark the end of the iteration
        self._add_cuda_event("iteration", "E", {})
        # Wait for all events to finish
        torch.cuda.synchronize()
        # Get wall clock duration for this iteration
        wall_duration = self._calibrate()

        # Calculate the delta of each event
        delta = self.pending_initial_delta
        cuda_duration = delta
        for i, begin in enumerate(self.pending[:-1]):
            end = self.pending[i + 1]
            begin["delta"] = delta
            # nanoseconds
            delta = int(end["cuda_event"].elapsed_time(begin["cuda_event"]) * 1e6)
            cuda_duration += delta
            del begin["cuda_event"]
            self._add_record(begin)
        end = self.pending[-1]
        end["delta"] = delta
        del end["cuda_event"]
        end["duration_wall"] = wall_duration
        end["duration_cuda"] = cuda_duration
        self._add_record(end)

        self.pending_initial_delta = None
        self.pending = None

    def _tick(self, name: str, phase: str, attrs: Dict[str, Any]) -> None:
        if self.is_tracing():
            self._add_cuda_event(name, phase, attrs)

    def tick(self, name: str, **attrs: Any) -> None:
        """Record an event."""
        self._tick(name, "i", attrs)

    def scope(self, name: str, **kwargs: Any) -> _TracerScope:
        """Time a scope of code."""
        return _TracerScope(self, name, kwargs)

    def scoped(self, func):
        """Decorator to time a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.scope(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def context(self, **ctx) -> _ContextScope:
        """Pass parameters around in a context."""
        return _ContextScope(self, ctx)

    def _push_context(self, ctx) -> None:
        self.contexts.append(ctx)

    def _pop_context(self) -> None:
        self.contexts.pop()

    def get(self, q: str) -> Optional[Any]:
        """Query the current context."""
        if not self.contexts:
            return None
        return self.contexts[-1].get(q)

    def log(self, filename) -> None:
        with open(filename, "w", newline="") as file:
            json.dump(self.record, file, indent=2)


tracers = Tracer()
