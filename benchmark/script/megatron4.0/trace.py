from functools import wraps
import json
import time
import torch
from typing import Any, Dict, List, Optional


class _TracerScope:
    def __init__(self, tracer: "Tracer", name: Optional[str], in_attrs: Dict[str, Any], out_attrs: Dict[str, Any]) -> None:
        self.tracer = tracer
        self.name = name
        self.in_attrs = in_attrs
        self.out_attrs = out_attrs

    def __enter__(self) -> None:
        self.tracer._push_scope(self)
        if self.name is not None:
            self.tracer._tick(self.name, "B", {})

    def __exit__(self, type, value, traceback) -> None:
        if self.name is not None:
            self.tracer._tick(self.name, "E", self.out_attrs)
        self.tracer._pop_scope()

    def get(self, q: str) -> Optional[Any]:
        """Get from in_attrs."""
        return self.in_attrs.get(q)

    def set(self, q: str, v: Any) -> bool:
        """Set to out_attrs, if this is required."""
        if "q" in self.out_attrs and self.out_attrs[q] is None:
            self.out_attrs[q] = v
            return True
        else:
            return False


class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self.record: List[Any] = []
        self.cur: int = None
        self.pending_initial_delta: int = None
        self.pending: List[Any] = None
        self.scopes: List[_TracerScope] = []

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

    def _process_event(attrs: Dict[str, Any], delta: int) -> None:
        attrs["delta"] = delta
        del attrs["cuda_event"]
        if "data" in attrs:
            # Since we put this in "E" phase, delta is the duration
            attrs["bandwidth"] = attrs["data"] / (delta / 1e9) # bps

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
            # nanoseconds
            next_delta = int(begin["cuda_event"].elapsed_time(end["cuda_event"]) * 1e6)
            Tracer._process_event(begin, delta)
            self._add_record(begin)
            delta = next_delta
            cuda_duration += delta
        end = self.pending[-1]
        Tracer._process_event(end, delta)
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

    def scope(self, name: Optional[str], *args, ctx: Dict[str, Any] = {}, slots: List[str] = [], **kwargs: Any) -> _TracerScope:
        """Create a scope of code.
        Args:
            name: Name of the scope. If None, the scope is not timed.
            ctx: Parameters to be passed to the scope.
            kwargs: Items to be recorded. If an item is None, it should be filled by some inner scope.
            slots: Parameters that are passed to the scope and must be filled. (They go to both ctx and kwargs.)
        """
        for slot in slots:
            ctx[slot] = True
            kwargs[slot] = None
        return _TracerScope(self, name=name, in_attrs=ctx, out_attrs=kwargs)

    def scoped(self, func):
        """Decorator to time a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.scope(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def _push_scope(self, scope) -> None:
        self.scopes.append(scope)

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def get(self, q: str) -> Optional[Any]:
        """Query parameter from scopes."""
        for scope in reversed(self.scopes):
            v = scope.get(q)
            if v is not None:
                return v
        return None

    def set(self, q: str, v: Any) -> None:
        """Set parameter to the nearest requiring scope."""
        for scope in reversed(self.scopes):
            if scope.set(q, v):
                return
        assert False, f"Cannot find a requiring scope for {q}"

    def log(self, filename) -> None:
        with open(filename, "w", newline="") as file:
            json.dump(self.record, file, indent=2)


tracers = Tracer()

def get_tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()
