from functools import wraps
import json
import time


class _TracerScope:
    def __init__(self, tracer, name, attrs):
        self.tracer = tracer
        self.name = name
        self.attrs = attrs
    def begin(self):
        self.tracer._tick(self.name, "B", self.attrs)
    def end(self):
        self.tracer._tick(self.name, "E", {})
    def __enter__(self):
        self.begin()
    def __exit__(self, type, value, traceback):
        self.end()


class Tracer:
    """Global tracer to record and print timestamp during training process"""

    def __init__(self) -> None:
        self.record = []
        self.cur = None

    def _tick(self, name, phase, attrs):
        cur = time.time_ns()
        if self.cur is None:
            delta = 0
        else:
            delta = cur - self.cur
        self.cur = cur

        attrs["name"] = name
        attrs["ph"] = phase
        attrs["delta"] = delta
        self.record.append(attrs)

    def tick(self, name, **attrs):
        self._tick(name, "i", attrs)

    def scope(self, name, **kwargs):
        return _TracerScope(self, name, kwargs)

    def scoped(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.scope(func.__name__):
                return func(*args, **kwargs)
        return wrapper

    def log(self, filename):
        with open(filename, "w", newline="") as file:
            json.dump(self.record, file, indent=2)


tracers = Tracer()
