import contextlib
import time


class ElapsedTracker(object):
    def __init__(self, elapsed=0.0):
        self._elapsed = elapsed

    def add(self, time):
        self._elapsed += time

    @property
    def elapsed(self):
        return self._elapsed


class Stopwatch(ElapsedTracker):
    def __init__(self, elapsed=0.0):
        super().__init__(elapsed)
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            self.stop()
        self._start_time = time.time()
        return self

    def stop(self):
        if self._start_time is not None:
            self._elapsed += time.time() - self._start_time
        self._start_time = None

    def add(self, time):
        self._elapsed += time

    @property
    def running(self):
        return self._start_time is not None

    @property
    def elapsed(self):
        if self.running:
            return self._elapsed + time.time() - self._start_time
        else:
            return self._elapsed

    @contextlib.contextmanager
    def pause(self):
        self.stop()
        yield
        self.start()
