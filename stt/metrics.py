"""Resource tracking — peak RSS, average CPU, wall time.

We sample on a background thread because some engines block in C without
yielding to Python long enough for a polled tracker to catch the peak.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class _Sample:
    rss: int
    cpu_pct: float


class ResourceTracker:
    """Measure wall time, peak RSS, and average CPU between start() and stop().

    Call start() right before the transcribe() call and stop() in a finally.
    Read wall_seconds / peak_rss_mb / avg_cpu after stop().
    """

    SAMPLE_INTERVAL_S = 0.2

    def __init__(self) -> None:
        self._process = psutil.Process(os.getpid())
        # Prime cpu_percent so the first real read isn't 0.0
        self._process.cpu_percent(interval=None)
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._peak_rss: int = 0
        self._cpu_samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        self._start_time = time.time()
        self._peak_rss = self._process.memory_info().rss
        self._cpu_samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._end_time = time.time()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ── sampling ─────────────────────────────────────────────────────
    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = self._process.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
                # cpu_percent without interval returns since last call —
                # so divisions by num cores give a 0..100 % per-core figure.
                cpu = self._process.cpu_percent(interval=None)
                self._cpu_samples.append(cpu)
            except Exception:
                # The process may have closed file descriptors mid-sample
                # under MLX shutdown; ignore transient errors.
                pass
            self._stop_event.wait(self.SAMPLE_INTERVAL_S)

    # ── readouts ─────────────────────────────────────────────────────
    @property
    def wall_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.time()
        return end - self._start_time

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss / (1024 * 1024)

    @property
    def avg_cpu(self) -> float:
        # Drop the priming-zero sample if present
        samples = [s for s in self._cpu_samples if s > 0]
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    # ── live snapshots (safe to call while running) ──────────────────
    @property
    def current_rss_mb(self) -> float:
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def last_cpu(self) -> float:
        return self._cpu_samples[-1] if self._cpu_samples else 0.0
