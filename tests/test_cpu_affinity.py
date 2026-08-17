"""
Tests for utils.cpu_affinity.apply_cpu_affinity — diagnostic logging.

Background: CPU 2 on this server is unstable and crashes runs with SIGSEGV
(exit code -11). On this machine it is already offline at the kernel level
(absent from /proc/cpuinfo). Previously the startup log printed

    "excluded faulty cores []"

in that case — an empty list that reads like "nothing was excluded", even though
the process is in fact already protected. The log must instead state clearly that
the requested core is already offline / unavailable.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import cpu_affinity
from utils.cpu_affinity import apply_cpu_affinity


class _ListHandler(logging.Handler):
    """Collects formatted log records emitted while attached."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


class _FakeCpuEnv:
    """Patches os.sched_{get,set}affinity and the sysfs online check."""

    def __init__(self, available, online_map):
        self.available = set(available)
        self.online_map = online_map
        self._real_ga = getattr(os, "sched_getaffinity", None)
        self._real_sa = getattr(os, "sched_setaffinity", None)
        self._real_status = cpu_affinity._cpu_online_status

    def __enter__(self):
        os.sched_getaffinity = lambda pid: set(self.available)
        os.sched_setaffinity = lambda pid, mask: None
        cpu_affinity._cpu_online_status = lambda c: self.online_map.get(c)
        return self

    def __exit__(self, *exc):
        if self._real_ga is not None:
            os.sched_getaffinity = self._real_ga
        elif hasattr(os, "sched_getaffinity"):
            del os.sched_getaffinity
        if self._real_sa is not None:
            os.sched_setaffinity = self._real_sa
        elif hasattr(os, "sched_setaffinity"):
            del os.sched_setaffinity
        cpu_affinity._cpu_online_status = self._real_status
        return False


def _run(excluded, available, online_map):
    """Run apply_cpu_affinity under a faked CPU environment; return the log line."""
    handler = _ListHandler()
    logger = cpu_affinity.logger
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with _FakeCpuEnv(available, online_map):
            apply_cpu_affinity(excluded)
    finally:
        logger.setLevel(prev_level)
        logger.removeHandler(handler)
    assert handler.messages, "expected an INFO log line, got none"
    return handler.messages[-1]


def test_offline_core_reports_already_offline_not_empty_list():
    """CPU 2 already offline at system level -> say so, not 'excluded faulty cores []'."""
    msg = _run([2], available={0, 1, 3, 4, 5}, online_map={2: False})
    assert "already offline at system level" in msg, msg
    assert "[]" not in msg, msg               # no misleading empty list
    assert "[2]" in msg, msg


def test_online_core_actually_excluded():
    """When the faulty core is online, it is removed from the affinity mask."""
    msg = _run([2], available={0, 1, 2, 3, 4, 5}, online_map={2: True})
    assert "excluded faulty cores [2]" in msg, msg
    assert "5/6" in msg, msg                  # 6 online, 1 removed


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
