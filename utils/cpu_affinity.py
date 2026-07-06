"""
GaussianImageDistribution - CPU Affinity Management

Excludes faulty CPU cores (e.g. an unstable core 2 on this server) from the
current process so that training / data loading is never scheduled onto them and
crashes with SIGSEGV (exit code -11).

The affinity set on the main process is inherited by forked DataLoader worker
processes and by intra-op threads, so a single call at startup protects the
whole run. Which cores to exclude is configured by ``config.EXCLUDED_CPUS``.
"""

import os
from typing import Iterable, List, Optional

from utils.logger import get_logger


logger = get_logger("cpu_affinity")


def apply_cpu_affinity(excluded: Optional[Iterable[int]] = None) -> List[int]:
    """
    Restrict this process (and its children / threads) to all online CPUs except
    those in ``excluded`` (default: ``config.EXCLUDED_CPUS``).

    No-op on platforms without ``os.sched_setaffinity`` (e.g. Windows) or when
    there is nothing to exclude. Best-effort: never raises.

    Returns:
        Sorted list of CPUs the process is now allowed to run on.
    """
    if not hasattr(os, "sched_setaffinity"):
        # Windows / unsupported: nothing to do.
        return []

    if excluded is None:
        # Lazy import avoids a circular import at module load time.
        import config
        excluded = getattr(config, "EXCLUDED_CPUS", [])

    excluded_set = {int(c) for c in excluded}
    available = set(os.sched_getaffinity(0))

    if not excluded_set:
        return sorted(available)

    allowed = sorted(available - excluded_set)
    if not allowed:
        logger.warning(
            f"CPU affinity unchanged: excluding {sorted(excluded_set)} would "
            f"leave no usable CPUs."
        )
        return sorted(available)

    try:
        os.sched_setaffinity(0, allowed)
    except OSError as e:
        logger.warning(f"Failed to set CPU affinity (excluded {sorted(excluded_set)}): {e}")
        return sorted(available)

    logger.info(
        f"CPU affinity set to {len(allowed)}/{len(available)} CPUs "
        f"(excluded faulty cores {sorted(excluded_set & available)})."
    )
    return allowed
