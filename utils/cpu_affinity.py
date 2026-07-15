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


def _cpu_online_status(cpu: int) -> Optional[bool]:
    """System-level online status of a core, read from sysfs.

    Returns ``True`` if the core is online, ``False`` if it exists but is
    offline (e.g. disabled at the kernel level, like CPU 2 on this server), or
    ``None`` when the status cannot be determined (cpu0 has no ``online`` file,
    or sysfs is absent on this platform). Used only to make the startup log
    precise about *why* a requested core was not removed from the affinity mask.
    """
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/online") as f:
            return f.read().strip() == "1"
    except OSError:
        return None


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

    # Distinguish cores we actually removed from the affinity mask this call
    # (``removed``) from requested cores that were already not schedulable here.
    # On this server CPU 2 is offline at the kernel level, so it never appears in
    # ``available``; reporting only ``removed`` would print an empty list and read
    # like "nothing was excluded", even though the process is already protected.
    removed = excluded_set & available
    already_unavailable = excluded_set - available
    already_offline = sorted(
        c for c in already_unavailable if _cpu_online_status(c) is False
    )
    already_excluded = sorted(already_unavailable - set(already_offline))

    status_parts = []
    if removed:
        status_parts.append(f"excluded faulty cores {sorted(removed)}")
    if already_offline:
        status_parts.append(f"{sorted(already_offline)} already offline at system level")
    if already_excluded:
        status_parts.append(f"{sorted(already_excluded)} already unavailable to this process")
    status = "; ".join(status_parts)

    logger.info(f"CPU affinity set to {len(allowed)}/{len(available)} CPUs ({status}).")
    return allowed
