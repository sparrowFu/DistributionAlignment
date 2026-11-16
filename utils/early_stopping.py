"""Early stopping that monitors comparable scores within a training stage.

This controller is independent of global best-checkpoint selection. Epochs use
the trainer's zero-based numbering, including ``last_epoch`` and the optional
``expected_epoch`` passed when restoring a checkpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class EarlyStoppingEvent:
    status: str
    should_stop: bool
    reset: bool


class StageAwareEarlyStopping:
    """Count non-improving epochs only after the complete schedule is eligible.

    ``min_epoch`` controls eligibility; a new stage establishes its own
    baseline instead of comparing against a score from an earlier objective.
    The caller must supply a consistently defined metric even within a stage,
    e.g. fixed-weight validation loss rather than ramped training loss.
    """

    _VERSION = 1

    def __init__(self, metric: str, mode: str, patience: int, enabled: bool,
                 min_epoch: int = 0):
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric must be a non-empty string")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if type(min_epoch) is not int or min_epoch < 0:
            raise ValueError("min_epoch must be a non-negative integer")
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.enabled = enabled
        self.min_epoch = min_epoch
        self.reset()

    def reset(self) -> None:
        """Discard monitoring history, retaining the configured policy."""
        self.best: Optional[float] = None
        self.bad_epochs = 0
        self.stage: Optional[str] = None
        self.last_epoch: Optional[int] = None

    def update(self, value: float, epoch: int,
               stage: str) -> EarlyStoppingEvent:
        """Observe an epoch once; equal scores count as non-improvement.

        Disabled/waiting observations advance the epoch clock but never accrue
        patience. Non-finite values cannot become a baseline or an optimum;
        once eligible, they count as bad epochs just like a failed validation.
        """
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if self.last_epoch is not None and epoch <= self.last_epoch:
            raise ValueError("epochs must increase; an epoch cannot be counted twice")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        score = float(value)
        changed = stage != self.stage
        self.stage, self.last_epoch = stage, epoch

        if not self.enabled:
            self.best, self.bad_epochs = None, 0
            return EarlyStoppingEvent("disabled", False, False)
        if epoch < self.min_epoch:
            self.best, self.bad_epochs = None, 0
            return EarlyStoppingEvent("waiting", False, False)

        if changed:
            self.best, self.bad_epochs = None, 0
        if math.isfinite(score):
            if self.best is None:
                self.best, self.bad_epochs = score, 0
                return EarlyStoppingEvent("baseline", False, changed)
            improved = score < self.best if self.mode == "min" else score > self.best
            if improved:
                self.best, self.bad_epochs = score, 0
                return EarlyStoppingEvent("improved", False, changed)

        self.bad_epochs += 1
        stop = self.bad_epochs >= self.patience
        return EarlyStoppingEvent("stopped" if stop else "bad_epoch", stop, changed)

    def state_dict(self) -> dict:
        """Return the monitoring policy and enough history for exact resume."""
        return {
            "version": self._VERSION,
            "metric": self.metric, "mode": self.mode,
            "patience": self.patience, "enabled": self.enabled,
            "min_epoch": self.min_epoch,
            "best": self.best, "bad_epochs": self.bad_epochs,
            "stage": self.stage, "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state, expected_epoch: Optional[int] = None) -> bool:
        """Restore a compatible state; reset and return False otherwise.

        Legacy counters cannot identify their objective/stage and are rejected.
        Configuration changes are also rejected, including toggling early
        stopping or selecting a different metric. ``expected_epoch`` must be
        the exact zero-based epoch represented by the resumed model weights.
        """
        self.reset()
        if not isinstance(state, Mapping):
            return False
        required = {"version", "metric", "mode", "patience", "enabled",
                    "min_epoch", "best", "bad_epochs", "stage", "last_epoch"}
        if not required <= state.keys():
            return False
        if type(state["version"]) is not int or state["version"] != self._VERSION:
            return False
        for key in ("metric", "mode", "patience", "enabled", "min_epoch"):
            value = state[key]
            if type(value) is not type(getattr(self, key)) or value != getattr(self, key):
                return False

        best, bad, stage, epoch = (state[k] for k in
                                   ("best", "bad_epochs", "stage", "last_epoch"))
        if type(bad) is not int or bad < 0:
            return False
        if best is not None:
            if type(best) not in (int, float):
                return False
            try:
                if not math.isfinite(best):
                    return False
            except OverflowError:
                return False
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            return False
        if stage is not None and (not isinstance(stage, str) or not stage.strip()):
            return False
        if (epoch is None) != (stage is None):
            return False
        if expected_epoch is not None:
            if type(expected_epoch) is not int or epoch != expected_epoch:
                return False
        inactive = epoch is None or not self.enabled or epoch < self.min_epoch
        if inactive and (best is not None or bad != 0):
            return False
        if not inactive and best is None and bad == 0:
            return False
        if not inactive:
            # A checkpoint cannot have accumulated more bad observations than
            # eligible epochs, or count its finite baseline as a bad epoch.
            max_bad = epoch - self.min_epoch + (1 if best is None else 0)
            if bad > max_bad:
                return False

        self.best = float(best) if best is not None else None
        self.bad_epochs, self.stage, self.last_epoch = bad, stage, epoch
        return True
