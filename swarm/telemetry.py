"""
Step-level telemetry and speculative pruning for training experiments.

Provides structured telemetry capture during training runs and pruning
policies to terminate unpromising runs early.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class StepMetrics:
    """Telemetry metrics captured at a single training step."""

    step: int
    loss: float
    lr_multiplier: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_seconds: float = 0.0
    peak_vram_mb: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "StepMetrics":
        return cls(
            step=int(d.get("step", 0)),
            loss=float(d.get("loss", 0.0)),
            lr_multiplier=float(d.get("lr_multiplier", 0.0)),
            tokens_per_sec=float(d.get("tokens_per_sec", 0.0)),
            elapsed_seconds=float(d.get("elapsed_seconds", 0.0)),
            peak_vram_mb=float(d.get("peak_vram_mb", 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "loss": self.loss,
            "lr_multiplier": self.lr_multiplier,
            "tokens_per_sec": self.tokens_per_sec,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_vram_mb": self.peak_vram_mb,
        }


class TelemetryStream:
    """Reads a JSONL telemetry file incrementally as lines are appended."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._offset = 0

    def read_new(self) -> list[StepMetrics]:
        """Read all new metrics appended since last call."""
        if not self.file_path.exists():
            return []

        metrics: list[StepMetrics] = []
        try:
            with open(self.file_path, "r") as f:
                f.seek(self._offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        metrics.append(StepMetrics.from_dict(data))
                    except (json.JSONDecodeError, ValueError):
                        continue
                self._offset = f.tell()
        except OSError:
            pass
        return metrics


class PruningPolicy(Protocol):
    """Interface for experiment pruning policies."""

    def should_cancel(
        self, metrics: list[StepMetrics], best_known_bpb: float
    ) -> bool: ...


class ThresholdPruner:
    """
    Pruning policy that cancels runs diverging or performing significantly
    worse than best known performance.
    """

    def __init__(self, threshold_multiplier: float = 2.0, min_steps: int = 30):
        self.threshold_multiplier = threshold_multiplier
        self.min_steps = min_steps

    def should_cancel(
        self, metrics: list[StepMetrics], best_known_bpb: float
    ) -> bool:
        if not metrics:
            return False

        latest = metrics[-1]

        # Immediate pruning on NaN or Inf loss
        if math.isnan(latest.loss) or math.isinf(latest.loss):
            return True

        # Need minimum steps to evaluate trend
        if len(metrics) < self.min_steps:
            return False

        if best_known_bpb <= 0:
            return False

        # If latest loss exceeds threshold multiple of best known score
        if latest.loss > (self.threshold_multiplier * best_known_bpb):
            return True

        return False


class NeverPrune:
    """No-op pruning policy for baseline or unconstrained runs."""

    def should_cancel(
        self, metrics: list[StepMetrics], best_known_bpb: float
    ) -> bool:
        return False
