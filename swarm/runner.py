"""
Structured execution adapter for train.py runs.

Replaces the worker's inline subprocess + regex scraping with a typed
RunConfig → RunResult interface. Handles prepare.py resolution via
PYTHONPATH instead of symlinks, and structured output parsing.
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .telemetry import TelemetryStream

logger = logging.getLogger(__name__)

# Timeout for a single experiment (training + compile + eval overhead)
DEFAULT_TIMEOUT = 900  # 15 minutes max


@dataclass
class RunConfig:
    """Configuration for a single train.py execution."""

    train_py_path: Path
    work_dir: Path
    timeout: int = DEFAULT_TIMEOUT
    env_overrides: dict[str, str] = field(default_factory=dict)
    telemetry_path: Path | None = None


@dataclass
class RunResult:
    """Structured result from a train.py execution."""

    return_code: int
    metrics: dict[str, float]  # {"val_bpb": 2.534, "peak_vram_mb": 27528.9, ...}
    log_content: str
    telemetry: list[dict] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


# The metric keys we extract from the final --- block, mapped to their
# output names and types.
_METRIC_KEYS = {
    "val_bpb": float,
    "peak_vram_mb": float,
    "training_seconds": float,
    "total_seconds": float,
    "total_tokens_M": float,
    "num_steps": int,
    "num_params_M": float,
    "depth": int,
    "mfu_percent": float,
}


def parse_final_metrics(log_content: str) -> dict[str, float]:
    """Parse the final metric block printed by train.py at the end of a run.

    Example log block:
        ---
        val_bpb:          2.534012
        training_seconds: 142.3
        ...
    """
    metrics: dict[str, float] = {}
    for key in _METRIC_KEYS:
        match = re.search(rf"^{key}:\s+([0-9.]+)", log_content, re.MULTILINE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass
    return metrics


def _repo_root() -> Path:
    """Find the project root (parent of swarm/)."""
    return Path(__file__).parent.parent


class TrainRunner:
    """Runs train.py in a subprocess and returns structured results."""

    def __init__(self, repo_root: Path | None = None):
        self._repo_root = repo_root or _repo_root()
        self._process: subprocess.Popen | None = None
        self._cancelled: bool = False

    def run(
        self,
        config: RunConfig,
        on_telemetry: Callable[[list[dict]], None] | None = None,
    ) -> RunResult:
        """Execute train.py and return a structured RunResult.

        Called from async code via run_in_executor.
        """
        config.work_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled = False

        # Build environment: inherit + PYTHONPATH for prepare.py + overrides
        env = {**os.environ}
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        # Point PYTHONPATH at repo root so `from prepare import ...` works
        # without symlinking prepare.py into every experiment directory
        existing_pp = env.get("PYTHONPATH", "")
        repo_str = str(self._repo_root)
        env["PYTHONPATH"] = f"{repo_str}:{existing_pp}" if existing_pp else repo_str

        # Telemetry file path
        if config.telemetry_path:
            env["BITRESEARCH_TELEMETRY_FILE"] = str(config.telemetry_path)

        env.update(config.env_overrides)

        log_file = config.work_dir / "run.log"
        t0 = time.time()
        telemetry_stream = (
            TelemetryStream(config.telemetry_path) if config.telemetry_path else None
        )
        all_telemetry: list[dict] = []

        try:
            log_fh = open(log_file, "w")
            self._process = subprocess.Popen(
                ["uv", "run", str(config.train_py_path)],
                cwd=str(config.work_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
            )

            deadline = time.time() + config.timeout
            return_code = None

            while True:
                if telemetry_stream:
                    new_metrics = telemetry_stream.read_new()
                    if new_metrics:
                        dicts = [m.to_dict() for m in new_metrics]
                        all_telemetry.extend(dicts)
                        if on_telemetry:
                            try:
                                on_telemetry(dicts)
                            except Exception as e:
                                logger.debug(f"Telemetry callback error: {e}")

                ret = self._process.poll()
                if ret is not None:
                    return_code = ret
                    break

                if time.time() > deadline:
                    self._process.kill()
                    self._process.wait()
                    log_fh.close()
                    return RunResult(
                        return_code=-1,
                        metrics={},
                        log_content=_tail(log_file, 50),
                        telemetry=all_telemetry,
                        error=f"Exceeded {config.timeout}s timeout",
                        elapsed_seconds=time.time() - t0,
                    )

                time.sleep(0.1)

            log_fh.close()
            elapsed = time.time() - t0

            # Drain any remaining telemetry written before exit
            if telemetry_stream:
                new_metrics = telemetry_stream.read_new()
                if new_metrics:
                    dicts = [m.to_dict() for m in new_metrics]
                    all_telemetry.extend(dicts)
                    if on_telemetry:
                        try:
                            on_telemetry(dicts)
                        except Exception:
                            pass

            log_content = _read_file(log_file)

            if self._cancelled:
                return RunResult(
                    return_code=-1,
                    metrics={},
                    log_content=log_content,
                    telemetry=all_telemetry,
                    error="Cancelled by coordinator",
                    elapsed_seconds=elapsed,
                )

            if return_code != 0:
                return RunResult(
                    return_code=return_code,
                    metrics={},
                    log_content=log_content,
                    telemetry=all_telemetry,
                    error=_tail(log_file, 50),
                    elapsed_seconds=elapsed,
                )

            metrics = parse_final_metrics(log_content)

            return RunResult(
                return_code=0,
                metrics=metrics,
                log_content=log_content,
                telemetry=all_telemetry,
                elapsed_seconds=elapsed,
            )

        except Exception as e:
            return RunResult(
                return_code=-1,
                metrics={},
                log_content="",
                telemetry=all_telemetry,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )
        finally:
            self._process = None

    def cancel(self):
        """Kill the currently running process."""
        self._cancelled = True
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except Exception:
                pass


def _read_file(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _tail(path: Path, n: int = 50) -> str:
    try:
        lines = path.read_text().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _read_jsonl(path: Path) -> list[dict]:
    results = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    except Exception:
        pass
    return results
