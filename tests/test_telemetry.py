"""Tests for swarm.telemetry — step-level telemetry streaming and pruning."""

import asyncio
import json
import math
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.coordinator import Coordinator
from swarm.protocol import Message, MessageType, make_telemetry_message
from swarm.telemetry import (
    NeverPrune,
    PruningPolicy,
    StepMetrics,
    TelemetryStream,
    ThresholdPruner,
)


def test_step_metrics_serialization():
    m = StepMetrics(
        step=10,
        loss=2.45,
        lr_multiplier=1.0,
        tokens_per_sec=15000.0,
        elapsed_seconds=12.5,
        peak_vram_mb=4096.0,
    )
    d = m.to_dict()
    assert d["step"] == 10
    assert d["loss"] == 2.45
    m2 = StepMetrics.from_dict(d)
    assert m2.step == m.step
    assert m2.loss == m.loss
    assert m2.peak_vram_mb == m.peak_vram_mb


def test_telemetry_stream_incremental(tmp_path):
    tfile = tmp_path / "telemetry.jsonl"
    stream = TelemetryStream(tfile)

    # Initially empty
    assert stream.read_new() == []

    # Write two steps
    with open(tfile, "a") as f:
        f.write(json.dumps({"step": 1, "loss": 3.5}) + "\n")
        f.write(json.dumps({"step": 2, "loss": 3.2}) + "\n")

    first_read = stream.read_new()
    assert len(first_read) == 2
    assert first_read[0].step == 1
    assert first_read[1].step == 2

    # Nothing new without writes
    assert stream.read_new() == []

    # Append a third step
    with open(tfile, "a") as f:
        f.write(json.dumps({"step": 3, "loss": 3.0}) + "\n")

    second_read = stream.read_new()
    assert len(second_read) == 1
    assert second_read[0].step == 3


def test_threshold_pruner_min_steps():
    pruner = ThresholdPruner(threshold_multiplier=2.0, min_steps=10)
    best_bpb = 2.0

    # 5 steps with high loss — not pruned because len < min_steps
    metrics = [StepMetrics(step=i, loss=10.0) for i in range(5)]
    assert not pruner.should_cancel(metrics, best_bpb)


def test_threshold_pruner_nan_inf():
    pruner = ThresholdPruner(threshold_multiplier=2.0, min_steps=30)
    best_bpb = 2.0

    metrics_nan = [StepMetrics(step=1, loss=float("nan"))]
    assert pruner.should_cancel(metrics_nan, best_bpb)

    metrics_inf = [StepMetrics(step=1, loss=float("inf"))]
    assert pruner.should_cancel(metrics_inf, best_bpb)


def test_threshold_pruner_exceeds_threshold():
    pruner = ThresholdPruner(threshold_multiplier=2.0, min_steps=5)
    best_bpb = 2.0  # threshold is 4.0

    # Normal run
    good_metrics = [StepMetrics(step=i, loss=2.5) for i in range(10)]
    assert not pruner.should_cancel(good_metrics, best_bpb)

    # Diverging run
    bad_metrics = [StepMetrics(step=i, loss=4.5) for i in range(10)]
    assert pruner.should_cancel(bad_metrics, best_bpb)


def test_never_prune():
    pruner = NeverPrune()
    metrics = [StepMetrics(step=i, loss=100.0) for i in range(50)]
    assert not pruner.should_cancel(metrics, 1.0)


@pytest.mark.asyncio
async def test_coordinator_telemetry_and_pruning(tmp_path):
    coord = Coordinator(
        repo_dir=str(tmp_path),
        pruning_policy=ThresholdPruner(threshold_multiplier=2.0, min_steps=3),
    )
    coord.tracker.best_val_bpb = 2.0

    # Mock a worker connection
    class MockWs:
        def __init__(self):
            self.sent_messages = []

        async def send(self, data):
            self.sent_messages.append(json.loads(data))

    ws = MockWs()
    from swarm.coordinator import WorkerState
    worker = WorkerState("test-worker", {"chip": "M1", "total_memory_gb": 16.0}, ws)
    worker.current_experiment_id = "exp-prune-test"
    coord.workers["test-worker"] = worker

    # Send 2 steps — below min_steps
    step_data = [
        {"step": 0, "loss": 5.0},
        {"step": 1, "loss": 5.0},
    ]
    await coord._handle_telemetry("exp-prune-test", step_data)
    assert len(coord.telemetry["exp-prune-test"]) == 2
    assert len(ws.sent_messages) == 0

    # Send step 3 with high loss — triggers pruning!
    await coord._handle_telemetry("exp-prune-test", [{"step": 2, "loss": 5.0}])
    assert len(coord.telemetry["exp-prune-test"]) == 3
    assert len(ws.sent_messages) == 1
    assert ws.sent_messages[0]["type"] == MessageType.EXPERIMENT_CANCEL
    assert ws.sent_messages[0]["payload"]["experiment_id"] == "exp-prune-test"
