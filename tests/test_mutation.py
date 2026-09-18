"""Tests for swarm.mutation — autonomous mutation strategies and engine."""

import asyncio
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.coordinator import Coordinator
from swarm.experiment import ExperimentStatus
from swarm.mutation import (
    ArchitectureScaleStrategy,
    HyperparamPerturbStrategy,
    MutationCandidate,
    MutationEngine,
)

SAMPLE_TRAIN_PY = """
import os
import mlx.core as mx

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**16
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

DEPTH = 4
DEVICE_BATCH_SIZE = 16
"""


def test_hyperparam_perturb_strategy():
    strat = HyperparamPerturbStrategy(seed=42)
    candidates = strat.mutate(SAMPLE_TRAIN_PY, history=[], count=3)

    assert len(candidates) == 3
    for cand in candidates:
        assert isinstance(cand, MutationCandidate)
        assert cand.mutation_type == "hyperparam_perturb"
        assert cand.train_py_content != SAMPLE_TRAIN_PY
        assert len(cand.mutations_applied) >= 1
        assert "Mutate" in cand.description


def test_architecture_scale_strategy():
    strat = ArchitectureScaleStrategy(seed=42)
    candidates = strat.mutate(SAMPLE_TRAIN_PY, history=[], count=3)

    assert len(candidates) >= 1
    for cand in candidates:
        assert isinstance(cand, MutationCandidate)
        assert cand.mutation_type == "architecture_scale"
        assert cand.train_py_content != SAMPLE_TRAIN_PY
        assert ("DEPTH" in cand.mutations_applied or "ASPECT_RATIO" in cand.mutations_applied)


def test_mutation_engine_batch_generation():
    engine = MutationEngine(seed=42)
    batch = engine.generate_batch(
        base_content=SAMPLE_TRAIN_PY,
        history=[],
        count=4,
        generation=1,
        parent_experiment_id="exp-root",
    )

    assert len(batch) == 4
    # All candidates must have unique contents
    hashes = {c.content_hash for c in batch}
    assert len(hashes) == 4

    for c in batch:
        assert c.generation == 1
        assert c.parent_experiment_id == "exp-root"
        assert c.train_py_content != SAMPLE_TRAIN_PY


@pytest.mark.asyncio
async def test_coordinator_generate_next_batch(tmp_path):
    train_py = tmp_path / "train.py"
    train_py.write_text(SAMPLE_TRAIN_PY)

    coord = Coordinator(
        repo_dir=str(tmp_path),
        autonomous=True,
        max_generations=2,
        patience=2,
        mutations_per_batch=2,
    )

    from swarm.coordinator import WorkerState
    # Add a mock worker so coord knows workers are alive
    class MockWs:
        async def send(self, data):
            pass

    coord.workers["w1"] = WorkerState("w1", {"chip": "M1", "total_memory_gb": 16.0}, MockWs())

    # Generation 1
    success = await coord._generate_next_batch()
    assert success
    assert coord.current_generation == 1
    assert len(coord.tracker.experiments) == 2
    # One was assigned to idle w1, one remains queued
    assert len(coord.tracker.get_queued_experiments()) == 1

    # Empty/complete to simulate finish
    for exp in coord.tracker.experiments.values():
        exp.status = ExperimentStatus.SUCCESS
        exp.val_bpb = 3.0
    coord.workers["w1"].status = "idle"
    coord.workers["w1"].current_experiment_id = None

    # Generation 2
    success2 = await coord._generate_next_batch()
    assert success2
    assert coord.current_generation == 2
    assert len(coord.tracker.experiments) == 4

    # Generation 3 should exceed max_generations=2
    success3 = await coord._generate_next_batch()
    assert not success3


@pytest.mark.asyncio
async def test_coordinator_patience_exhaustion(tmp_path):
    train_py = tmp_path / "train.py"
    train_py.write_text(SAMPLE_TRAIN_PY)

    coord = Coordinator(
        repo_dir=str(tmp_path),
        autonomous=True,
        max_generations=10,
        patience=1,
        mutations_per_batch=1,
    )

    from swarm.coordinator import WorkerState
    coord.workers["w1"] = WorkerState("w1", {}, None)
    coord.generations_without_improvement = 1

    # Should stop due to patience exhaustion
    success = await coord._generate_next_batch()
    assert not success
