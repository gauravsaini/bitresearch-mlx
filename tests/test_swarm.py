"""
Integration tests for the distributed swarm.

Tests the full coordinator → worker → experiment → result pipeline
on localhost using a mock train.py that completes in seconds.
Run: uv run python -m pytest tests/ -v
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest
import websockets

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.coordinator import Coordinator
from swarm.experiment import ExperimentTracker
from swarm.hardware import detect_hardware
from swarm.protocol import (
    ExperimentResult,
    ExperimentSpec,
    Message,
    MessageType,
    make_register_message,
    make_result_message,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_TRAIN_PY = textwrap.dedent('''\
    """Mock train.py for testing — prints fake results instantly."""
    import time
    import random

    time.sleep(1)  # Simulate brief training

    val_bpb = 2.5 + random.uniform(-0.1, 0.1)
    peak_vram_mb = 1024.0
    training_seconds = 1.0
    total_seconds = 1.5

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      0.00")
    print(f"total_tokens_M:   1.0")
    print(f"num_steps:        10")
    print(f"num_params_M:     1.0")
    print(f"depth:            2")
''')


@pytest.fixture
def mock_train_py(tmp_path):
    """Write a mock train.py that finishes instantly."""
    train_file = tmp_path / "train.py"
    train_file.write_text(MOCK_TRAIN_PY)
    return train_file


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(repo_dir), capture_output=True)
    (repo_dir / "train.py").write_text(MOCK_TRAIN_PY)
    (repo_dir / "results.tsv").write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
    subprocess.run(["git", "add", "-A"], cwd=str(repo_dir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo_dir), capture_output=True)
    return repo_dir


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestHardwareDetection:
    """Test hardware detection works on this machine."""

    def test_detect_hardware(self):
        hw = detect_hardware()
        assert hw.hostname
        assert hw.chip
        assert hw.total_memory_gb > 0
        assert hw.macos_version
        assert hw.mlx_available is True
        assert hw.tier in ("base", "pro", "max", "ultra")

    def test_hardware_json_roundtrip(self):
        hw = detect_hardware()
        json_str = hw.to_json()
        from swarm.hardware import HardwareInfo
        hw2 = HardwareInfo.from_json(json_str)
        assert hw2.hostname == hw.hostname
        assert hw2.chip == hw.chip
        assert hw2.total_memory_gb == hw.total_memory_gb

    def test_tier_classification(self):
        from swarm.hardware import HardwareInfo
        hw = HardwareInfo(
            hostname="test", chip="M4 Ultra", total_memory_gb=192.0,
            gpu_cores=80, cpu_cores_performance=16, cpu_cores_efficiency=16,
            memory_bandwidth_gbps=819.2, macos_version="15.0",
            python_version="3.11", mlx_available=True,
        )
        assert hw.tier == "ultra"

        hw.total_memory_gb = 64.0
        assert hw.tier == "max"

        hw.total_memory_gb = 32.0
        assert hw.tier == "pro"

        hw.total_memory_gb = 16.0
        assert hw.tier == "base"


class TestProtocol:
    """Test message protocol serialization."""

    def test_message_roundtrip(self):
        msg = Message(type=MessageType.WORKER_REGISTER, payload={"worker_id": "test-1"}, sender_id="test-1")
        json_str = msg.to_json()
        msg2 = Message.from_json(json_str)
        assert msg2.type == MessageType.WORKER_REGISTER
        assert msg2.payload["worker_id"] == "test-1"

    def test_experiment_spec_roundtrip(self):
        spec = ExperimentSpec(
            experiment_id="abc123",
            train_py_content="print('hello')",
            description="test experiment",
            branch_name="test/branch",
            min_memory_gb=16.0,
        )
        d = spec.to_dict()
        spec2 = ExperimentSpec.from_dict(d)
        assert spec2.experiment_id == "abc123"
        assert spec2.train_py_content == "print('hello')"
        assert spec2.min_memory_gb == 16.0

    def test_experiment_result_tsv(self):
        result = ExperimentResult(
            experiment_id="abc123",
            worker_id="mini-a1b2c3",
            val_bpb=2.534,
            peak_vram_mb=27528.9,
            training_seconds=312.4,
            total_seconds=405.7,
            total_tokens_m=39.8,
            num_steps=46,
            num_params_m=50.3,
            depth=8,
            status="success",
            worker_chip="Apple M2",
        )
        row = result.tsv_row("abc1234")
        assert "abc1234" in row
        assert "2.534000" in row
        assert "mini-a1b2c3" in row


class TestExperimentTracker:
    """Test experiment tracking and git integration."""

    def test_create_experiment(self, temp_repo):
        tracker = ExperimentTracker(str(temp_repo))
        exp = tracker.create_experiment(
            description="test depth=8",
            train_py_content=MOCK_TRAIN_PY,
        )
        assert exp.experiment_id
        assert exp.description == "test depth=8"
        assert exp.status.value == "queued"

    def test_complete_and_keep(self, temp_repo):
        tracker = ExperimentTracker(str(temp_repo))
        exp = tracker.create_experiment(
            description="baseline",
            train_py_content=MOCK_TRAIN_PY,
        )
        tracker.complete_experiment(
            experiment_id=exp.experiment_id,
            val_bpb=2.534,
            peak_vram_mb=27000.0,
            training_seconds=300.0,
            total_seconds=400.0,
            total_tokens_m=39.8,
            num_steps=46,
            num_params_m=50.3,
            depth=8,
            status="success",
        )
        assert tracker.best_val_bpb == 2.534
        assert tracker.should_keep(exp.experiment_id)

    def test_discard_worse(self, temp_repo):
        tracker = ExperimentTracker(str(temp_repo))

        # Better experiment
        exp1 = tracker.create_experiment(description="good", train_py_content=MOCK_TRAIN_PY)
        tracker.complete_experiment(
            experiment_id=exp1.experiment_id, val_bpb=2.0,
            peak_vram_mb=1000, training_seconds=300, total_seconds=400,
            total_tokens_m=10, num_steps=10, num_params_m=10, depth=4, status="success",
        )

        # Worse experiment
        exp2 = tracker.create_experiment(description="bad", train_py_content=MOCK_TRAIN_PY)
        tracker.complete_experiment(
            experiment_id=exp2.experiment_id, val_bpb=3.0,
            peak_vram_mb=1000, training_seconds=300, total_seconds=400,
            total_tokens_m=10, num_steps=10, num_params_m=10, depth=4, status="success",
        )

        assert tracker.should_keep(exp1.experiment_id)
        assert not tracker.should_keep(exp2.experiment_id)

    def test_write_results_tsv(self, temp_repo):
        tracker = ExperimentTracker(str(temp_repo))
        exp = tracker.create_experiment(description="test", train_py_content=MOCK_TRAIN_PY)
        tracker.complete_experiment(
            experiment_id=exp.experiment_id, val_bpb=2.5,
            peak_vram_mb=1024, training_seconds=300, total_seconds=400,
            total_tokens_m=10, num_steps=10, num_params_m=10, depth=4, status="success",
        )
        tracker.write_results_tsv()

        tsv = (temp_repo / "results.tsv").read_text()
        assert "commit\tval_bpb" in tsv
        assert "2.500000" in tsv

    def test_git_commit(self, temp_repo):
        tracker = ExperimentTracker(str(temp_repo))
        exp = tracker.create_experiment(description="git test", train_py_content=MOCK_TRAIN_PY + "\n# changed")
        tracker.complete_experiment(
            experiment_id=exp.experiment_id, val_bpb=2.5,
            peak_vram_mb=1024, training_seconds=300, total_seconds=400,
            total_tokens_m=10, num_steps=10, num_params_m=10, depth=4, status="success",
        )
        commit = tracker.git_commit_experiment(exp.experiment_id)
        assert commit is not None
        assert len(commit) == 7


# ---------------------------------------------------------------------------
# Integration Test: Coordinator + Worker on localhost
# ---------------------------------------------------------------------------

class TestDistributedIntegration:
    """
    Full integration test: coordinator + mock worker on localhost.
    Verifies the complete message round-trip without needing multiple machines.
    """

    @pytest.mark.asyncio
    async def test_coordinator_worker_roundtrip(self, temp_repo):
        """Test: coordinator starts → worker connects → experiment assigned → result returned."""
        port = 18765  # Use a non-standard port to avoid conflicts
        coordinator = Coordinator(repo_dir=str(temp_repo), port=port)

        # Submit an experiment to the queue
        exp = coordinator.tracker.create_experiment(
            description="integration test",
            train_py_content=MOCK_TRAIN_PY,
        )

        # Start coordinator in background
        server = await websockets.serve(
            coordinator._handle_connection,
            "127.0.0.1",
            port,
            ping_interval=None,
            max_size=50 * 1024 * 1024,
        )

        try:
            # Simulate a worker connecting
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                # 1. Register
                hw = detect_hardware()
                reg_msg = make_register_message("test-worker", hw.to_dict())
                await ws.send(reg_msg.to_json())

                # 2. Wait for ACK and experiment assignment
                messages_received = []
                for _ in range(5):
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2)
                        msg = Message.from_json(raw)
                        messages_received.append(msg)
                        if msg.type == MessageType.EXPERIMENT_ASSIGN:
                            break
                    except asyncio.TimeoutError:
                        break

                # Verify we got an experiment assignment
                assign_msgs = [m for m in messages_received if m.type == MessageType.EXPERIMENT_ASSIGN]
                assert len(assign_msgs) > 0, f"Expected experiment assignment, got: {[m.type for m in messages_received]}"

                assigned = ExperimentSpec.from_dict(assign_msgs[0].payload)
                assert assigned.experiment_id == exp.experiment_id
                assert assigned.description == "integration test"

                # 3. Send back a result
                result = ExperimentResult(
                    experiment_id=assigned.experiment_id,
                    worker_id="test-worker",
                    val_bpb=2.45,
                    peak_vram_mb=1024.0,
                    training_seconds=1.0,
                    total_seconds=1.5,
                    total_tokens_m=1.0,
                    num_steps=10,
                    num_params_m=1.0,
                    depth=2,
                    status="success",
                    description="integration test",
                    worker_chip=hw.chip,
                    worker_memory_gb=hw.total_memory_gb,
                    worker_tier=hw.tier,
                )
                result_msg = make_result_message(result)
                await ws.send(result_msg.to_json())

                # Give coordinator time to process
                await asyncio.sleep(0.5)

            # 4. Verify result was recorded
            assert coordinator.tracker.best_val_bpb == 2.45
            assert coordinator.tracker.best_experiment_id == exp.experiment_id

            # 5. Verify results.tsv was written
            tsv = (temp_repo / "results.tsv").read_text()
            assert "2.450000" in tsv

        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_multiple_workers(self, temp_repo):
        """Test: two workers connect and each gets a different experiment."""
        port = 18766
        coordinator = Coordinator(repo_dir=str(temp_repo), port=port)

        # Queue two experiments
        exp1 = coordinator.tracker.create_experiment(
            description="experiment A",
            train_py_content=MOCK_TRAIN_PY,
        )
        exp2 = coordinator.tracker.create_experiment(
            description="experiment B",
            train_py_content=MOCK_TRAIN_PY,
        )

        server = await websockets.serve(
            coordinator._handle_connection,
            "127.0.0.1",
            port,
            ping_interval=None,
            max_size=50 * 1024 * 1024,
        )

        try:
            hw = detect_hardware()

            # Worker 1 connects
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws1:
                reg1 = make_register_message("worker-1", hw.to_dict())
                await ws1.send(reg1.to_json())

                # Collect messages for worker 1
                w1_assigned = None
                for _ in range(5):
                    try:
                        raw = await asyncio.wait_for(ws1.recv(), timeout=2)
                        msg = Message.from_json(raw)
                        if msg.type == MessageType.EXPERIMENT_ASSIGN:
                            w1_assigned = ExperimentSpec.from_dict(msg.payload)
                            break
                    except asyncio.TimeoutError:
                        break

                assert w1_assigned is not None, "Worker 1 should get an experiment"

                # Worker 2 connects
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws2:
                    reg2 = make_register_message("worker-2", hw.to_dict())
                    await ws2.send(reg2.to_json())

                    w2_assigned = None
                    for _ in range(5):
                        try:
                            raw = await asyncio.wait_for(ws2.recv(), timeout=2)
                            msg = Message.from_json(raw)
                            if msg.type == MessageType.EXPERIMENT_ASSIGN:
                                w2_assigned = ExperimentSpec.from_dict(msg.payload)
                                break
                        except asyncio.TimeoutError:
                            break

                    assert w2_assigned is not None, "Worker 2 should get an experiment"

                    # They should get different experiments
                    assert w1_assigned.experiment_id != w2_assigned.experiment_id
                    assigned_ids = {w1_assigned.experiment_id, w2_assigned.experiment_id}
                    expected_ids = {exp1.experiment_id, exp2.experiment_id}
                    assert assigned_ids == expected_ids

        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_worker_reconnect_requeues(self, temp_repo):
        """Test: when a worker disconnects, its assigned experiment is re-queued."""
        port = 18767
        coordinator = Coordinator(repo_dir=str(temp_repo), port=port)

        exp = coordinator.tracker.create_experiment(
            description="reconnect test",
            train_py_content=MOCK_TRAIN_PY,
        )

        server = await websockets.serve(
            coordinator._handle_connection,
            "127.0.0.1",
            port,
            ping_interval=None,
            max_size=50 * 1024 * 1024,
        )

        try:
            hw = detect_hardware()

            # Worker connects and gets experiment
            ws1 = await websockets.connect(f"ws://127.0.0.1:{port}")
            reg = make_register_message("worker-flaky", hw.to_dict())
            await ws1.send(reg.to_json())

            assigned = None
            for _ in range(5):
                try:
                    raw = await asyncio.wait_for(ws1.recv(), timeout=2)
                    msg = Message.from_json(raw)
                    if msg.type == MessageType.EXPERIMENT_ASSIGN:
                        assigned = ExperimentSpec.from_dict(msg.payload)
                        break
                except asyncio.TimeoutError:
                    break

            assert assigned is not None

            # Worker disconnects without sending result
            await ws1.close()
            await asyncio.sleep(0.5)

            # Experiment should be re-queued
            exp_state = coordinator.tracker.experiments[exp.experiment_id]
            assert exp_state.status.value == "queued"

        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_submit_experiment_websocket(self, temp_repo):
        """Test: client submits an experiment over websocket and receives coordinator ack."""
        port = 18768
        coordinator = Coordinator(repo_dir=str(temp_repo), port=port)

        server = await websockets.serve(
            coordinator._handle_connection,
            "127.0.0.1",
            port,
            ping_interval=None,
            max_size=50 * 1024 * 1024,
        )

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                submit_msg = Message(
                    type=MessageType.WORKER_STATUS,
                    payload={
                        "action": "submit_experiment",
                        "description": "cli submit test",
                        "train_py_content": MOCK_TRAIN_PY,
                        "min_memory_gb": 0.0,
                        "preferred_tier": "",
                    },
                    sender_id="cli",
                )
                await ws.send(submit_msg.to_json())
                raw_resp = await asyncio.wait_for(ws.recv(), timeout=5)
                resp = Message.from_json(raw_resp)
                assert resp.type == MessageType.COORDINATOR_ACK
                assert resp.payload.get("status") == "submitted"
                exp_id = resp.payload.get("experiment_id")
                assert exp_id in coordinator.tracker.experiments
                assert coordinator.tracker.experiments[exp_id].description == "cli submit test"
        finally:
            server.close()
            await server.wait_closed()


# ---------------------------------------------------------------------------
# End-to-End Test: Real subprocess worker
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    End-to-end test using actual subprocess execution.
    This tests the worker's ability to run a mock train.py and parse output.
    """

    def test_mock_train_runs(self, mock_train_py, tmp_path):
        """Verify mock train.py produces parseable output."""
        result = subprocess.run(
            [sys.executable, str(mock_train_py)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "val_bpb:" in result.stdout
        assert "peak_vram_mb:" in result.stdout
        assert "num_steps:" in result.stdout

    def test_worker_result_parsing(self, mock_train_py, tmp_path):
        """Test the worker's log parsing logic."""
        from swarm.worker import Worker

        # Run mock training and capture output
        log_file = tmp_path / "run.log"
        result = subprocess.run(
            [sys.executable, str(mock_train_py)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        log_file.write_text(result.stdout)

        # Parse using worker logic
        worker = Worker(worker_id="test", work_dir=str(tmp_path))
        spec = ExperimentSpec(
            experiment_id="test123",
            train_py_content=MOCK_TRAIN_PY,
            description="parser test",
            branch_name="test",
        )
        parsed = worker._parse_results(log_file, spec, 2.0)

        assert parsed.status == "success"
        assert 2.0 < parsed.val_bpb < 3.0
        assert parsed.peak_vram_mb == 1024.0
        assert parsed.num_steps == 10
        assert parsed.depth == 2

    @pytest.mark.asyncio
    async def test_full_pipeline_roundtrip(self, temp_repo, tmp_path):
        """Test full live loop: coordinator + worker subprocess execution + CLI submit."""
        from swarm.worker import Worker

        port = 18770
        coord = Coordinator(repo_dir=str(temp_repo), port=port)
        server = await websockets.serve(coord._handle_connection, "127.0.0.1", port)

        work_dir = tmp_path / "worker_scratch"
        worker = Worker(worker_id="test-e2e-worker", coordinator_url=f"ws://127.0.0.1:{port}", work_dir=str(work_dir))
        worker_task = asyncio.create_task(worker._connect_and_serve())

        try:
            # Submit via WebSocket client (as CLI submit does)
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                submit_msg = Message(
                    type=MessageType.WORKER_STATUS,
                    payload={
                        "action": "submit_experiment",
                        "description": "end-to-end pipeline verification",
                        "train_py_content": MOCK_TRAIN_PY,
                        "min_memory_gb": 0.0,
                        "preferred_tier": "",
                    },
                    sender_id="cli",
                )
                await ws.send(submit_msg.to_json())
                resp_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                resp = Message.from_json(resp_raw)
                exp_id = resp.payload["experiment_id"]

            # Wait for completion
            for _ in range(30):
                await asyncio.sleep(0.5)
                exp = coord.tracker.experiments.get(exp_id)
                if exp and exp.status.value in ("success", "crash", "timeout"):
                    break

            assert exp is not None
            assert exp.status.value == "success"
            assert 2.0 < exp.val_bpb < 3.0
            tsv = (temp_repo / "results.tsv").read_text()
            assert exp.description in tsv
            assert "keep" in tsv

        finally:
            worker_task.cancel()
            server.close()
            await server.wait_closed()
