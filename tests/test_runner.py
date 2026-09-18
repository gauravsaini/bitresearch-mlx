"""Tests for swarm.runner — structured execution adapter."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.runner import RunConfig, RunResult, TrainRunner, parse_final_metrics


# ---------------------------------------------------------------------------
# parse_final_metrics — pure function tests
# ---------------------------------------------------------------------------

MOCK_OUTPUT = textwrap.dedent("""\
    Data/tokenizer loaded in 1.2s
    Model compiled in 3.4s
    Training completed in 300.0s
    Starting final eval...
    Final eval completed in 10.0s
    ---
    val_bpb:          2.534000
    training_seconds: 312.4
    total_seconds:    405.7
    peak_vram_mb:     27528.9
    mfu_percent:      0.00
    total_tokens_M:   39.8
    num_steps:        46
    num_params_M:     50.3
    depth:            8
""")


class TestParseFinalMetrics:
    def test_parses_all_metrics(self):
        m = parse_final_metrics(MOCK_OUTPUT)
        assert m["val_bpb"] == 2.534
        assert m["peak_vram_mb"] == 27528.9
        assert m["training_seconds"] == 312.4
        assert m["total_seconds"] == 405.7
        assert m["total_tokens_M"] == 39.8
        assert m["num_steps"] == 46
        assert m["num_params_M"] == 50.3
        assert m["depth"] == 8
        assert m["mfu_percent"] == 0.0

    def test_empty_input(self):
        assert parse_final_metrics("") == {}

    def test_partial_output(self):
        m = parse_final_metrics("val_bpb:          2.1\n")
        assert m == {"val_bpb": 2.1}

    def test_crash_output(self):
        m = parse_final_metrics("Traceback (most recent call last):\n  File...")
        assert m == {}


# ---------------------------------------------------------------------------
# TrainRunner — subprocess integration tests
# ---------------------------------------------------------------------------

MOCK_TRAIN_PY = textwrap.dedent('''\
    """Mock train.py for testing — prints fake results instantly."""
    import time
    import random

    time.sleep(0.2)

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


class TestTrainRunner:
    def test_successful_run(self, tmp_path):
        train_py = tmp_path / "train.py"
        train_py.write_text(MOCK_TRAIN_PY)

        runner = TrainRunner(repo_root=tmp_path)
        config = RunConfig(train_py_path=train_py, work_dir=tmp_path)
        result = runner.run(config)

        assert result.return_code == 0
        assert result.error is None
        assert 2.0 < result.metrics["val_bpb"] < 3.0
        assert result.metrics["peak_vram_mb"] == 1024.0
        assert result.metrics["num_steps"] == 10
        assert result.elapsed_seconds > 0

    def test_crash_run(self, tmp_path):
        train_py = tmp_path / "train.py"
        train_py.write_text("raise RuntimeError('boom')\n")

        runner = TrainRunner(repo_root=tmp_path)
        config = RunConfig(train_py_path=train_py, work_dir=tmp_path)
        result = runner.run(config)

        assert result.return_code != 0
        assert result.metrics == {}
        assert result.error is not None
        assert "boom" in result.error

    def test_timeout(self, tmp_path):
        train_py = tmp_path / "train.py"
        train_py.write_text("import time; time.sleep(60)\n")

        runner = TrainRunner(repo_root=tmp_path)
        config = RunConfig(train_py_path=train_py, work_dir=tmp_path, timeout=2)
        result = runner.run(config)

        assert result.return_code == -1
        assert "timeout" in result.error.lower()

    def test_env_overrides(self, tmp_path):
        train_py = tmp_path / "train.py"
        train_py.write_text(textwrap.dedent("""\
            import os
            val = os.environ.get("TEST_OVERRIDE", "missing")
            print(f"val_bpb:          1.0")
            print(f"custom_val: {val}")
        """))

        runner = TrainRunner(repo_root=tmp_path)
        config = RunConfig(
            train_py_path=train_py,
            work_dir=tmp_path,
            env_overrides={"TEST_OVERRIDE": "found"},
        )
        result = runner.run(config)

        assert result.return_code == 0
        assert "found" in result.log_content

    def test_telemetry_jsonl(self, tmp_path):
        telemetry_path = tmp_path / "telemetry.jsonl"
        train_py = tmp_path / "train.py"
        train_py.write_text(textwrap.dedent(f"""\
            import json, os
            tf = os.environ.get("BITRESEARCH_TELEMETRY_FILE")
            if tf:
                with open(tf, "w") as f:
                    json.dump({{"step": 0, "loss": 5.0}}, f)
                    f.write("\\n")
                    json.dump({{"step": 1, "loss": 4.5}}, f)
                    f.write("\\n")
            print("val_bpb:          2.0")
        """))

        runner = TrainRunner(repo_root=tmp_path)
        config = RunConfig(
            train_py_path=train_py,
            work_dir=tmp_path,
            telemetry_path=telemetry_path,
        )
        result = runner.run(config)

        assert result.return_code == 0
        assert len(result.telemetry) == 2
        assert result.telemetry[0]["step"] == 0
        assert result.telemetry[1]["loss"] == 4.5
