"""Tests for swarm.scaler — adaptive parameter scaling for Apple Silicon."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.hardware import HardwareInfo
from swarm.scaler import ParameterScaler, ScaledParams


@pytest.fixture
def scaler():
    return ParameterScaler()


def make_hardware(chip: str, memory_gb: float, gpu_cores: int = 8) -> HardwareInfo:
    return HardwareInfo(
        hostname="test-host",
        chip=chip,
        total_memory_gb=memory_gb,
        gpu_cores=gpu_cores,
        cpu_cores_performance=4,
        cpu_cores_efficiency=4,
        memory_bandwidth_gbps=100.0,
        macos_version="15.0",
        python_version="3.11",
        mlx_available=True,
    )


def test_scaler_16gb_m1(scaler):
    hw = make_hardware(chip="Apple M1", memory_gb=16.0, gpu_cores=8)
    # Small model: 124M params, depth=12, n_embd=768
    scaled = scaler.scale(
        hardware=hw,
        model_params_m=124.0,
        depth=12,
        n_embd=768,
        seq_len=2048,
        target_total_batch=65536,
    )
    assert scaled.device_batch_size >= 1
    assert scaled.estimated_peak_gb <= hw.total_memory_gb * 0.85
    assert scaled.memory_headroom_gb >= 0
    assert not scaled.warnings
    env = scaled.to_env()
    assert "BITRESEARCH_DEVICE_BATCH_SIZE" in env
    assert "BITRESEARCH_TOTAL_BATCH_SIZE" in env


def test_scaler_scales_up_with_ram(scaler):
    small_hw = make_hardware(chip="Apple M1", memory_gb=16.0, gpu_cores=8)
    large_hw = make_hardware(chip="Apple M2 Ultra", memory_gb=128.0, gpu_cores=76)
    # Same model: 500M params, depth=24, n_embd=1024
    scaled_small = scaler.scale(
        hardware=small_hw,
        model_params_m=500.0,
        depth=24,
        n_embd=1024,
    )
    scaled_large = scaler.scale(
        hardware=large_hw,
        model_params_m=500.0,
        depth=24,
        n_embd=1024,
    )
    assert scaled_large.device_batch_size >= scaled_small.device_batch_size
    assert scaled_large.estimated_peak_gb <= 128.0 * 0.85


def test_scaler_oversized_model_warning(scaler):
    tiny_hw = make_hardware(chip="Apple M1", memory_gb=8.0, gpu_cores=8)
    # Huge model that cannot fit in 8GB (needs ~8GB base just for weights + optimizer)
    scaled = scaler.scale(
        hardware=tiny_hw,
        model_params_m=1000.0,
        depth=32,
        n_embd=2048,
    )
    assert len(scaled.warnings) > 0
    assert scaled.device_batch_size == 1
    assert "exceeds usable memory" in scaled.warnings[0]


def test_scaler_math_consistency(scaler):
    hw = make_hardware(chip="Apple M3 Max", memory_gb=64.0, gpu_cores=40)
    seq_len = 1024
    target_total_batch = 32768
    scaled = scaler.scale(
        hardware=hw,
        model_params_m=200.0,
        depth=16,
        n_embd=768,
        seq_len=seq_len,
        target_total_batch=target_total_batch,
    )
    tokens_per_step = scaled.device_batch_size * seq_len
    expected_total = tokens_per_step * scaled.grad_accum_steps
    assert scaled.total_batch_size == expected_total

