"""
Adaptive parameter scaler for heterogeneous Apple Silicon.

Given a HardwareInfo and model config, computes the largest safe
device_batch_size and matching grad_accum_steps so each worker
runs at its limit without OOM.
"""

from dataclasses import dataclass, field

from .hardware import HardwareInfo


@dataclass
class ScaledParams:
    """Optimal training parameters for a specific worker."""

    device_batch_size: int
    grad_accum_steps: int
    total_batch_size: int
    estimated_peak_gb: float
    memory_headroom_gb: float
    warnings: list[str] = field(default_factory=list)

    def to_env(self) -> dict[str, str]:
        """Return env-var overrides for the runner."""
        return {
            "BITRESEARCH_DEVICE_BATCH_SIZE": str(self.device_batch_size),
            "BITRESEARCH_TOTAL_BATCH_SIZE": str(self.total_batch_size),
        }


class ParameterScaler:
    """Compute optimal batch sizes per worker hardware.

    Memory model (rough, conservative for MLX unified memory):
        params_gb  = model_params_m * 2 / 1024   (bfloat16)
        optim_gb   = 3 * params_gb                (Adam m + v + fp32 copy)
        act_gb     = batch * seq_len * n_embd * depth * 4 / 1e9
        peak_gb    = params_gb + optim_gb + act_gb
    """

    MEMORY_HEADROOM = 0.85  # Use at most 85% of unified memory

    def scale(
        self,
        hardware: HardwareInfo,
        model_params_m: float,
        depth: int,
        n_embd: int,
        seq_len: int = 2048,
        target_total_batch: int = 65536,
    ) -> ScaledParams:
        usable_gb = hardware.total_memory_gb * self.MEMORY_HEADROOM
        params_gb = model_params_m * 2 / 1024
        optim_gb = 3 * params_gb
        base_gb = params_gb + optim_gb

        warnings: list[str] = []
        if base_gb > usable_gb:
            warnings.append(
                f"Model alone ({base_gb:.1f}GB) exceeds usable memory ({usable_gb:.1f}GB)"
            )
            return ScaledParams(
                device_batch_size=1,
                grad_accum_steps=target_total_batch // seq_len,
                total_batch_size=target_total_batch,
                estimated_peak_gb=base_gb,
                memory_headroom_gb=usable_gb - base_gb,
                warnings=warnings,
            )

        # Binary search for largest batch size (powers of 2) that fits
        best_batch = 1
        for exp in range(1, 12):  # 2^1 .. 2^11 = 2 .. 2048
            batch = 2**exp
            act_gb = batch * seq_len * n_embd * depth * 4 / 1e9
            peak = base_gb + act_gb
            if peak <= usable_gb:
                best_batch = batch
            else:
                break

        # Compute peak for chosen batch
        act_gb = best_batch * seq_len * n_embd * depth * 4 / 1e9
        estimated_peak = base_gb + act_gb

        # Grad accum to hit target total batch
        tokens_per_step = best_batch * seq_len
        grad_accum = max(1, target_total_batch // tokens_per_step)
        actual_total_batch = tokens_per_step * grad_accum

        return ScaledParams(
            device_batch_size=best_batch,
            grad_accum_steps=grad_accum,
            total_batch_size=actual_total_batch,
            estimated_peak_gb=round(estimated_peak, 2),
            memory_headroom_gb=round(usable_gb - estimated_peak, 2),
            warnings=warnings,
        )
