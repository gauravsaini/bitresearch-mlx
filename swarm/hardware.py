"""
Hardware detection and capability reporting for Apple Silicon machines.
Reports chip type, memory, GPU cores, and memory bandwidth for
hardware-aware experiment assignment.
"""

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class HardwareInfo:
    """Capability description of an Apple Silicon machine."""

    hostname: str
    chip: str
    total_memory_gb: float
    gpu_cores: int
    cpu_cores_performance: int
    cpu_cores_efficiency: int
    memory_bandwidth_gbps: float
    macos_version: str
    python_version: str
    mlx_available: bool

    @property
    def tier(self) -> str:
        """Classify machine into a tier for experiment assignment.

        - ultra:  >=128 GB  — can run very large models
        - max:    >=64 GB   — can run large models
        - pro:    >=32 GB   — standard experiments
        - base:   <32 GB    — smaller/faster experiments
        """
        if self.total_memory_gb >= 128:
            return "ultra"
        elif self.total_memory_gb >= 64:
            return "max"
        elif self.total_memory_gb >= 32:
            return "pro"
        else:
            return "base"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "HardwareInfo":
        d = json.loads(data)
        d.pop("tier", None)
        return cls(**d)


def _run_sysctl(key: str) -> str:
    """Read a sysctl value."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _run_system_profiler_hardware() -> dict:
    """Get hardware info from system_profiler (cached)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)
        items = data.get("SPHardwareDataType", [{}])
        return items[0] if items else {}
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        return {}


# Known Apple Silicon memory bandwidth (GB/s), conservative estimates
_BANDWIDTH_TABLE = {
    "M1": 68.25,
    "M1 Pro": 200.0,
    "M1 Max": 400.0,
    "M1 Ultra": 800.0,
    "M2": 100.0,
    "M2 Pro": 200.0,
    "M2 Max": 400.0,
    "M2 Ultra": 800.0,
    "M3": 100.0,
    "M3 Pro": 150.0,
    "M3 Max": 400.0,
    "M3 Ultra": 800.0,
    "M4": 120.0,
    "M4 Pro": 273.0,
    "M4 Max": 546.0,
    "M4 Ultra": 819.2,
}


def _estimate_bandwidth(chip: str) -> float:
    """Estimate memory bandwidth from chip name."""
    for key, bw in sorted(_BANDWIDTH_TABLE.items(), key=lambda x: -len(x[0])):
        if key.lower() in chip.lower():
            return bw
    return 100.0  # Conservative fallback


def detect_hardware() -> HardwareInfo:
    """Detect the current machine's hardware capabilities."""
    hw = _run_system_profiler_hardware()

    # Hostname
    hostname = platform.node() or os.uname().nodename

    # Chip name
    chip = hw.get("chip_type", _run_sysctl("machdep.cpu.brand_string") or "Unknown")

    # Total memory
    mem_bytes_str = _run_sysctl("hw.memsize")
    try:
        total_memory_gb = int(mem_bytes_str) / (1024**3)
    except (ValueError, TypeError):
        total_memory_gb = 0.0

    # GPU cores
    gpu_str = hw.get("number_gpus", "0")
    # system_profiler returns GPU core counts in chip info
    try:
        gpu_cores = int("".join(c for c in str(gpu_str) if c.isdigit()) or "0")
    except ValueError:
        gpu_cores = 0

    # If system_profiler didn't give us GPU cores, try IOKit
    if gpu_cores == 0:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            display_data = json.loads(result.stdout)
            displays = display_data.get("SPDisplaysDataType", [])
            for d in displays:
                cores_str = d.get("sppci_cores", "")
                if cores_str:
                    gpu_cores = int("".join(c for c in str(cores_str) if c.isdigit()) or "0")
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    # CPU cores
    try:
        perf_cores = int(_run_sysctl("hw.perflevel0.logicalcpu") or "0")
    except ValueError:
        perf_cores = 0
    try:
        eff_cores = int(_run_sysctl("hw.perflevel1.logicalcpu") or "0")
    except ValueError:
        eff_cores = 0

    # Fallback for CPU cores
    if perf_cores == 0 and eff_cores == 0:
        try:
            total = int(_run_sysctl("hw.logicalcpu") or "0")
            perf_cores = total
        except ValueError:
            perf_cores = os.cpu_count() or 1

    # Memory bandwidth
    bandwidth = _estimate_bandwidth(chip)

    # macOS version
    macos_version = platform.mac_ver()[0] or "unknown"

    # Python version
    python_version = platform.python_version()

    # MLX availability
    try:
        import mlx.core  # noqa: F401
        mlx_available = True
    except ImportError:
        mlx_available = False

    return HardwareInfo(
        hostname=hostname,
        chip=chip,
        total_memory_gb=round(total_memory_gb, 1),
        gpu_cores=gpu_cores,
        cpu_cores_performance=perf_cores,
        cpu_cores_efficiency=eff_cores,
        memory_bandwidth_gbps=bandwidth,
        macos_version=macos_version,
        python_version=python_version,
        mlx_available=mlx_available,
    )


if __name__ == "__main__":
    info = detect_hardware()
    print(json.dumps(info.to_dict(), indent=2))
