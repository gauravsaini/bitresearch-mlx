"""
Worker daemon — runs on each Apple Silicon Mac in the swarm.
Connects to the coordinator via WebSocket, receives experiment assignments,
executes train.py locally, and reports results back.
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import websockets

from .discovery import ServiceDiscovery, WorkerAdvertiser, get_local_ip
from .hardware import HardwareInfo, detect_hardware
from .protocol import (
    ExperimentResult,
    ExperimentSpec,
    Message,
    MessageType,
    make_heartbeat_message,
    make_register_message,
    make_result_message,
)

logger = logging.getLogger(__name__)

# Timeout for a single experiment (training + compile + eval overhead)
EXPERIMENT_TIMEOUT = 900  # 15 minutes max


class Worker:
    """
    A swarm worker that runs training experiments on the local Apple Silicon GPU.

    Lifecycle:
    1. Detect local hardware capabilities
    2. Discover coordinator via mDNS (or connect to specified address)
    3. Register with coordinator
    4. Wait for experiment assignments
    5. Execute train.py in a subprocess
    6. Parse and report results
    7. Return to idle state
    """

    def __init__(
        self,
        worker_id: str | None = None,
        coordinator_url: str | None = None,
        work_dir: str | None = None,
        port: int = 0,
    ):
        self.worker_id = worker_id or f"{os.uname().nodename.split('.')[0]}-{uuid.uuid4().hex[:6]}"
        self.coordinator_url = coordinator_url
        self.work_dir = Path(work_dir) if work_dir else Path.home() / ".bitresearch" / "worker"
        self.port = port or 8766

        self.hardware = detect_hardware()
        self.status = "initializing"
        self.current_experiment: ExperimentSpec | None = None
        self.current_process: subprocess.Popen | None = None
        self.ws = None

        # Discovery and advertising
        self.advertiser = WorkerAdvertiser(
            port=self.port,
            worker_id=self.worker_id,
            hardware_info=self.hardware.to_dict(),
        )
        self.discovery = ServiceDiscovery()

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the worker: discover coordinator, connect, and serve."""
        logger.info(f"Worker {self.worker_id} starting...")
        logger.info(f"Hardware: {self.hardware.chip} / {self.hardware.total_memory_gb}GB / tier={self.hardware.tier}")
        logger.info(f"Work directory: {self.work_dir}")

        # Start mDNS advertising
        self.advertiser.start()

        # Find coordinator
        if not self.coordinator_url:
            logger.info("Searching for coordinator via mDNS...")
            self.discovery.start()
            coordinator = self.discovery.find_coordinator(timeout=30)
            if coordinator:
                addr = coordinator["addresses"][0]
                port = coordinator["port"]
                self.coordinator_url = f"ws://{addr}:{port}"
                logger.info(f"Found coordinator at {self.coordinator_url}")
            else:
                logger.error("Could not find coordinator on local network")
                logger.info("Hint: Start the coordinator first, or specify --coordinator-url")
                return

        self.status = "connecting"
        await self._connect_and_serve()

    async def _connect_and_serve(self):
        """Connect to coordinator and enter the main event loop."""
        retry_delay = 2
        max_retry_delay = 60

        while True:
            try:
                logger.info(f"Connecting to coordinator at {self.coordinator_url}...")
                async with websockets.connect(
                    self.coordinator_url,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=50 * 1024 * 1024,  # 50MB for train.py transfer
                ) as ws:
                    self.ws = ws
                    retry_delay = 2  # Reset on success

                    # Register
                    reg_msg = make_register_message(self.worker_id, self.hardware.to_dict())
                    await ws.send(reg_msg.to_json())
                    self.status = "idle"
                    logger.info("Registered with coordinator")

                    # Main loop: handle messages + send heartbeats
                    await self._message_loop(ws)

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                ConnectionRefusedError,
                OSError,
            ) as e:
                logger.warning(f"Connection lost: {e}. Retrying in {retry_delay}s...")
                self.status = "reconnecting"
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    async def _message_loop(self, ws):
        """Handle incoming messages from coordinator."""
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

        try:
            async for raw_msg in ws:
                try:
                    msg = Message.from_json(raw_msg)
                    await self._handle_message(msg, ws)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid message received: {raw_msg[:100]}")
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self, ws):
        """Send periodic heartbeats to coordinator."""
        while True:
            try:
                status = "busy" if self.current_experiment else "idle"
                hb = make_heartbeat_message(self.worker_id, status)
                await ws.send(hb.to_json())
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def _handle_message(self, msg: Message, ws):
        """Dispatch incoming messages."""
        if msg.type == MessageType.EXPERIMENT_ASSIGN:
            spec = ExperimentSpec.from_dict(msg.payload)
            logger.info(f"Received experiment: {spec.experiment_id} — {spec.description}")
            # Run in background so we keep handling messages
            asyncio.create_task(self._run_experiment(spec, ws))

        elif msg.type == MessageType.EXPERIMENT_CANCEL:
            exp_id = msg.payload.get("experiment_id")
            if self.current_experiment and self.current_experiment.experiment_id == exp_id:
                logger.info(f"Cancelling experiment {exp_id}")
                self._kill_current_process()

        elif msg.type == MessageType.COORDINATOR_ACK:
            logger.debug("Coordinator acknowledged")

    async def _run_experiment(self, spec: ExperimentSpec, ws):
        """Execute a training experiment and report results."""
        self.current_experiment = spec
        self.status = "running"

        exp_dir = self.work_dir / f"exp-{spec.experiment_id[:8]}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write train.py to experiment directory
        train_py = exp_dir / "train.py"
        with open(train_py, "w") as f:
            f.write(spec.train_py_content)

        # Symlink prepare.py if not present (worker needs data access)
        prepare_py = exp_dir / "prepare.py"
        if not prepare_py.exists():
            # Look for prepare.py in the repo directory
            repo_prepare = Path(__file__).parent.parent / "prepare.py"
            if repo_prepare.exists():
                os.symlink(repo_prepare.resolve(), prepare_py)
            else:
                logger.warning("prepare.py not found — experiment may fail")

        log_file = exp_dir / "run.log"
        result = None

        try:
            logger.info(f"Starting experiment {spec.experiment_id[:8]}: {spec.description}")
            t0 = time.time()

            # Run uv run train.py in the experiment directory
            self.current_process = subprocess.Popen(
                ["uv", "run", "train.py"],
                cwd=str(exp_dir),
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
            )

            # Wait with timeout
            try:
                return_code = self.current_process.wait(timeout=EXPERIMENT_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning(f"Experiment {spec.experiment_id[:8]} timed out")
                self.current_process.kill()
                self.current_process.wait()
                result = ExperimentResult(
                    experiment_id=spec.experiment_id,
                    worker_id=self.worker_id,
                    val_bpb=0.0,
                    peak_vram_mb=0.0,
                    training_seconds=0.0,
                    total_seconds=time.time() - t0,
                    total_tokens_m=0.0,
                    num_steps=0,
                    num_params_m=0.0,
                    depth=0,
                    status="timeout",
                    error_message="Exceeded 15 minute timeout",
                    train_py_content=spec.train_py_content,
                    description=spec.description,
                    worker_chip=self.hardware.chip,
                    worker_memory_gb=self.hardware.total_memory_gb,
                    worker_tier=self.hardware.tier,
                )
                await self._send_result(result, ws)
                return

            # Parse output
            total_seconds = time.time() - t0

            if return_code != 0:
                # Read last 50 lines for error
                error_lines = self._tail_file(log_file, 50)
                result = ExperimentResult(
                    experiment_id=spec.experiment_id,
                    worker_id=self.worker_id,
                    val_bpb=0.0,
                    peak_vram_mb=0.0,
                    training_seconds=0.0,
                    total_seconds=total_seconds,
                    total_tokens_m=0.0,
                    num_steps=0,
                    num_params_m=0.0,
                    depth=0,
                    status="crash",
                    error_message=error_lines,
                    train_py_content=spec.train_py_content,
                    description=spec.description,
                    worker_chip=self.hardware.chip,
                    worker_memory_gb=self.hardware.total_memory_gb,
                    worker_tier=self.hardware.tier,
                )
            else:
                # Parse results from log
                result = self._parse_results(log_file, spec, total_seconds)

        except Exception as e:
            logger.error(f"Experiment {spec.experiment_id[:8]} failed: {e}")
            result = ExperimentResult(
                experiment_id=spec.experiment_id,
                worker_id=self.worker_id,
                val_bpb=0.0,
                peak_vram_mb=0.0,
                training_seconds=0.0,
                total_seconds=0.0,
                total_tokens_m=0.0,
                num_steps=0,
                num_params_m=0.0,
                depth=0,
                status="crash",
                error_message=str(e),
                train_py_content=spec.train_py_content,
                description=spec.description,
                worker_chip=self.hardware.chip,
                worker_memory_gb=self.hardware.total_memory_gb,
                worker_tier=self.hardware.tier,
            )
        finally:
            self.current_experiment = None
            self.current_process = None
            self.status = "idle"

        if result:
            await self._send_result(result, ws)

    def _parse_results(self, log_file: Path, spec: ExperimentSpec, total_seconds: float) -> ExperimentResult:
        """Parse training results from the log file."""
        content = ""
        try:
            with open(log_file, "r") as f:
                content = f.read()
        except Exception:
            pass

        def extract(key: str, default: float = 0.0) -> float:
            match = re.search(rf"^{key}:\s+([0-9.]+)", content, re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
            return default

        val_bpb = extract("val_bpb")
        peak_vram_mb = extract("peak_vram_mb")
        training_seconds = extract("training_seconds")
        total_tokens_m = extract("total_tokens_M")
        num_steps = int(extract("num_steps"))
        num_params_m = extract("num_params_M")
        depth = int(extract("depth"))

        status = "success" if val_bpb > 0 else "crash"

        return ExperimentResult(
            experiment_id=spec.experiment_id,
            worker_id=self.worker_id,
            val_bpb=val_bpb,
            peak_vram_mb=peak_vram_mb,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            total_tokens_m=total_tokens_m,
            num_steps=num_steps,
            num_params_m=num_params_m,
            depth=depth,
            status=status,
            train_py_content=spec.train_py_content,
            description=spec.description,
            worker_chip=self.hardware.chip,
            worker_memory_gb=self.hardware.total_memory_gb,
            worker_tier=self.hardware.tier,
        )

    async def _send_result(self, result: ExperimentResult, ws):
        """Send experiment result to coordinator."""
        msg = make_result_message(result)
        try:
            await ws.send(msg.to_json())
            logger.info(
                f"Reported result: {result.experiment_id[:8]} — "
                f"val_bpb={result.val_bpb:.6f} status={result.status}"
            )
        except Exception as e:
            logger.error(f"Failed to send result: {e}")

    def _kill_current_process(self):
        """Kill the currently running experiment process."""
        if self.current_process:
            try:
                self.current_process.kill()
                self.current_process.wait(timeout=10)
            except Exception:
                pass

    @staticmethod
    def _tail_file(path: Path, n: int = 50) -> str:
        """Read last n lines of a file."""
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except Exception:
            return ""

    def stop(self):
        """Clean shutdown."""
        self._kill_current_process()
        self.advertiser.stop()
        self.discovery.stop()
        logger.info("Worker stopped")


async def run_worker(
    worker_id: str | None = None,
    coordinator_url: str | None = None,
    work_dir: str | None = None,
    port: int = 0,
):
    """Entry point to run a worker."""
    worker = Worker(
        worker_id=worker_id,
        coordinator_url=coordinator_url,
        work_dir=work_dir,
        port=port,
    )

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(worker)))

    await worker.start()


async def _shutdown(worker: Worker):
    """Graceful shutdown handler."""
    logger.info("Shutting down worker...")
    worker.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
