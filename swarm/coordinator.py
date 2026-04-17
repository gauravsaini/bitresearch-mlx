"""
Coordinator — the brain of the BitResearch swarm.
Manages workers, distributes experiments, collects results,
and maintains the experiment history in git.
"""

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

import websockets

from .discovery import CoordinatorAdvertiser, get_local_ip
from .experiment import Experiment, ExperimentStatus, ExperimentTracker
from .hardware import HardwareInfo
from .protocol import (
    ExperimentResult,
    ExperimentSpec,
    Message,
    MessageType,
    make_experiment_assign_message,
)

logger = logging.getLogger(__name__)


class WorkerState:
    """Track state of a connected worker."""

    def __init__(self, worker_id: str, hardware: dict, ws):
        self.worker_id = worker_id
        self.hardware = hardware
        self.ws = ws
        self.status = "idle"
        self.current_experiment_id: str | None = None
        self.last_heartbeat = time.time()
        self.connected_at = time.time()
        self.experiments_completed = 0
        self.total_training_seconds = 0.0

    @property
    def tier(self) -> str:
        return self.hardware.get("tier", "unknown")

    @property
    def chip(self) -> str:
        return self.hardware.get("chip", "unknown")

    @property
    def memory_gb(self) -> float:
        return float(self.hardware.get("total_memory_gb", 0))

    @property
    def is_idle(self) -> bool:
        return self.status == "idle" and self.current_experiment_id is None

    @property
    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 60


class Coordinator:
    """
    Swarm coordinator that manages distributed experiments.

    Responsibilities:
    - Accept worker connections via WebSocket
    - Advertise presence via mDNS
    - Distribute experiments to workers based on hardware capabilities
    - Collect and aggregate results
    - Manage git state (commits, branches, keep/discard)
    - Provide live status dashboard
    """

    def __init__(
        self,
        repo_dir: str,
        port: int = 8765,
        max_concurrent: int = 0,
    ):
        self.repo_dir = Path(repo_dir)
        self.port = port
        self.max_concurrent = max_concurrent  # 0 = unlimited

        self.workers: dict[str, WorkerState] = {}
        self.tracker = ExperimentTracker(str(self.repo_dir))
        self.advertiser = CoordinatorAdvertiser(port=self.port)

        # WebSocket server
        self.server = None

        # Stats
        self.started_at = time.time()

    async def start(self):
        """Start the coordinator server."""
        logger.info(f"Starting coordinator on port {self.port}...")
        logger.info(f"Repository: {self.repo_dir}")
        logger.info(f"Local IP: {get_local_ip()}")

        # Start mDNS advertising
        self.advertiser.start()

        # Start WebSocket server
        self.server = await websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self.port,
            ping_interval=20,
            ping_timeout=60,
            max_size=50 * 1024 * 1024,
        )

        logger.info(f"Coordinator listening on ws://0.0.0.0:{self.port}")
        logger.info("Waiting for workers to connect...")

        # Start background tasks
        asyncio.create_task(self._cleanup_loop())

        # Keep running
        await asyncio.Future()

    async def _handle_connection(self, ws, path=None):
        """Handle a new WebSocket connection."""
        worker_id = None
        try:
            async for raw_msg in ws:
                try:
                    msg = Message.from_json(raw_msg)

                    if msg.type == MessageType.WORKER_REGISTER:
                        worker_id = msg.payload.get("worker_id", "unknown")
                        hardware = msg.payload.get("hardware", {})
                        self.workers[worker_id] = WorkerState(worker_id, hardware, ws)
                        logger.info(
                            f"Worker registered: {worker_id} "
                            f"({hardware.get('chip', '?')} / "
                            f"{hardware.get('total_memory_gb', '?')}GB / "
                            f"tier={hardware.get('tier', '?')})"
                        )
                        # Acknowledge
                        ack = Message(type=MessageType.COORDINATOR_ACK, sender_id="coordinator")
                        await ws.send(ack.to_json())

                        # Check for queued experiments
                        await self._try_assign_experiments()

                    elif msg.type == MessageType.WORKER_HEARTBEAT:
                        wid = msg.payload.get("worker_id")
                        if wid and wid in self.workers:
                            self.workers[wid].last_heartbeat = time.time()
                            self.workers[wid].status = msg.payload.get("status", "idle")
                            if msg.payload.get("status") == "idle":
                                self.workers[wid].current_experiment_id = None
                                await self._try_assign_experiments()

                    elif msg.type == MessageType.EXPERIMENT_RESULT:
                        result = ExperimentResult.from_dict(msg.payload)
                        await self._handle_result(result)

                    elif (
                        msg.type == MessageType.WORKER_STATUS
                        and msg.payload.get("action") == "submit_experiment"
                    ):
                        exp_id = await self.submit_experiment(
                            description=msg.payload.get("description", ""),
                            train_py_content=msg.payload.get("train_py_content", ""),
                            min_memory_gb=float(msg.payload.get("min_memory_gb", 0.0)),
                            preferred_tier=msg.payload.get("preferred_tier", ""),
                        )
                        ack = Message(
                            type=MessageType.COORDINATOR_ACK,
                            payload={"status": "submitted", "experiment_id": exp_id},
                            sender_id="coordinator",
                        )
                        await ws.send(ack.to_json())

                except json.JSONDecodeError:
                    logger.warning(f"Invalid message from {worker_id or 'unknown'}")

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if worker_id and worker_id in self.workers:
                logger.info(f"Worker disconnected: {worker_id}")
                # Re-queue any experiment this worker was running
                worker = self.workers[worker_id]
                if worker.current_experiment_id:
                    exp = self.tracker.experiments.get(worker.current_experiment_id)
                    if exp and exp.status in (ExperimentStatus.ASSIGNED, ExperimentStatus.RUNNING):
                        exp.status = ExperimentStatus.QUEUED
                        exp.assigned_worker = ""
                        logger.info(f"Re-queued experiment {exp.experiment_id[:8]}")
                del self.workers[worker_id]

    async def _handle_result(self, result: ExperimentResult):
        """Process an experiment result from a worker."""
        logger.info(
            f"Result from {result.worker_id}: "
            f"exp={result.experiment_id[:8]} "
            f"val_bpb={result.val_bpb:.6f} "
            f"status={result.status}"
        )

        # Update experiment tracker
        self.tracker.complete_experiment(
            experiment_id=result.experiment_id,
            val_bpb=result.val_bpb,
            peak_vram_mb=result.peak_vram_mb,
            training_seconds=result.training_seconds,
            total_seconds=result.total_seconds,
            total_tokens_m=result.total_tokens_m,
            num_steps=result.num_steps,
            num_params_m=result.num_params_m,
            depth=result.depth,
            status=result.status,
            error_message=result.error_message,
            worker_chip=result.worker_chip,
            worker_memory_gb=result.worker_memory_gb,
        )

        # Update worker stats
        if result.worker_id in self.workers:
            worker = self.workers[result.worker_id]
            worker.experiments_completed += 1
            worker.total_training_seconds += result.training_seconds
            worker.current_experiment_id = None
            worker.status = "idle"

        # Git: commit if successful and best so far
        if result.status == "success" and self.tracker.should_keep(result.experiment_id):
            commit = self.tracker.git_commit_experiment(result.experiment_id)
            if commit:
                logger.info(f"✅ KEPT: {result.experiment_id[:8]} → {commit} (val_bpb={result.val_bpb:.6f})")
            else:
                logger.warning(f"Git commit failed for {result.experiment_id[:8]}")
        elif result.status == "success":
            logger.info(
                f"❌ DISCARDED: {result.experiment_id[:8]} "
                f"(val_bpb={result.val_bpb:.6f} ≥ best {self.tracker.best_val_bpb:.6f})"
            )
        else:
            logger.info(f"💥 {result.status.upper()}: {result.experiment_id[:8]}")
            if result.error_message:
                for line in result.error_message.split("\n")[-5:]:
                    logger.info(f"   {line}")

        # Write updated results
        self.tracker.write_results_tsv()

        # Try to assign more experiments
        await self._try_assign_experiments()

    async def submit_experiment(
        self,
        description: str,
        train_py_content: str,
        min_memory_gb: float = 0.0,
        preferred_tier: str = "",
    ) -> str:
        """Submit a new experiment to the queue. Returns experiment ID."""
        exp = self.tracker.create_experiment(
            description=description,
            train_py_content=train_py_content,
            min_memory_gb=min_memory_gb,
            preferred_tier=preferred_tier,
        )
        logger.info(f"Queued experiment {exp.experiment_id[:8]}: {description}")

        # Try to assign immediately
        await self._try_assign_experiments()

        return exp.experiment_id

    async def _try_assign_experiments(self):
        """Try to assign queued experiments to idle workers."""
        queued = self.tracker.get_queued_experiments()
        if not queued:
            return

        idle_workers = [w for w in self.workers.values() if w.is_idle and w.is_alive]
        if not idle_workers:
            return

        for exp in queued:
            if not idle_workers:
                break

            # Find best worker for this experiment
            worker = self._select_worker(exp, idle_workers)
            if not worker:
                continue

            # Assign
            exp.status = ExperimentStatus.ASSIGNED
            exp.assigned_worker = worker.worker_id
            worker.current_experiment_id = exp.experiment_id
            worker.status = "busy"
            idle_workers.remove(worker)

            spec = ExperimentSpec(
                experiment_id=exp.experiment_id,
                train_py_content=exp.train_py_content,
                description=exp.description,
                branch_name=exp.branch_name,
                parent_commit=exp.parent_commit,
                min_memory_gb=exp.min_memory_gb,
                preferred_tier=exp.preferred_tier,
            )

            msg = make_experiment_assign_message(spec)
            try:
                await worker.ws.send(msg.to_json())
                logger.info(
                    f"Assigned {exp.experiment_id[:8]} → {worker.worker_id} "
                    f"({worker.chip} / {worker.memory_gb}GB)"
                )
            except Exception as e:
                logger.error(f"Failed to assign to {worker.worker_id}: {e}")
                exp.status = ExperimentStatus.QUEUED
                exp.assigned_worker = ""
                worker.current_experiment_id = None
                worker.status = "idle"

    def _select_worker(self, exp: Experiment, idle_workers: list[WorkerState]) -> WorkerState | None:
        """Select the best worker for an experiment based on hardware requirements."""
        candidates = idle_workers.copy()

        # Filter by memory requirement
        if exp.min_memory_gb > 0:
            candidates = [w for w in candidates if w.memory_gb >= exp.min_memory_gb]

        # Filter by tier preference
        if exp.preferred_tier:
            preferred = [w for w in candidates if w.tier == exp.preferred_tier]
            if preferred:
                candidates = preferred

        if not candidates:
            return None

        # Sort by: fewer completed experiments first (load balancing), then by memory (smaller first)
        candidates.sort(key=lambda w: (w.experiments_completed, w.memory_gb))

        return candidates[0]

    async def _cleanup_loop(self):
        """Periodic cleanup of dead workers and stale experiments."""
        while True:
            await asyncio.sleep(30)

            # Check for dead workers
            dead = [wid for wid, w in self.workers.items() if not w.is_alive]
            for wid in dead:
                worker = self.workers[wid]
                logger.warning(f"Worker {wid} missed heartbeat, removing")
                if worker.current_experiment_id:
                    exp = self.tracker.experiments.get(worker.current_experiment_id)
                    if exp and exp.status in (ExperimentStatus.ASSIGNED, ExperimentStatus.RUNNING):
                        exp.status = ExperimentStatus.QUEUED
                        exp.assigned_worker = ""
                del self.workers[wid]

            if dead:
                await self._try_assign_experiments()

    def status_summary(self) -> dict:
        """Return current swarm status."""
        uptime = time.time() - self.started_at
        active_workers = {
            wid: {
                "chip": w.chip,
                "memory_gb": w.memory_gb,
                "tier": w.tier,
                "status": w.status,
                "experiments_completed": w.experiments_completed,
                "alive": w.is_alive,
            }
            for wid, w in self.workers.items()
        }

        return {
            "uptime_seconds": round(uptime, 1),
            "workers": active_workers,
            "worker_count": len(self.workers),
            "idle_workers": sum(1 for w in self.workers.values() if w.is_idle),
            "experiments": self.tracker.summary(),
        }

    def stop(self):
        """Clean shutdown."""
        if self.server:
            self.server.close()
        self.advertiser.stop()
        self.tracker.write_results_tsv()
        logger.info("Coordinator stopped")


async def run_coordinator(
    repo_dir: str,
    port: int = 8765,
    max_concurrent: int = 0,
):
    """Entry point to run the coordinator."""
    coordinator = Coordinator(
        repo_dir=repo_dir,
        port=port,
        max_concurrent=max_concurrent,
    )

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(coordinator)))

    await coordinator.start()


async def _shutdown(coordinator: Coordinator):
    """Graceful shutdown handler."""
    logger.info("Shutting down coordinator...")
    coordinator.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
