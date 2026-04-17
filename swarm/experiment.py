"""
Experiment definition, tracking, and git integration.
Manages the lifecycle of experiments across the distributed swarm.
"""

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    SUCCESS = "success"
    CRASH = "crash"
    TIMEOUT = "timeout"
    DISCARDED = "discarded"


@dataclass
class Experiment:
    """A single experiment in the swarm."""

    experiment_id: str
    description: str
    train_py_content: str
    branch_name: str
    status: ExperimentStatus = ExperimentStatus.QUEUED
    assigned_worker: str = ""
    parent_commit: str = ""
    commit_hash: str = ""
    val_bpb: float = 0.0
    peak_vram_mb: float = 0.0
    training_seconds: float = 0.0
    total_seconds: float = 0.0
    total_tokens_m: float = 0.0
    num_steps: int = 0
    num_params_m: float = 0.0
    depth: int = 0
    error_message: str = ""
    worker_chip: str = ""
    worker_memory_gb: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    # Hardware constraint
    min_memory_gb: float = 0.0
    preferred_tier: str = ""

    @property
    def memory_gb(self) -> float:
        return round(self.peak_vram_mb / 1024, 1)

    def tsv_row(self) -> str:
        """Format as results.tsv row."""
        commit = self.commit_hash[:7] if self.commit_hash else "0000000"
        status = "keep" if self.status == ExperimentStatus.SUCCESS else str(self.status.value)
        bpb = f"{self.val_bpb:.6f}" if self.val_bpb > 0 else "0.000000"
        mem = f"{self.memory_gb}" if self.peak_vram_mb > 0 else "0.0"
        worker_tag = f" [worker:{self.assigned_worker}/{self.worker_chip}]" if self.assigned_worker else ""
        return f"{commit}\t{bpb}\t{mem}\t{status}\t{self.description}{worker_tag}"


class ExperimentTracker:
    """Tracks experiments across the swarm and manages git state."""

    def __init__(self, repo_dir: str):
        self.repo_dir = Path(repo_dir)
        self.experiments: dict[str, Experiment] = {}
        self.best_val_bpb: float = float("inf")
        self.best_experiment_id: str = ""
        self.results_file = self.repo_dir / "results.tsv"

    def generate_experiment_id(self, description: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = str(time.time())
        raw = f"{description}:{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def create_experiment(
        self,
        description: str,
        train_py_content: str,
        branch_name: str = "",
        parent_commit: str = "",
        min_memory_gb: float = 0.0,
        preferred_tier: str = "",
    ) -> Experiment:
        """Create and register a new experiment."""
        exp_id = self.generate_experiment_id(description)
        if not branch_name:
            branch_name = f"swarm/exp-{exp_id[:8]}"

        exp = Experiment(
            experiment_id=exp_id,
            description=description,
            train_py_content=train_py_content,
            branch_name=branch_name,
            parent_commit=parent_commit,
            min_memory_gb=min_memory_gb,
            preferred_tier=preferred_tier,
        )
        self.experiments[exp_id] = exp
        return exp

    def get_queued_experiments(self) -> list[Experiment]:
        """Get all experiments waiting to be assigned."""
        return [
            e for e in self.experiments.values()
            if e.status == ExperimentStatus.QUEUED
        ]

    def get_running_experiments(self) -> list[Experiment]:
        """Get all currently running experiments."""
        return [
            e for e in self.experiments.values()
            if e.status in (ExperimentStatus.ASSIGNED, ExperimentStatus.RUNNING)
        ]

    def complete_experiment(
        self,
        experiment_id: str,
        val_bpb: float,
        peak_vram_mb: float,
        training_seconds: float,
        total_seconds: float,
        total_tokens_m: float,
        num_steps: int,
        num_params_m: float,
        depth: int,
        status: str,
        error_message: str = "",
        worker_chip: str = "",
        worker_memory_gb: float = 0.0,
    ):
        """Record experiment completion."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return

        exp.val_bpb = val_bpb
        exp.peak_vram_mb = peak_vram_mb
        exp.training_seconds = training_seconds
        exp.total_seconds = total_seconds
        exp.total_tokens_m = total_tokens_m
        exp.num_steps = num_steps
        exp.num_params_m = num_params_m
        exp.depth = depth
        exp.error_message = error_message
        exp.worker_chip = worker_chip
        exp.worker_memory_gb = worker_memory_gb
        exp.completed_at = time.time()

        if status == "success":
            exp.status = ExperimentStatus.SUCCESS
            if val_bpb < self.best_val_bpb:
                self.best_val_bpb = val_bpb
                self.best_experiment_id = experiment_id
        elif status == "timeout":
            exp.status = ExperimentStatus.TIMEOUT
        else:
            exp.status = ExperimentStatus.CRASH

    def should_keep(self, experiment_id: str) -> bool:
        """Determine if an experiment's result should be kept (lower val_bpb wins)."""
        exp = self.experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.SUCCESS:
            return False
        return exp.experiment_id == self.best_experiment_id

    def write_results_tsv(self):
        """Write all experiment results to results.tsv."""
        header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        rows = []

        completed = [
            e for e in self.experiments.values()
            if e.status in (ExperimentStatus.SUCCESS, ExperimentStatus.CRASH,
                           ExperimentStatus.TIMEOUT, ExperimentStatus.DISCARDED)
        ]
        completed.sort(key=lambda e: e.completed_at)

        for exp in completed:
            rows.append(exp.tsv_row())

        with open(self.results_file, "w") as f:
            f.write(header)
            for row in rows:
                f.write(row + "\n")

    def git_commit_experiment(self, experiment_id: str) -> Optional[str]:
        """Commit an experiment's train.py to git and return the commit hash."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return None

        train_py_path = self.repo_dir / "train.py"

        # Write the experiment's train.py
        with open(train_py_path, "w") as f:
            f.write(exp.train_py_content)

        try:
            # Stage and commit
            subprocess.run(
                ["git", "add", "train.py"],
                cwd=str(self.repo_dir),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", f"experiment: {exp.description}"],
                cwd=str(self.repo_dir),
                capture_output=True,
                check=True,
            )
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            commit_hash = result.stdout.strip()
            exp.commit_hash = commit_hash

            # Also write and commit results.tsv
            self.write_results_tsv()
            subprocess.run(
                ["git", "add", "results.tsv"],
                cwd=str(self.repo_dir),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=str(self.repo_dir),
                capture_output=True,
                check=True,
            )

            return commit_hash

        except subprocess.CalledProcessError as e:
            exp.error_message = f"Git error: {e.stderr}"
            return None

    def git_revert_to(self, commit_hash: str):
        """Revert the repo to a specific commit."""
        try:
            subprocess.run(
                ["git", "reset", "--hard", commit_hash],
                cwd=str(self.repo_dir),
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            pass

    def summary(self) -> dict:
        """Return a summary of all experiments."""
        total = len(self.experiments)
        by_status = {}
        for exp in self.experiments.values():
            status = exp.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_experiments": total,
            "by_status": by_status,
            "best_val_bpb": self.best_val_bpb if self.best_val_bpb < float("inf") else None,
            "best_experiment_id": self.best_experiment_id or None,
        }
