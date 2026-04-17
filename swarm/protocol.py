"""
WebSocket message protocol for coordinator <-> worker communication.
All messages are JSON-encoded with a 'type' field for dispatch.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MessageType(str, Enum):
    """All message types in the BitResearch protocol."""

    # Worker -> Coordinator
    WORKER_REGISTER = "worker_register"
    WORKER_HEARTBEAT = "worker_heartbeat"
    EXPERIMENT_RESULT = "experiment_result"
    WORKER_STATUS = "worker_status"

    # Coordinator -> Worker
    EXPERIMENT_ASSIGN = "experiment_assign"
    EXPERIMENT_CANCEL = "experiment_cancel"
    SYNC_DATA = "sync_data"
    COORDINATOR_ACK = "coordinator_ack"


@dataclass
class Message:
    """Base message container."""

    type: str
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sender_id: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "Message":
        d = json.loads(data)
        return cls(**d)


@dataclass
class ExperimentSpec:
    """Specification for a training experiment to run on a worker."""

    experiment_id: str
    train_py_content: str
    description: str
    branch_name: str
    parent_commit: str = ""
    # Hardware requirements
    min_memory_gb: float = 0.0
    preferred_tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentResult:
    """Result of a completed training experiment."""

    experiment_id: str
    worker_id: str
    val_bpb: float
    peak_vram_mb: float
    training_seconds: float
    total_seconds: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int
    status: str  # "success", "crash", "timeout"
    error_message: str = ""
    train_py_content: str = ""
    description: str = ""
    # Worker hardware context
    worker_chip: str = ""
    worker_memory_gb: float = 0.0
    worker_tier: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def memory_gb(self) -> float:
        return round(self.peak_vram_mb / 1024, 1)

    def tsv_row(self, commit_hash: str = "0000000") -> str:
        """Format as a results.tsv row."""
        status = "keep" if self.status == "success" else self.status
        return (
            f"{commit_hash}\t"
            f"{self.val_bpb:.6f}\t"
            f"{self.memory_gb}\t"
            f"{status}\t"
            f"{self.description} [worker:{self.worker_id}/{self.worker_chip}]"
        )


def make_register_message(worker_id: str, hardware_info: dict) -> Message:
    """Create a worker registration message."""
    return Message(
        type=MessageType.WORKER_REGISTER,
        payload={"worker_id": worker_id, "hardware": hardware_info},
        sender_id=worker_id,
    )


def make_heartbeat_message(worker_id: str, status: str = "idle") -> Message:
    """Create a heartbeat message."""
    return Message(
        type=MessageType.WORKER_HEARTBEAT,
        payload={"worker_id": worker_id, "status": status},
        sender_id=worker_id,
    )


def make_experiment_assign_message(spec: ExperimentSpec, coordinator_id: str = "coordinator") -> Message:
    """Create an experiment assignment message."""
    return Message(
        type=MessageType.EXPERIMENT_ASSIGN,
        payload=spec.to_dict(),
        sender_id=coordinator_id,
    )


def make_result_message(result: ExperimentResult) -> Message:
    """Create an experiment result message."""
    return Message(
        type=MessageType.EXPERIMENT_RESULT,
        payload=result.to_dict(),
        sender_id=result.worker_id,
    )


def make_cancel_message(experiment_id: str, coordinator_id: str = "coordinator") -> Message:
    """Create an experiment cancellation message."""
    return Message(
        type=MessageType.EXPERIMENT_CANCEL,
        payload={"experiment_id": experiment_id},
        sender_id=coordinator_id,
    )
