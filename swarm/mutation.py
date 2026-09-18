"""
Mutation engine for autonomous experiment generation.

Generates candidate variations of train.py by perturbing hyperparameters,
architectural parameters, or optimization schedules.
"""

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class MutationCandidate:
    """A generated experiment variation ready to run."""

    train_py_content: str
    description: str
    mutation_type: str
    generation: int = 0
    parent_experiment_id: str = ""
    mutations_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.train_py_content.encode()).hexdigest()[:12]


class MutationStrategy(Protocol):
    """Protocol for mutation generators."""

    def mutate(
        self, base_content: str, history: list[Any], count: int = 1
    ) -> list[MutationCandidate]: ...


class HyperparamPerturbStrategy:
    """
    Perturbs numeric hyperparameters in train.py (learning rates, weight decay,
    warmup/warmdown schedules) within safe bounds.
    """

    # param_name -> (min_val, max_val, is_float, default)
    TUNABLE_PARAMS = {
        "EMBEDDING_LR": (0.05, 1.5, True, 0.6),
        "UNEMBEDDING_LR": (0.0005, 0.02, True, 0.004),
        "MATRIX_LR": (0.005, 0.15, True, 0.04),
        "SCALAR_LR": (0.05, 1.5, True, 0.5),
        "WEIGHT_DECAY": (0.0, 0.5, True, 0.2),
        "WARMDOWN_RATIO": (0.2, 0.8, True, 0.5),
        "WARMUP_RATIO": (0.0, 0.08, True, 0.0),
    }

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def _extract_param(self, content: str, param: str) -> float | None:
        pattern = rf"^{param}\s*=\s*([0-9.eE+-]+)"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _replace_param(self, content: str, param: str, new_val: float) -> str:
        pattern = rf"^({param}\s*=\s*)([0-9.eE+-]+)"
        # Format cleanly: if integer or small float
        val_str = f"{new_val:.6g}"
        return re.sub(pattern, rf"\g<1>{val_str}", content, flags=re.MULTILINE)

    def mutate(
        self, base_content: str, history: list[Any], count: int = 1
    ) -> list[MutationCandidate]:
        candidates: list[MutationCandidate] = []
        available_params = list(self.TUNABLE_PARAMS.keys())

        attempts = 0
        while len(candidates) < count and attempts < count * 10:
            attempts += 1
            # Pick 1 to 2 parameters to mutate simultaneously
            num_to_mutate = self._rng.choice([1, 1, 2])
            chosen = self._rng.sample(available_params, k=num_to_mutate)

            content = base_content
            mutations: dict[str, Any] = {}
            descriptions = []

            for param in chosen:
                min_v, max_v, _, default = self.TUNABLE_PARAMS[param]
                current = self._extract_param(content, param)
                if current is None:
                    current = default

                # Perturb by a factor between 0.6x and 1.6x, or jitter
                factor = self._rng.uniform(0.65, 1.55)
                # If current is 0.0 (like WARMUP_RATIO), jitter directly
                if current == 0.0:
                    new_val = self._rng.uniform(0.01, 0.05)
                else:
                    new_val = current * factor

                # Clamp to bounds
                new_val = max(min_v, min(max_v, new_val))
                if round(new_val, 6) == round(current, 6):
                    continue

                content = self._replace_param(content, param, new_val)
                mutations[param] = round(new_val, 6)
                descriptions.append(f"{param}: {current:.4g}→{new_val:.4g}")

            if not mutations:
                continue

            desc = "Mutate " + ", ".join(descriptions)
            candidates.append(
                MutationCandidate(
                    train_py_content=content,
                    description=desc,
                    mutation_type="hyperparam_perturb",
                    mutations_applied=mutations,
                )
            )

        return candidates


class ArchitectureScaleStrategy:
    """
    Mutates structural architectural parameters like DEPTH and ASPECT_RATIO.
    """

    ALLOWED_DEPTHS = [2, 3, 4, 6, 8]
    ALLOWED_ASPECT_RATIOS = [32, 48, 64, 96, 128]

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def _extract_int(self, content: str, param: str) -> int | None:
        pattern = rf"^{param}\s*=\s*([0-9]+)"
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _replace_int(self, content: str, param: str, new_val: int) -> str:
        pattern = rf"^({param}\s*=\s*)([0-9]+)"
        return re.sub(pattern, rf"\g<1>{new_val}", content, flags=re.MULTILINE)

    def mutate(
        self, base_content: str, history: list[Any], count: int = 1
    ) -> list[MutationCandidate]:
        candidates: list[MutationCandidate] = []
        current_depth = self._extract_int(base_content, "DEPTH") or 4
        current_aspect = self._extract_int(base_content, "ASPECT_RATIO") or 64

        attempts = 0
        while len(candidates) < count and attempts < count * 10:
            attempts += 1
            choice = self._rng.choice(["depth", "aspect", "both"])
            content = base_content
            mutations = {}
            descriptions = []

            if choice in ("depth", "both"):
                depth_choices = [d for d in self.ALLOWED_DEPTHS if d != current_depth]
                if depth_choices:
                    new_depth = self._rng.choice(depth_choices)
                    content = self._replace_int(content, "DEPTH", new_depth)
                    mutations["DEPTH"] = new_depth
                    descriptions.append(f"DEPTH: {current_depth}→{new_depth}")

            if choice in ("aspect", "both"):
                aspect_choices = [
                    a for a in self.ALLOWED_ASPECT_RATIOS if a != current_aspect
                ]
                if aspect_choices:
                    new_aspect = self._rng.choice(aspect_choices)
                    content = self._replace_int(content, "ASPECT_RATIO", new_aspect)
                    mutations["ASPECT_RATIO"] = new_aspect
                    descriptions.append(f"ASPECT_RATIO: {current_aspect}→{new_aspect}")

            if not mutations:
                continue

            desc = "Arch: " + ", ".join(descriptions)
            candidates.append(
                MutationCandidate(
                    train_py_content=content,
                    description=desc,
                    mutation_type="architecture_scale",
                    mutations_applied=mutations,
                )
            )

        return candidates


class MutationEngine:
    """
    Coordinates mutation strategies to generate batches of diverse experiment
    candidates for the swarm.
    """

    def __init__(
        self,
        strategies: list[MutationStrategy] | None = None,
        seed: int | None = None,
    ):
        self.strategies = strategies or [
            HyperparamPerturbStrategy(seed=seed),
            ArchitectureScaleStrategy(seed=seed),
        ]
        self._seen_hashes: set[str] = set()

    def generate_batch(
        self,
        base_content: str,
        history: list[Any],
        count: int,
        generation: int = 0,
        parent_experiment_id: str = "",
    ) -> list[MutationCandidate]:
        """Generate `count` unique mutation candidates."""
        batch: list[MutationCandidate] = []
        if not self.strategies or count <= 0:
            return batch

        strategy_idx = 0
        attempts = 0
        max_attempts = count * 20

        while len(batch) < count and attempts < max_attempts:
            attempts += 1
            strat = self.strategies[strategy_idx % len(self.strategies)]
            strategy_idx += 1

            generated = strat.mutate(base_content, history, count=1)
            for c in generated:
                h = c.content_hash
                if h not in self._seen_hashes and c.train_py_content != base_content:
                    self._seen_hashes.add(h)
                    c.generation = generation
                    c.parent_experiment_id = parent_experiment_id
                    batch.append(c)
                    if len(batch) >= count:
                        break

        return batch
