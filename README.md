This is a distributed Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), extended for **swarm execution across multiple Macs**. Inspired by [Darkbloom's](https://darkbloom.dev) architecture for distributed inference on Apple Silicon, this project distributes autonomous research loops across idle Macs on your local network.

Full credit to [@karpathy](https://github.com/karpathy) for the core idea: fixed-time autonomous research loops via `program.md`. This project keeps the same rules — one mutable `train.py`, one metric (`val_bpb`), a fixed 5-minute training budget, and keep-or-revert via git — but scales horizontally across a swarm of Apple Silicon machines using mDNS discovery and WebSocket communication.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               Coordinator (Mac)                 │
│  - Manages experiment queue                     │
│  - Distributes train.py variants to workers     │
│  - Collects results → updates results.tsv       │
│  - Git branch management per experiment         │
│  - Decides keep/discard based on val_bpb        │
│  - Auto-advertises via mDNS (Bonjour)           │
└───────────────┬─────────────────────────────────┘
                │  mDNS discovery + WebSocket
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│  Apple Silicon Macs
│M1 Pro  │ │M4 Max  │ │M2 Ultra│
│32GB    │ │128GB   │ │192GB   │
│ MLX    │ │ MLX    │ │ MLX    │
│train.py│ │train.py│ │train.py│
└────────┘ └────────┘ └────────┘
```

Workers auto-discover the coordinator via **Bonjour/mDNS** (zero-config, built into macOS). No port forwarding, no cloud infra. The coordinator assigns experiments to workers based on hardware capabilities — bigger models go to bigger machines.

## Quick Start

Requirements: Apple Silicon Mac(s), Python 3.10+, [uv](https://docs.astral.sh/uv/).

### 1. Install

```bash
# Clone the repo
gh repo clone <your-username>/bitresearch-mlx
cd bitresearch-mlx

# Install dependencies
uv sync
```

### 2. Prepare Data (on every machine)

```bash
uv run bitresearch prepare
# or directly:
uv run prepare.py
```

### 3. Start the Coordinator (on one machine)

```bash
uv run bitresearch coordinator
```

The coordinator advertises itself via Bonjour. Workers on the same network will auto-discover it.

### 4. Start Workers (on each additional Mac)

```bash
# Auto-discover coordinator via mDNS:
uv run bitresearch worker

# Or specify coordinator explicitly:
uv run bitresearch worker --coordinator-url ws://192.168.1.100:8765
```

### 5. Submit Experiments

Point an AI coding agent (Claude Code, etc.) at `program.md` and let it submit experiments:

```bash
# Submit a train.py variant:
uv run bitresearch submit \
  --coordinator-url ws://localhost:8765 \
  --description "increase depth to 8" \
  --train-py train.py
```

### 6. Check Status

```bash
uv run bitresearch status
uv run bitresearch hardware
```

## Single-Machine Mode

Works exactly like upstream autoresearch-mlx — run experiments locally:

```bash
uv run prepare.py
uv run train.py
```

Then point your coding agent at `program.md` for autonomous experimentation.

## What Matters

| File | Role |
|------|------|
| `prepare.py` | Data prep, tokenizer, dataloader, evaluation. **Read-only.** |
| `train.py` | Model, optimizer, training loop. **The file the agent edits.** |
| `program.md` | The autonomous experiment protocol. |
| `results.tsv` | Logged experiment history. |
| `swarm/` | Distributed swarm infrastructure. |

## Swarm Features

- **Zero-config discovery**: Workers find the coordinator via mDNS/Bonjour — built into every Mac.
- **Hardware-aware routing**: Coordinator knows each worker's chip, memory, and tier. Routes experiments accordingly.
- **Fault-tolerant**: Workers can join/leave freely. Experiments from disconnected workers are re-queued.
- **Git-native**: Every experiment is a git commit. Best results are kept; worse results are discarded.
- **Parallel exploration**: Multiple workers explore different architectures simultaneously.

## Hardware Tiers

| Tier | Memory | Best For |
|------|--------|----------|
| `base` | < 32 GB | Smaller, faster-training models |
| `pro` | 32–63 GB | Standard experiments |
| `max` | 64–127 GB | Large models |
| `ultra` | ≥ 128 GB | Very large models, big batch sizes |

The coordinator assigns experiments to appropriate tiers. The Mac Mini finding from autoresearch-mlx — where smaller hardware found different optimal architectures — is exactly what swarm mode is designed to exploit.

## Differences from autoresearch-mlx

- **Distributed**: Multiple Macs run experiments in parallel.
- **Coordinator/Worker architecture**: Inspired by [Darkbloom](https://darkbloom.dev)'s provider network.
- **mDNS discovery**: Zero-config networking via Bonjour.
- **Hardware-aware**: Experiments are routed to appropriate hardware.
- **Rich CLI**: Beautiful terminal UI with experiment tables and hardware info.
- **Same core**: Uses identical `prepare.py`, `train.py`, and evaluation from autoresearch-mlx.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and nanochat
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon MLX port
- [Darkbloom / Eigen Labs](https://darkbloom.dev) — Distributed inference architecture on Apple Silicon
- [scasella/nanochat-mlx](https://github.com/scasella/nanochat-mlx) — MLX GPT and optimizer reference
- [awni/picochat](https://github.com/awni/picochat) — MLX training patterns
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
