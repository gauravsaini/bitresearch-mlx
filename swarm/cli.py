"""
CLI entry point for BitResearch MLX — distributed swarm autoresearch.

Commands:
  bitresearch coordinator  — Start the swarm coordinator
  bitresearch worker       — Start a worker node
  bitresearch submit       — Submit an experiment to the swarm
  bitresearch status       — Show swarm status
  bitresearch hardware     — Show local hardware info
  bitresearch prepare      — Run data preparation
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def setup_logging(verbose: bool = False):
    """Configure rich logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool):
    """BitResearch MLX — Distributed Swarm Autoresearch on Apple Silicon.

    Run autonomous ML research experiments across a swarm of Macs.
    Inspired by Karpathy's autoresearch + Darkbloom's distributed inference.
    """
    setup_logging(verbose)


@main.command()
@click.option("--port", "-p", default=8765, help="WebSocket server port")
@click.option("--repo-dir", "-d", default=".", help="Repository directory")
@click.option("--max-concurrent", "-c", default=0, help="Max concurrent experiments (0=unlimited)")
@click.option("--autonomous/--no-autonomous", default=False, help="Enable autonomous hypothesis loop")
@click.option("--max-generations", default=10, help="Max generations in autonomous loop")
@click.option("--patience", default=3, help="Generations without improvement before stopping")
@click.option("--mutations-per-batch", default=0, help="Mutations per batch (0 = worker count)")
def coordinator(
    port: int,
    repo_dir: str,
    max_concurrent: int,
    autonomous: bool,
    max_generations: int,
    patience: int,
    mutations_per_batch: int,
):
    """Start the swarm coordinator.

    The coordinator manages the experiment queue, distributes work to
    connected workers, and aggregates results. It advertises itself via
    mDNS/Bonjour so workers on the local network auto-discover it.

    Run this on the machine where you want to manage the git repository
    and experiment history.
    """
    from .coordinator import run_coordinator

    repo_path = Path(repo_dir).resolve()
    if not (repo_path / ".git").exists():
        console.print("[yellow]Warning:[/] No .git directory found. Initialize with 'git init' first.")

    banner = Text()
    banner.append("BitResearch MLX", style="bold cyan")
    banner.append(" — Coordinator\n", style="dim")
    banner.append(f"Port: {port}\n")
    banner.append(f"Repo: {repo_path}\n")
    banner.append(f"Max concurrent: {'unlimited' if max_concurrent == 0 else max_concurrent}\n")
    banner.append(f"Autonomous: {'Enabled' if autonomous else 'Disabled'}\n")
    if autonomous:
        banner.append(f"Max generations: {max_generations} | Patience: {patience}\n")
    banner.append("\nWorkers will auto-discover via mDNS (Bonjour)", style="dim italic")
    console.print(Panel(banner, title="🐝 Swarm Coordinator", border_style="cyan"))

    asyncio.run(run_coordinator(
        repo_dir=str(repo_path),
        port=port,
        max_concurrent=max_concurrent,
        autonomous=autonomous,
        max_generations=max_generations,
        patience=patience,
        mutations_per_batch=mutations_per_batch,
    ))


@main.command()
@click.option("--coordinator-url", "-c", default=None, help="Coordinator WebSocket URL (auto-discovers if not set)")
@click.option("--worker-id", "-i", default=None, help="Worker ID (auto-generated if not set)")
@click.option("--work-dir", "-d", default=None, help="Working directory for experiments")
@click.option("--port", "-p", default=8766, help="Worker advertisement port")
def worker(coordinator_url: str | None, worker_id: str | None, work_dir: str | None, port: int):
    """Start a worker node.

    The worker connects to the coordinator (discovered via mDNS or
    specified explicitly), receives experiment assignments, runs
    train.py locally on this machine's Apple Silicon GPU, and reports
    results back.

    Each worker needs:
    - Apple Silicon Mac with MLX installed
    - Data prepared via 'bitresearch prepare'
    """
    from .worker import run_worker

    # Detect and display hardware
    from .hardware import detect_hardware
    hw = detect_hardware()

    banner = Text()
    banner.append("BitResearch MLX", style="bold green")
    banner.append(" — Worker\n", style="dim")
    banner.append(f"Chip: {hw.chip}\n")
    banner.append(f"Memory: {hw.total_memory_gb} GB\n")
    banner.append(f"Tier: {hw.tier}\n")
    banner.append(f"GPU cores: {hw.gpu_cores}\n")
    if coordinator_url:
        banner.append(f"Coordinator: {coordinator_url}\n")
    else:
        banner.append("Coordinator: auto-discover via mDNS\n", style="dim italic")
    console.print(Panel(banner, title="⚡ Swarm Worker", border_style="green"))

    asyncio.run(run_worker(
        worker_id=worker_id,
        coordinator_url=coordinator_url,
        work_dir=work_dir,
        port=port,
    ))


@main.command()
@click.option("--coordinator-url", "-c", required=True, help="Coordinator WebSocket URL")
@click.option("--description", "-m", required=True, help="Experiment description")
@click.option("--train-py", "-f", default="train.py", help="Path to train.py variant")
@click.option("--min-memory", default=0.0, help="Minimum worker memory (GB)")
@click.option("--tier", default="", help="Preferred worker tier (base/pro/max/ultra)")
def submit(coordinator_url: str, description: str, train_py: str, min_memory: float, tier: str):
    """Submit an experiment to the swarm.

    Send a train.py variant to the coordinator for execution on an
    available worker. The coordinator will select the best worker
    based on hardware requirements.
    """
    import websockets
    from .protocol import ExperimentSpec, Message, MessageType

    train_py_path = Path(train_py)
    if not train_py_path.exists():
        console.print(f"[red]Error:[/] {train_py} not found")
        sys.exit(1)

    content = train_py_path.read_text()

    async def _submit():
        async with websockets.connect(coordinator_url) as ws:
            msg = Message(
                type=MessageType.WORKER_STATUS,
                payload={
                    "action": "submit_experiment",
                    "description": description,
                    "train_py_content": content,
                    "min_memory_gb": min_memory,
                    "preferred_tier": tier,
                },
                sender_id="cli",
            )
            await ws.send(msg.to_json())
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            console.print(f"[green]Submitted:[/] {description}")
            console.print(f"Response: {response}")

    try:
        asyncio.run(_submit())
    except Exception as e:
        console.print(f"[red]Failed to submit:[/] {e}")
        sys.exit(1)


@main.command()
def hardware():
    """Show local hardware capabilities.

    Detects and displays Apple Silicon hardware information including
    chip type, memory, GPU cores, memory bandwidth, and the assigned
    tier for experiment routing.
    """
    from .hardware import detect_hardware
    hw = detect_hardware()
    info = hw.to_dict()

    table = Table(title="🍎 Apple Silicon Hardware", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")

    display_order = [
        ("hostname", "Hostname"),
        ("chip", "Chip"),
        ("total_memory_gb", "Total Memory (GB)"),
        ("gpu_cores", "GPU Cores"),
        ("cpu_cores_performance", "CPU Cores (Performance)"),
        ("cpu_cores_efficiency", "CPU Cores (Efficiency)"),
        ("memory_bandwidth_gbps", "Memory Bandwidth (GB/s)"),
        ("tier", "Swarm Tier"),
        ("macos_version", "macOS Version"),
        ("python_version", "Python Version"),
        ("mlx_available", "MLX Available"),
    ]

    for key, label in display_order:
        value = info.get(key, "?")
        if isinstance(value, bool):
            value = "✅ Yes" if value else "❌ No"
        elif isinstance(value, float):
            value = f"{value:.1f}"
        table.add_row(label, str(value))

    console.print(table)


@main.command()
@click.option("--num-shards", "-n", default=10, help="Number of data shards to download")
def prepare(num_shards: int):
    """Prepare training data and tokenizer.

    Downloads data shards from HuggingFace and trains a BPE tokenizer.
    This must be run on every machine in the swarm before it can execute
    experiments.

    Data is cached in ~/.cache/autoresearch/.
    """
    console.print("[cyan]Preparing data and tokenizer...[/]")
    console.print(f"Shards to download: {num_shards}")

    # Run prepare.py
    import subprocess
    result = subprocess.run(
        ["uv", "run", "prepare.py", "--num-shards", str(num_shards)],
        cwd=str(Path(__file__).parent.parent),
    )

    if result.returncode == 0:
        console.print("[green]✅ Data preparation complete![/]")
    else:
        console.print("[red]❌ Data preparation failed[/]")
        sys.exit(1)


@main.command()
def status():
    """Show swarm status (local summary).

    Displays the current experiment history from results.tsv and
    local hardware info.
    """
    # Show results.tsv if it exists
    results_path = Path("results.tsv")
    if results_path.exists():
        table = Table(title="📊 Experiment History", border_style="yellow")
        table.add_column("Commit", style="dim")
        table.add_column("val_bpb", style="bold green")
        table.add_column("Memory (GB)", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Description")

        with open(results_path) as f:
            lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split("\t")
            if len(parts) >= 5:
                status_val = parts[3]
                style = "green" if status_val == "keep" else ("red" if status_val == "crash" else "yellow")
                table.add_row(parts[0], parts[1], parts[2], Text(status_val, style=style), parts[4])

        console.print(table)
    else:
        console.print("[dim]No results.tsv found. Run some experiments first.[/]")

    # Show hardware
    from .hardware import detect_hardware
    hw = detect_hardware()
    console.print(f"\n[cyan]Local:[/] {hw.chip} / {hw.total_memory_gb}GB / tier={hw.tier}")


if __name__ == "__main__":
    main()
