This is a **distributed** Apple Silicon (MLX) port of Karpathy's autoresearch — extended for swarm execution across multiple Macs. All training runs natively on MLX with unified memory. No PyTorch or CUDA required.

**Monorepo note:** This project may live inside a larger repo. Always stage only `bitresearch-mlx/` paths. Never use blind `git add -A`.

## Setup

### Single Machine Mode
To set up a new experiment on a single machine, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `uv run train.py` once to establish YOUR baseline on this hardware. Do NOT use baseline numbers from other platforms.
6. **Confirm and go**: Confirm setup looks good.

### Swarm Mode
To run experiments across multiple Macs:

1. **Start the coordinator** on one machine: `uv run bitresearch coordinator`
2. **Start workers** on each additional Mac: `uv run bitresearch worker`
3. Workers auto-discover the coordinator via mDNS (Bonjour).
4. **Submit experiments** by modifying `train.py` and running:
   ```
   uv run bitresearch submit -c ws://COORDINATOR_IP:8765 -m "description" -f train.py
   ```
5. The coordinator distributes experiments to workers based on hardware capabilities.
6. Results flow back automatically and are logged to `results.tsv`.

**Hardware-aware routing**: The coordinator classifies workers into tiers:
- `base` (< 32 GB): Smaller, faster models. Often finds different optima.
- `pro` (32–63 GB): Standard experiments.
- `max` (64–127 GB): Large models.
- `ultra` (≥ 128 GB): Very large models, big batch sizes.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. MLX uses unified memory shared between CPU and GPU. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format
Once the script finishes it prints a summary like this:

```
---
val_bpb:          2.534000
training_seconds: 312.4
total_seconds:    405.7
peak_vram_mb:     27528.9
mfu_percent:      0.00
total_tokens_M:   39.8
num_steps:        46
num_params_M:     50.3
depth:            8
```

## Logging results
When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

In **swarm mode**, the coordinator automatically logs results with worker metadata appended:
```
383abb4	2.667000	26.9	keep	baseline [worker:mini-a1b2c3/Apple M2]
```

## Experiment Loop

### Single Machine
LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code
3. `git add train.py && git commit -m "experiment: <description>"`
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to diagnose
7. Record the results in the tsv
8. If val_bpb improved (lower), `git add results.tsv && git commit --amend --no-edit`
9. If val_bpb is equal or worse, `git reset --hard <previous kept commit>`

### Swarm Mode
The coordinator handles steps 3, 7, 8, 9 automatically. You just:

1. Edit `train.py` with your idea
2. Submit: `uv run bitresearch submit -c ws://COORDINATOR:8765 -m "description" -f train.py`
3. Wait for results (or submit more experiments in parallel!)
4. Check: `uv run bitresearch status`

Multiple experiments run simultaneously across the swarm. The coordinator keeps the best result and discards the rest, maintaining a clean git history.

**Timeout**: Each experiment should take ~7 minutes total. If a run exceeds 15 minutes, the worker kills it and treats it as a failure.

**Crashes**: If a run crashes, the worker reports the error back to the coordinator. The coordinator logs it and moves on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause. The loop runs until manually stopped. You are autonomous.
