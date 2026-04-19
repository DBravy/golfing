"""
Prioritized replay training. Extends the correctness-gating intuition with a
memory buffer of samples the model got wrong, sampled with probability
proportional to loss. Batches strictly alternate between fresh data and replay.

Design choices (set by user):
  - Priority = mean CE loss on incorrect answer tokens. Fully-correct samples
    are never stored. If a replayed sample becomes fully correct, it's removed.
  - Displacement-based eviction: when buffer is full, a new sample replaces the
    current-minimum-priority sample iff its priority is higher.
  - Decay factor 0.99 applied globally every step. Priorities of samples that
    sit unvisited drift down; fresh-arrival priorities dominate over time.
  - Strict alternation: step 0 fresh, step 1 replay, step 2 fresh, ...
  - Option (c): per-sample priorities updated in place using the losses
    already computed during the training forward pass. No extra forward pass.
  - No warmup. Replay starts as soon as the buffer has any samples.

Usage:
    # Single run with defaults
    python memory.py

    # Match the baseline/gated seed sweep
    python memory.py --seeds 0,1,2,3 --log_root runs/memory_seeds

    # Sweep buffer size
    python memory.py --seeds 0,1,2 --sweep_buffer_size 250,1000,4000 \\
        --log_root runs/buffer_sweep

    # Stack memory on top of gating inside fresh batches
    python memory.py --gate_fresh --seeds 0,1,2,3 --log_root runs/memory_gated
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.baseline import (
    Config,
    TASK_NAMES,
    VOCAB_SIZE,
    PAD_ID,
    sample_task,
    TinyGPT,
    count_params,
    evaluate,
    get_lr,
    auto_name,
    crossing_step,
    _parse_int_list,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class MemoryConfig(Config):
    buffer_size: int = 1000
    decay_factor: float = 0.99
    # When True, fresh batches ALSO gate out correct tokens (stacked on replay).
    # When False, fresh batches train on all answer tokens; only replay biases
    # toward hard samples. Default False for clean attribution.
    gate_fresh: bool = False


def auto_name_memory(cfg: MemoryConfig) -> str:
    base = auto_name(cfg)
    gate_tag = "_gfresh" if cfg.gate_fresh else ""
    return f"mem_b{cfg.buffer_size}_dec{cfg.decay_factor:g}{gate_tag}_{base}"


# =============================================================================
# MEMORY
# =============================================================================

class Memory:
    """Prioritized replay buffer. Fixed capacity, top-k by priority, global decay.

    Implementation is deliberately simple: parallel Python lists. At n=1000,
    O(n) scans per insert/sample/decay are ~microseconds and not the bottleneck.
    """

    def __init__(self, capacity: int, decay: float):
        self.capacity = capacity
        self.decay = decay
        self.seqs = []            # List[List[int]]
        self.answer_starts = []   # List[int]
        self.task_ids = []        # List[int]
        self.priorities = []      # List[float]

    def __len__(self):
        return len(self.seqs)

    def decay_all(self):
        """Multiplicative decay on every priority. Called once per step."""
        d = self.decay
        self.priorities = [p * d for p in self.priorities]

    def insert(self, seq, answer_start, task_id, priority):
        """Insert a fresh incorrect sample. Displaces min if buffer is full."""
        if priority <= 0:
            return False  # don't store correct samples
        if len(self.seqs) < self.capacity:
            self.seqs.append(list(seq))
            self.answer_starts.append(answer_start)
            self.task_ids.append(task_id)
            self.priorities.append(priority)
            return True
        # Buffer full: find min priority.
        min_idx = 0
        min_p = self.priorities[0]
        for i in range(1, len(self.priorities)):
            if self.priorities[i] < min_p:
                min_p = self.priorities[i]
                min_idx = i
        if priority > min_p:
            self.seqs[min_idx] = list(seq)
            self.answer_starts[min_idx] = answer_start
            self.task_ids[min_idx] = task_id
            self.priorities[min_idx] = priority
            return True
        return False

    def sample_indices(self, n: int, np_rng):
        """Priority-weighted sampling with replacement. Returns list of ints."""
        if len(self) == 0:
            return None
        pri = np.asarray(self.priorities, dtype=np.float64)
        total = pri.sum()
        if total <= 0:
            return None
        probs = pri / total
        idx = np_rng.choice(len(self), size=n, replace=True, p=probs)
        return [int(i) for i in idx]

    def apply_replay_updates(self, unique_idx_to_priority):
        """Given {buffer_idx: new_priority}, update in place (priority > 0) or
        remove (priority <= 0). Handles index shifts from removal safely.
        """
        # Separate keeps from removes.
        keeps = [(i, p) for i, p in unique_idx_to_priority.items() if p > 0]
        removes = sorted(
            [i for i, p in unique_idx_to_priority.items() if p <= 0],
            reverse=True,
        )
        # Keeps first: direct index write, no shift.
        for i, p in keeps:
            self.priorities[i] = p
        # Removes descending: later indices pop first, earlier indices unaffected.
        for i in removes:
            self._remove_at(i)

    def _remove_at(self, idx):
        self.seqs.pop(idx)
        self.answer_starts.pop(idx)
        self.task_ids.pop(idx)
        self.priorities.pop(idx)

    def task_fractions(self):
        """Fraction of buffer belonging to each task (for diagnostics)."""
        if len(self) == 0:
            return {name: 0.0 for name in TASK_NAMES}
        counts = [0] * len(TASK_NAMES)
        for t in self.task_ids:
            counts[t] += 1
        n = len(self)
        return {TASK_NAMES[i]: counts[i] / n for i in range(len(TASK_NAMES))}


# =============================================================================
# BATCH HELPERS
# =============================================================================
# Refactored from baseline so fresh and replay batches can share padding logic.

def generate_fresh_samples(rng, cfg, n):
    """Return a list of (ids, answer_start, task_idx) tuples, sampled fresh."""
    samples = []
    for _ in range(n):
        task_idx = rng.choices(range(len(TASK_NAMES)), weights=cfg.task_mix, k=1)[0]
        task_name = TASK_NAMES[task_idx]
        ids, ans_start = sample_task(rng, task_name, cfg)
        samples.append((ids, ans_start, task_idx))
    return samples


def pad_samples(samples):
    """Pad a list of (ids, answer_start, task_idx) into the batch tensor shape
    that matches baseline's make_batch output: (input_ids, loss_mask, task_ids,
    answer_starts).
    """
    n = len(samples)
    max_len = max(len(s[0]) for s in samples)
    input_ids = torch.full((n, max_len), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((n, max_len), dtype=torch.float)
    task_ids = torch.zeros(n, dtype=torch.long)
    ans_starts = []
    for i, (ids, ans_start, task_idx) in enumerate(samples):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        loss_mask[i, ans_start:len(ids)] = 1.0
        task_ids[i] = task_idx
        ans_starts.append(ans_start)
    return input_ids, loss_mask, task_ids, ans_starts


def compute_per_sample_priority(logits, targets, mask, loss_per_tok):
    """For each sample in the batch, compute priority = mean CE loss over
    incorrect answer tokens. Fully-correct samples get priority 0.
    Returns a Python list of length B.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets)
        incorrect_mask = mask.bool() & ~correct   # (B, T)
        has_incorrect = incorrect_mask.any(dim=-1)    # (B,)
        # Sum losses on incorrect positions, count them, divide. Vectorized.
        sums = (loss_per_tok * incorrect_mask.float()).sum(dim=-1)
        counts = incorrect_mask.float().sum(dim=-1).clamp(min=1.0)
        means = sums / counts
        # Zero out samples with no incorrect tokens.
        priorities = torch.where(has_incorrect, means, torch.zeros_like(means))
    return priorities.cpu().tolist()


# =============================================================================
# TRAINING
# =============================================================================

def train_memory(cfg: MemoryConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device} | buffer={cfg.buffer_size} | decay={cfg.decay_factor} | "
          f"gate_fresh={cfg.gate_fresh}")

    # Probe max_len.
    probe_rng = random.Random(0)
    max_len = 0
    for _ in range(200):
        for name in TASK_NAMES:
            ids, _ = sample_task(probe_rng, name, cfg)
            max_len = max(max_len, len(ids))
    max_len = max_len + 4
    print(f"max_len: {max_len}")

    model = TinyGPT(cfg, max_len=max_len).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log = {
        "step": [],
        "loss": [],
        "per_task_acc": {name: [] for name in TASK_NAMES},
        "per_task_loss": {name: [] for name in TASK_NAMES},
        "buffer_size": [],
        "buffer_task_frac": {name: [] for name in TASK_NAMES},
        "mean_priority": [],
        "replay_frac_window": [],  # fraction of steps this eval window that were replay
        "n_params": n_params,
        "buffer_capacity": cfg.buffer_size,
        "decay_factor": cfg.decay_factor,
        "gate_fresh": cfg.gate_fresh,
    }

    memory = Memory(capacity=cfg.buffer_size, decay=cfg.decay_factor)
    rng = random.Random(cfg.seed + 1)
    np_rng = np.random.default_rng(cfg.seed + 100)

    model.train()
    t_start = time.time()
    running_loss = 0.0
    running_per_task_loss = {name: 0.0 for name in TASK_NAMES}
    running_per_task_count = {name: 0 for name in TASK_NAMES}
    replay_steps_window = 0
    fresh_steps_window = 0

    for step in range(cfg.total_steps):
        # Strict alternation. Fall back to fresh if replay buffer empty.
        want_replay = (step % 2 == 1)
        is_replay = want_replay and len(memory) > 0

        if is_replay:
            indices = memory.sample_indices(cfg.batch_size, np_rng)
            if indices is None:
                is_replay = False

        if is_replay:
            samples = [
                (memory.seqs[i], memory.answer_starts[i], memory.task_ids[i])
                for i in indices
            ]
            replay_steps_window += 1
        else:
            samples = generate_fresh_samples(rng, cfg, cfg.batch_size)
            indices = None
            fresh_steps_window += 1

        input_ids, loss_mask, task_ids, _ = pad_samples(samples)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        # Optional: gate correct tokens out of fresh batches.
        # Replay batches are trained on fully (they're already selected as hard).
        if cfg.gate_fresh and not is_replay:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                gate = (preds != targets).float()
            effective_mask = mask * gate
        else:
            effective_mask = mask

        total_kept = effective_mask.sum().clamp(min=1.0)
        loss = (loss_per_tok * effective_mask).sum() / total_kept

        # Per-sample priorities (option c: reuse the forward pass).
        priorities = compute_per_sample_priority(logits, targets, mask, loss_per_tok)

        # Per-task loss tracking (for logs that compare to baseline/gated).
        with torch.no_grad():
            for t_idx, name in enumerate(TASK_NAMES):
                task_filter = (task_ids == t_idx).float().unsqueeze(-1) * mask
                tf = task_filter.sum()
                if tf > 0:
                    tl = (loss_per_tok * task_filter).sum() / tf
                    running_per_task_loss[name] += tl.item()
                    running_per_task_count[name] += 1

        optimizer.zero_grad(set_to_none=True)
        lr_now = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running_loss += loss.item()

        # Memory update.
        if is_replay:
            # Dedupe indices (sampling with replacement can repeat). All copies
            # of the same buffer index have the same priority (same sample, same
            # forward pass), so just take one per unique index.
            unique_map = {}
            for batch_pos, buf_idx in enumerate(indices):
                if buf_idx not in unique_map:
                    unique_map[buf_idx] = priorities[batch_pos]
            memory.apply_replay_updates(unique_map)
        else:
            # Insert incorrect fresh samples. Fully-correct ones are discarded
            # by insert() (priority <= 0 short-circuits).
            for i, (ids, ans_start, task_idx) in enumerate(samples):
                memory.insert(ids, ans_start, task_idx, priorities[i])

        # Global decay, every step.
        memory.decay_all()

        if (step + 1) % cfg.eval_every == 0 or step == 0:
            denom = cfg.eval_every if step > 0 else 1
            avg_loss = running_loss / max(1, denom)
            running_loss = 0.0

            accs = evaluate(model, cfg, device)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed

            log["step"].append(step + 1)
            log["loss"].append(avg_loss)
            log["buffer_size"].append(len(memory))
            tf = memory.task_fractions()
            mean_p = float(np.mean(memory.priorities)) if len(memory) > 0 else 0.0
            log["mean_priority"].append(mean_p)
            total_window = replay_steps_window + fresh_steps_window
            log["replay_frac_window"].append(
                replay_steps_window / total_window if total_window > 0 else 0.0
            )
            replay_steps_window = 0
            fresh_steps_window = 0

            for name in TASK_NAMES:
                log["per_task_acc"][name].append(accs[name])
                cnt = max(1, running_per_task_count[name])
                log["per_task_loss"][name].append(running_per_task_loss[name] / cnt)
                log["buffer_task_frac"][name].append(tf[name])
                running_per_task_loss[name] = 0.0
                running_per_task_count[name] = 0

            acc_str = " ".join(f"{n[:4]}:{accs[n]:.2f}" for n in TASK_NAMES)
            tf_str = " ".join(f"{n[:4]}:{tf[n]:.2f}" for n in TASK_NAMES)
            print(
                f"step {step+1:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                f"{rate:.1f} st/s | acc [{acc_str}] | "
                f"buf {len(memory):4d} mean_p {mean_p:.3f} | buf_dist [{tf_str}]"
            )
            with open(log_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

    torch.save(
        {"model_state": model.state_dict(), "config": asdict(cfg), "final_log": log},
        log_dir / "final.pt",
    )

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Final per-task accuracy:")
    for name in TASK_NAMES:
        print(f"  {name:10s}: {log['per_task_acc'][name][-1]:.4f}")
    print(f"Final buffer size: {len(memory)}/{cfg.buffer_size}")
    print(f"Final buffer task dist: {memory.task_fractions()}")

    return log


# =============================================================================
# ANALYSIS
# =============================================================================

def summarize_run_memory(log, cfg: MemoryConfig):
    s = {
        "run_name": auto_name_memory(cfg),
        "config": asdict(cfg),
        "n_params": log["n_params"],
        "buffer_capacity": log["buffer_capacity"],
        "decay_factor": log["decay_factor"],
        "gate_fresh": log["gate_fresh"],
        "final_acc": {name: log["per_task_acc"][name][-1] for name in TASK_NAMES},
        "best_acc": {name: max(log["per_task_acc"][name]) for name in TASK_NAMES},
        "crossing_step_50": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.5)
            for name in TASK_NAMES
        },
        "crossing_step_90": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.9)
            for name in TASK_NAMES
        },
        "final_buffer_size": log["buffer_size"][-1],
        "final_buffer_task_frac": {
            name: log["buffer_task_frac"][name][-1] for name in TASK_NAMES
        },
        "final_mean_priority": log["mean_priority"][-1],
    }
    return s


def print_summary_table_memory(summaries):
    if not summaries:
        return
    print("\n" + "=" * 100)
    print("SWEEP SUMMARY (memory)")
    print("=" * 100)

    header = f"{'run':65s}  " + "  ".join(f"{n[:6]:>6s}" for n in TASK_NAMES)

    print("\nFinal accuracy:")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = "  ".join(f"{s['final_acc'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:65s}  {vals}")

    print("\nStep at which task first reached 50% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_50"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:65s}  " + "  ".join(vals))

    print("\nStep at which task first reached 90% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_90"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:65s}  " + "  ".join(vals))

    print("\nFinal buffer task distribution:")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = "  ".join(f"{s['final_buffer_task_frac'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:65s}  {vals}")

    print()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Four-task trainer with prioritized replay buffer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python memory.py
  python memory.py --seeds 0,1,2,3 --log_root runs/memory_seeds
  python memory.py --seeds 0,1,2 --sweep_buffer_size 250,1000,4000 --log_root runs/buffer
  python memory.py --gate_fresh --seeds 0,1,2,3 --log_root runs/memory_gated
""",
    )
    # Standard args.
    p.add_argument("--n_layers", type=int, default=MemoryConfig.n_layers)
    p.add_argument("--d_model", type=int, default=MemoryConfig.d_model)
    p.add_argument("--n_heads", type=int, default=MemoryConfig.n_heads)
    p.add_argument("--total_steps", type=int, default=MemoryConfig.total_steps)
    p.add_argument("--batch_size", type=int, default=MemoryConfig.batch_size)
    p.add_argument("--lr", type=float, default=MemoryConfig.lr)
    p.add_argument("--eval_every", type=int, default=MemoryConfig.eval_every)
    p.add_argument("--device", type=str, default=MemoryConfig.device, choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_dir", type=str, default="runs/memory")
    p.add_argument("--log_root", type=str, default="runs/memory_sweep")
    p.add_argument("--seed", type=int, default=MemoryConfig.seed)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--sweep_d_model", type=str, default=None)
    p.add_argument("--sweep_n_layers", type=str, default=None)
    p.add_argument("--sweep_lr", type=str, default=None)
    # Memory-specific args.
    p.add_argument("--buffer_size", type=int, default=MemoryConfig.buffer_size)
    p.add_argument("--decay_factor", type=float, default=MemoryConfig.decay_factor)
    p.add_argument("--gate_fresh", action="store_true",
                   help="Also gate correct tokens in fresh batches (stacks with replay).")
    p.add_argument("--sweep_buffer_size", type=str, default=None,
                   help="Comma-separated buffer_size values to sweep.")
    p.add_argument("--sweep_decay_factor", type=str, default=None,
                   help="Comma-separated decay_factor values to sweep.")
    return p.parse_args()


def build_configs(args):
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]
    d_models = _parse_int_list(args.sweep_d_model) if args.sweep_d_model else [args.d_model]
    n_layers_list = _parse_int_list(args.sweep_n_layers) if args.sweep_n_layers else [args.n_layers]
    lrs = [float(x) for x in args.sweep_lr.split(",")] if args.sweep_lr else [args.lr]
    buffer_sizes = (
        _parse_int_list(args.sweep_buffer_size)
        if args.sweep_buffer_size else [args.buffer_size]
    )
    decays = (
        [float(x) for x in args.sweep_decay_factor.split(",")]
        if args.sweep_decay_factor else [args.decay_factor]
    )

    is_sweep = (
        len(seeds) > 1
        or len(d_models) > 1
        or len(n_layers_list) > 1
        or len(lrs) > 1
        or len(buffer_sizes) > 1
        or len(decays) > 1
    )

    configs = []
    for n_layers in n_layers_list:
        for d_model in d_models:
            for lr in lrs:
                for bs in buffer_sizes:
                    for dec in decays:
                        for seed in seeds:
                            cfg = MemoryConfig(
                                n_layers=n_layers,
                                d_model=d_model,
                                n_heads=args.n_heads,
                                total_steps=args.total_steps,
                                batch_size=args.batch_size,
                                lr=lr,
                                eval_every=args.eval_every,
                                device=args.device,
                                seed=seed,
                                buffer_size=bs,
                                decay_factor=dec,
                                gate_fresh=args.gate_fresh,
                            )
                            if is_sweep:
                                cfg.log_dir = str(Path(args.log_root) / auto_name_memory(cfg))
                            else:
                                cfg.log_dir = args.log_dir
                            configs.append(cfg)

    for cfg in configs:
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"n_heads={cfg.n_heads} must divide d_model={cfg.d_model}.")

    return configs, is_sweep


def main():
    args = parse_args()
    configs, is_sweep = build_configs(args)

    if is_sweep:
        log_root = Path(args.log_root)
        log_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep mode: {len(configs)} runs under {log_root}")
        for i, cfg in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {auto_name_memory(cfg)}")
        print()

        summaries = []
        t0 = time.time()
        for i, cfg in enumerate(configs):
            print("=" * 100)
            print(f"RUN {i+1}/{len(configs)}: {auto_name_memory(cfg)}")
            print("=" * 100)
            log = train_memory(cfg)
            summaries.append(summarize_run_memory(log, cfg))
            with open(log_root / "summary.json", "w") as f:
                json.dump(summaries, f, indent=2)

        elapsed = time.time() - t0
        print(f"\nSweep complete: {len(configs)} runs in {elapsed/60:.1f} min "
              f"(avg {elapsed/len(configs)/60:.1f} min/run)")
        print_summary_table_memory(summaries)
        print(f"\nPer-run logs:      {log_root}/<run_name>/log.json")
        print(f"Aggregate summary: {log_root}/summary.json")
    else:
        cfg = configs[0]
        log = train_memory(cfg)
        summary = summarize_run_memory(log, cfg)
        with open(Path(cfg.log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
