"""
Four-task baseline for parameter golf research.

Tasks:
    1. 3-digit addition (natural order output)
    2. Sorting (length 5, ascending)
    3. Reversal (length 5)
    4. Modular addition mod 97

Goal: establish a baseline where the tiny transformer shows the interference
pattern we're hypothesizing about. Easy tasks solved fast, hard tasks plateau
or struggle. No intervention yet.

Run: python baseline.py
Adjust CONFIG block at top to sweep model sizes or task mixtures.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # model
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.0

    # training
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 20000
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # eval
    eval_every: int = 250
    eval_batch_size: int = 256
    n_eval_problems: int = 512  # per task

    # tasks
    task_mix: tuple = (0.25, 0.25, 0.25, 0.25)  # addition, sort, reverse, modadd
    mod_p: int = 97
    seq_len_sort: int = 5
    seq_len_reverse: int = 5
    add_digits: int = 3

    # misc
    seed: int = 0
    device: str = "cpu"  # "cpu", "mps", or "cuda"
    log_dir: str = "runs/baseline"
    compile_model: bool = False


# =============================================================================
# TOKENIZER
# =============================================================================
# Shared vocabulary across all four tasks.
# Digits 0-9, task markers, separators, and padding.

DIGITS = [str(i) for i in range(10)]
SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<SEP>",
           "<ADD>", "<SORT>", "<REV>", "<MOD>",
           "+", "=", ","]

VOCAB = SPECIAL + DIGITS
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
VOCAB_SIZE = len(VOCAB)

PAD_ID = TOK2ID["<PAD>"]
BOS_ID = TOK2ID["<BOS>"]
EOS_ID = TOK2ID["<EOS>"]
SEP_ID = TOK2ID["<SEP>"]


def encode(tokens):
    return [TOK2ID[t] for t in tokens]


def decode(ids):
    return [ID2TOK[int(i)] for i in ids]


# =============================================================================
# TASK GENERATORS
# =============================================================================
# Each task returns (input_ids, target_ids, answer_start_idx).
# The model is trained to predict target_ids shifted by one (standard causal LM).
# answer_start_idx tells us where the actual answer begins, which is what we
# measure accuracy on (so we don't count the model getting "=" right as signal).

def make_addition(rng, n_digits=3):
    """3-digit addition, natural order (most significant digit first).
    Format: <BOS> <ADD> a2 a1 a0 + b2 b1 b0 = c3 c2 c1 c0 <EOS>
    Output length is fixed at n_digits+1 to cover carry.
    """
    max_val = 10 ** n_digits - 1
    a = rng.randint(0, max_val)
    b = rng.randint(0, max_val)
    c = a + b

    a_digits = list(str(a).zfill(n_digits))
    b_digits = list(str(b).zfill(n_digits))
    c_digits = list(str(c).zfill(n_digits + 1))

    tokens = ["<BOS>", "<ADD>"] + a_digits + ["+"] + b_digits + ["="] + c_digits + ["<EOS>"]
    ids = encode(tokens)

    # Answer starts right after the "=" token.
    answer_start = tokens.index("=") + 1
    return ids, answer_start


def make_sort(rng, length=5):
    """Sort digits ascending.
    Format: <BOS> <SORT> d1 d2 ... dn = s1 s2 ... sn <EOS>
    """
    digits = [rng.randint(0, 9) for _ in range(length)]
    sorted_digits = sorted(digits)

    tokens = ["<BOS>", "<SORT>"] + [str(d) for d in digits] + ["="] + [str(d) for d in sorted_digits] + ["<EOS>"]
    ids = encode(tokens)
    answer_start = tokens.index("=") + 1
    return ids, answer_start


def make_reverse(rng, length=5):
    """Reverse digits.
    Format: <BOS> <REV> d1 d2 ... dn = dn ... d2 d1 <EOS>
    """
    digits = [rng.randint(0, 9) for _ in range(length)]
    reversed_digits = list(reversed(digits))

    tokens = ["<BOS>", "<REV>"] + [str(d) for d in digits] + ["="] + [str(d) for d in reversed_digits] + ["<EOS>"]
    ids = encode(tokens)
    answer_start = tokens.index("=") + 1
    return ids, answer_start


def make_modadd(rng, p=97):
    """Modular addition mod p. Classic grokking setup.
    Format: <BOS> <MOD> a_digits + b_digits = c_digits <EOS>
    We render the numbers as digit sequences so they share vocabulary with
    the other tasks. p=97 means operands are 0-96, two digits each.
    """
    a = rng.randint(0, p - 1)
    b = rng.randint(0, p - 1)
    c = (a + b) % p

    n_d = len(str(p - 1))  # 2 digits for p=97
    a_digits = list(str(a).zfill(n_d))
    b_digits = list(str(b).zfill(n_d))
    c_digits = list(str(c).zfill(n_d))

    tokens = ["<BOS>", "<MOD>"] + a_digits + ["+"] + b_digits + ["="] + c_digits + ["<EOS>"]
    ids = encode(tokens)
    answer_start = tokens.index("=") + 1
    return ids, answer_start


TASK_NAMES = ["addition", "sort", "reverse", "modadd"]
TASK_FNS = {
    "addition": make_addition,
    "sort": make_sort,
    "reverse": make_reverse,
    "modadd": make_modadd,
}


def sample_task(rng, task_name, cfg):
    if task_name == "addition":
        return make_addition(rng, n_digits=cfg.add_digits)
    elif task_name == "sort":
        return make_sort(rng, length=cfg.seq_len_sort)
    elif task_name == "reverse":
        return make_reverse(rng, length=cfg.seq_len_reverse)
    elif task_name == "modadd":
        return make_modadd(rng, p=cfg.mod_p)
    else:
        raise ValueError(task_name)


# =============================================================================
# DATASET / BATCHING
# =============================================================================

def make_batch(rng, cfg, task_mix=None):
    """Sample a batch of mixed-task sequences, pad to max length in batch."""
    if task_mix is None:
        task_mix = cfg.task_mix

    sequences = []
    answer_starts = []
    task_ids = []

    for _ in range(cfg.batch_size):
        task_idx = rng.choices(range(4), weights=task_mix, k=1)[0]
        task_name = TASK_NAMES[task_idx]
        ids, ans_start = sample_task(rng, task_name, cfg)
        sequences.append(ids)
        answer_starts.append(ans_start)
        task_ids.append(task_idx)

    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((cfg.batch_size, max_len), PAD_ID, dtype=torch.long)
    # We want to compute loss only on answer tokens. Make a mask.
    loss_mask = torch.zeros((cfg.batch_size, max_len), dtype=torch.float)

    for i, (seq, ans_start) in enumerate(zip(sequences, answer_starts)):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        # Predict positions [ans_start, ..., len(seq)-1] from positions
        # [ans_start-1, ..., len(seq)-2] in standard shifted-target fashion.
        # Mask in target-index space: positions ans_start..len(seq)-1.
        loss_mask[i, ans_start:len(seq)] = 1.0

    return input_ids, loss_mask, torch.tensor(task_ids, dtype=torch.long), answer_starts


# =============================================================================
# MODEL
# =============================================================================
# Minimal GPT-style decoder. No fancy tricks. This is the control.

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # B, H, T, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_ff = cfg.d_model * cfg.d_ff_mult
        self.fc1 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config, max_len=64):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_emb = nn.Embedding(max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Tie output to input embedding, saves params in tight-capacity regime.
        self.register_buffer("_pos_ids", torch.arange(max_len).unsqueeze(0), persistent=False)
        self.apply(self._init_weights)
        # Scale residual-path projections per GPT-2 convention.
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, T = ids.shape
        pos = self._pos_ids[:, :T]
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # Weight-tied unembed.
        logits = x @ self.tok_emb.weight.T
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# EVAL
# =============================================================================

@torch.no_grad()
def evaluate(model, cfg, device, seed_offset=10_000_000):
    """Exact-match accuracy per task on freshly-sampled problems.
    We use a fixed seed offset so eval problems are consistent across steps.
    """
    model.eval()
    results = {}
    for task_name in TASK_NAMES:
        rng = random.Random(seed_offset + hash(task_name) % 1_000_000)
        n_correct = 0
        n_total = 0
        # Chunk into batches.
        for chunk_start in range(0, cfg.n_eval_problems, cfg.eval_batch_size):
            chunk_size = min(cfg.eval_batch_size, cfg.n_eval_problems - chunk_start)
            seqs, ans_starts = [], []
            for _ in range(chunk_size):
                ids, ans_start = sample_task(rng, task_name, cfg)
                seqs.append(ids)
                ans_starts.append(ans_start)

            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((chunk_size, max_len), PAD_ID, dtype=torch.long, device=device)
            for i, s in enumerate(seqs):
                input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

            logits = model(input_ids)  # B, T, V
            preds = logits.argmax(dim=-1)  # B, T

            # For each sequence, check that predicted tokens at positions
            # [ans_start, ..., len(seq)-1] match target tokens at those positions.
            # Predicted token at position t comes from logits at position t-1.
            for i, (s, ans_start) in enumerate(zip(seqs, ans_starts)):
                target = torch.tensor(s[ans_start:], device=device)
                pred = preds[i, ans_start - 1:len(s) - 1]
                if torch.equal(pred, target):
                    n_correct += 1
                n_total += 1

        results[task_name] = n_correct / n_total
    model.train()
    return results


# =============================================================================
# TRAINING
# =============================================================================

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    # Cosine decay to 10% of peak.
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # Figure out a safe max_len by generating a few samples.
    probe_rng = random.Random(0)
    max_len = 0
    for _ in range(200):
        for name in TASK_NAMES:
            ids, _ = sample_task(probe_rng, name, cfg)
            max_len = max(max_len, len(ids))
    max_len = max_len + 4  # small safety margin
    print(f"max_len: {max_len}")

    model = TinyGPT(cfg, max_len=max_len).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                   betas=(0.9, 0.95))

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility.
    with open(log_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    log = {
        "step": [],
        "loss": [],
        "per_task_acc": {name: [] for name in TASK_NAMES},
        "per_task_loss": {name: [] for name in TASK_NAMES},
        "n_params": n_params,
    }

    rng = random.Random(cfg.seed + 1)
    model.train()
    t_start = time.time()
    running_loss = 0.0
    running_per_task_loss = {name: 0.0 for name in TASK_NAMES}
    running_per_task_count = {name: 0 for name in TASK_NAMES}

    for step in range(cfg.total_steps):
        input_ids, loss_mask, task_ids, _ = make_batch(rng, cfg)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        # Shift for causal LM: predict ids[:, 1:] from logits[:, :-1].
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]  # align with targets

        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        masked = loss_per_tok * mask
        total_masked = mask.sum().clamp(min=1.0)
        loss = masked.sum() / total_masked

        # Per-task loss tracking (for logging).
        with torch.no_grad():
            for t_idx, name in enumerate(TASK_NAMES):
                task_mask = (task_ids == t_idx).float().unsqueeze(-1) * mask
                tm_sum = task_mask.sum()
                if tm_sum > 0:
                    tl = (loss_per_tok * task_mask).sum() / tm_sum
                    running_per_task_loss[name] += tl.item()
                    running_per_task_count[name] += 1

        lr_now = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % cfg.eval_every == 0 or step == 0:
            avg_loss = running_loss / max(1, cfg.eval_every if step > 0 else 1)
            running_loss = 0.0

            accs = evaluate(model, cfg, device)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed

            log["step"].append(step + 1)
            log["loss"].append(avg_loss)
            for name in TASK_NAMES:
                log["per_task_acc"][name].append(accs[name])
                if running_per_task_count[name] > 0:
                    ptl = running_per_task_loss[name] / running_per_task_count[name]
                else:
                    ptl = float("nan")
                log["per_task_loss"][name].append(ptl)
                running_per_task_loss[name] = 0.0
                running_per_task_count[name] = 0

            acc_str = " ".join(f"{name[:4]}:{accs[name]:.2f}" for name in TASK_NAMES)
            print(f"step {step+1:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                  f"{rate:.1f} steps/s | {acc_str}")

            # Checkpoint log.
            with open(log_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

    # Save final model.
    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "final_log": log,
    }, log_dir / "final.pt")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Final per-task accuracy:")
    for name in TASK_NAMES:
        print(f"  {name:10s}: {log['per_task_acc'][name][-1]:.4f}")

    return log


# =============================================================================
# SWEEP / ANALYSIS HELPERS
# =============================================================================

def auto_name(cfg: Config) -> str:
    """Generate a stable directory name from a config."""
    return f"d{cfg.d_model}_L{cfg.n_layers}_h{cfg.n_heads}_bs{cfg.batch_size}_lr{cfg.lr:g}_seed{cfg.seed}"


def crossing_step(steps, accs, threshold):
    """Return the first step at which accuracy crosses `threshold`.
    Returns None if never crossed. Useful for locating grokking transitions.
    """
    for s, a in zip(steps, accs):
        if a >= threshold:
            return s
    return None


def summarize_run(log, cfg: Config):
    """Compute a compact summary of a single run. Designed for cross-run comparison."""
    summary = {
        "run_name": auto_name(cfg),
        "config": asdict(cfg),
        "n_params": log["n_params"],
        "final_acc": {name: log["per_task_acc"][name][-1] for name in TASK_NAMES},
        "best_acc": {name: max(log["per_task_acc"][name]) for name in TASK_NAMES},
        # Step at which each task first reaches a given accuracy. None if never.
        "crossing_step_50": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.5)
            for name in TASK_NAMES
        },
        "crossing_step_90": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.9)
            for name in TASK_NAMES
        },
    }
    return summary


def print_summary_table(summaries):
    """Pretty-print a cross-run summary to stdout."""
    if not summaries:
        return
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)

    # Final accuracy table
    header = f"{'run':40s}  " + "  ".join(f"{n[:6]:>6s}" for n in TASK_NAMES)
    print("\nFinal accuracy:")
    print(header)
    print("-" * len(header))
    for s in summaries:
        accs = "  ".join(f"{s['final_acc'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:40s}  {accs}")

    # Step-to-50% table (grokking step proxy)
    print("\nStep at which task first reached 50% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_50"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:40s}  " + "  ".join(vals))

    # Step-to-90% table
    print("\nStep at which task first reached 90% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_90"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:40s}  " + "  ".join(vals))
    print()


# =============================================================================
# CLI
# =============================================================================

def _parse_int_list(s):
    """Parse '0,1,2' or '32,48,64' into [0, 1, 2] / [32, 48, 64]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Four-task baseline trainer with built-in sweep support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run (original behavior)
  python baseline.py

  # Run 5 seeds of the default config
  python baseline.py --seeds 0,1,2,3,4 --log_root runs/default_seeds

  # Sweep model width across 3 seeds (3 x 3 = 9 runs)
  python baseline.py --seeds 0,1,2 --sweep_d_model 32,48,64 --log_root runs/width

  # Sweep depth x width x seed
  python baseline.py --seeds 0,1 --sweep_d_model 32,64 --sweep_n_layers 1,2,3 \\
      --log_root runs/capacity

Notes:
  - n_heads must divide d_model. If you sweep d_model, either pick values that
    divide cleanly by n_heads (default 4) or also pass --n_heads.
  - Each run writes to {log_root}/{auto_name}/. A summary.json aggregating
    all runs is written to {log_root}/summary.json and updated after each run.
""",
    )
    # Model
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    # Training
    p.add_argument("--total_steps", type=int, default=Config.total_steps)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--eval_every", type=int, default=Config.eval_every)
    # Infra
    p.add_argument("--device", type=str, default=Config.device, choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_dir", type=str, default=Config.log_dir,
                   help="Output dir for single runs. Ignored in sweep mode.")
    p.add_argument("--log_root", type=str, default="runs/sweep",
                   help="Parent dir for sweep runs. Used when --seeds has >1 value or any --sweep_* is set.")
    p.add_argument("--seed", type=int, default=Config.seed, help="Single-run seed.")
    # Sweep args
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds, e.g. '0,1,2,3,4'. Overrides --seed.")
    p.add_argument("--sweep_d_model", type=str, default=None,
                   help="Comma-separated d_model values to sweep over.")
    p.add_argument("--sweep_n_layers", type=str, default=None,
                   help="Comma-separated n_layers values to sweep over.")
    p.add_argument("--sweep_lr", type=str, default=None,
                   help="Comma-separated learning rate values to sweep over.")
    return p.parse_args()


def build_configs(args):
    """Expand CLI args into a list of Config objects to run.
    Returns (configs, is_sweep). In sweep mode the log_dir is auto-set per run.
    """
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]
    d_models = _parse_int_list(args.sweep_d_model) if args.sweep_d_model else [args.d_model]
    n_layers_list = _parse_int_list(args.sweep_n_layers) if args.sweep_n_layers else [args.n_layers]
    lrs = [float(x) for x in args.sweep_lr.split(",")] if args.sweep_lr else [args.lr]

    is_sweep = (
        len(seeds) > 1
        or len(d_models) > 1
        or len(n_layers_list) > 1
        or len(lrs) > 1
    )

    configs = []
    # Outer loop over architecture, inner over seeds, so seeds for a given
    # architecture run contiguously. Makes incremental inspection easier.
    for n_layers in n_layers_list:
        for d_model in d_models:
            for lr in lrs:
                for seed in seeds:
                    cfg = Config(
                        n_layers=n_layers,
                        d_model=d_model,
                        n_heads=args.n_heads,
                        total_steps=args.total_steps,
                        batch_size=args.batch_size,
                        lr=lr,
                        eval_every=args.eval_every,
                        device=args.device,
                        seed=seed,
                    )
                    if is_sweep:
                        cfg.log_dir = str(Path(args.log_root) / auto_name(cfg))
                    else:
                        cfg.log_dir = args.log_dir
                    configs.append(cfg)

    # Validate n_heads divides d_model for every config before starting.
    for cfg in configs:
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"n_heads={cfg.n_heads} must divide d_model={cfg.d_model}. "
                f"Pass a compatible --n_heads."
            )

    return configs, is_sweep


def main():
    args = parse_args()
    configs, is_sweep = build_configs(args)

    if is_sweep:
        log_root = Path(args.log_root)
        log_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep mode: {len(configs)} runs will be written under {log_root}")
        for i, cfg in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {auto_name(cfg)}")
        print()

        summaries = []
        t_sweep_start = time.time()
        for i, cfg in enumerate(configs):
            print("=" * 80)
            print(f"RUN {i+1}/{len(configs)}: {auto_name(cfg)}")
            print("=" * 80)
            log = train(cfg)
            summaries.append(summarize_run(log, cfg))

            # Incremental summary so a crashed/Ctrl-C'd sweep still yields data.
            with open(log_root / "summary.json", "w") as f:
                json.dump(summaries, f, indent=2)

        elapsed = time.time() - t_sweep_start
        print(f"\nSweep complete: {len(configs)} runs in {elapsed/60:.1f} min "
              f"(avg {elapsed/len(configs)/60:.1f} min/run)")
        print_summary_table(summaries)
        print(f"\nPer-run logs:     {log_root}/<run_name>/log.json")
        print(f"Aggregate summary: {log_root}/summary.json")
    else:
        # Single-run path, unchanged.
        cfg = configs[0]
        log = train(cfg)
        # Also write a single-run summary for consistency.
        summary = summarize_run(log, cfg)
        with open(Path(cfg.log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
