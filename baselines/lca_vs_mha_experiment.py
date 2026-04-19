"""
Sparsification-vs-multihead experiment on TinyStories.

Compares three attention variants on character-level language modeling.

Conditions:
    mha      : Multi-head attention (H=4). Baseline.
    sha      : Single-head attention. No sparsification.
    sha_lca  : Single-head, LCA on both Q and K.

LCA formulation (Option 2): Q = xW_Q is computed densely, then LCA runs
as a sparsifying nonlinearity on the result (same for K). Dynamics:
    v_{t+1} = v_t + eta * (u - v_t - a_t @ G)
    a_t     = sign(v_t) * relu(|v_t| - lambda)
G is a learnable lateral-inhibition matrix (symmetric, zero diagonal).
lambda is learnable via log parameterization.

Task: character-level next-token prediction on TinyStories. Each run trains
until val loss plateaus. 3 seeds per condition by default.

Metrics reported: val loss (nats/char), bits per char (bpc), attention
entropy, stable rank of attention, and LCA sparsity (fraction of zero
entries in Q/K after LCA).

Requires: `pip install datasets` for loading TinyStories.

Usage:
    python lca_vs_mha_experiment.py
    python lca_vs_mha_experiment.py --quick
    python lca_vs_mha_experiment.py --condition sha_lca --seeds 0
    python lca_vs_mha_experiment.py --cpu
"""

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Device
# ============================================================

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Data: TinyStories + character-level tokenizer
# ============================================================

def load_tinystories(n_train_stories, n_val_stories):
    """Stream TinyStories from HuggingFace and keep first N stories."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The `datasets` library is required. Install with: pip install datasets"
        )

    print(f"Loading TinyStories (streaming): "
          f"{n_train_stories} train, {n_val_stories} val stories")
    train_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_stream = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    train_texts = []
    for i, ex in enumerate(train_stream):
        if i >= n_train_stories:
            break
        train_texts.append(ex["text"])

    val_texts = []
    for i, ex in enumerate(val_stream):
        if i >= n_val_stories:
            break
        val_texts.append(ex["text"])

    print(f"  loaded {len(train_texts)} train, {len(val_texts)} val stories")
    return train_texts, val_texts


class CharTokenizer:
    """Simple character-level tokenizer. 0 = PAD, 1 = UNK."""
    def __init__(self, texts):
        chars = set()
        for t in texts:
            chars.update(t)
        self.chars = sorted(chars)
        self.PAD = 0
        self.UNK = 1
        self.stoi = {c: i + 2 for i, c in enumerate(self.chars)}
        self.itos = {i + 2: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 2

    def encode(self, text):
        return [self.stoi.get(c, self.UNK) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "?") for i in ids)


def tokenize_and_concat(texts, tokenizer, sep="\n"):
    sep_ids = tokenizer.encode(sep)
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t))
        all_ids.extend(sep_ids)
    return np.array(all_ids, dtype=np.int64)


class LMDataset(Dataset):
    """
    Samples fixed-length windows from a token stream.

    Train mode (random=True): each __getitem__ returns a random window.
    Eval mode (random=False): non-overlapping consecutive chunks (deterministic).
    """
    def __init__(self, tokens, seq_len, random=True, n_samples=None, seed=0):
        self.tokens = tokens
        self.seq_len = seq_len
        self.random = random
        self._gen = torch.Generator().manual_seed(seed)
        if random:
            self.n_samples = n_samples if n_samples is not None else max(1, len(tokens) // seq_len)
        else:
            self.n_samples = max(1, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.random:
            hi = len(self.tokens) - self.seq_len - 1
            start = int(torch.randint(0, max(hi, 1), (1,), generator=self._gen).item())
        else:
            start = idx * self.seq_len
        x = torch.from_numpy(self.tokens[start:start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1:start + self.seq_len + 1].astype(np.int64))
        return x, y


# ============================================================
# LCA sparsifier (Option 2)
# ============================================================

class LCASparsifier(nn.Module):
    def __init__(self, dim, n_iters=10, lam_init=0.5, eta=0.1):
        super().__init__()
        self.dim = dim
        self.n_iters = n_iters
        self.eta = eta
        self.log_lam = nn.Parameter(torch.log(torch.tensor(float(lam_init))))
        self.G_raw = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def get_G(self):
        G = 0.5 * (self.G_raw + self.G_raw.T)
        G = G - torch.diag_embed(torch.diagonal(G))
        return G

    def forward(self, u, return_stats=False):
        lam = torch.exp(self.log_lam)
        G = self.get_G()
        v = torch.zeros_like(u)
        a = torch.zeros_like(u)
        for _ in range(self.n_iters):
            dv = u - v - a @ G
            v = v + self.eta * dv
            a = torch.sign(v) * F.relu(torch.abs(v) - lam)
        if return_stats:
            stats = {
                "sparsity": (a.abs() < 1e-6).float().mean().item(),
                "lambda": lam.item(),
            }
            return a, stats
        return a


# ============================================================
# Attention
# ============================================================

def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_attn=False):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        out = self.out(out)
        if return_attn:
            return out, attn
        return out


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, lca_q=False, lca_k=False,
                 lca_iters=10, lca_lam=0.5):
        super().__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.lca_q = LCASparsifier(d_model, lca_iters, lca_lam) if lca_q else None
        self.lca_k = LCASparsifier(d_model, lca_iters, lca_lam) if lca_k else None

    def forward(self, x, return_attn=False, return_stats=False):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        stats = {}
        if self.lca_q is not None:
            if return_stats:
                q, sq = self.lca_q(q, return_stats=True)
                stats["q"] = sq
            else:
                q = self.lca_q(q)
        if self.lca_k is not None:
            if return_stats:
                k, sk = self.lca_k(k, return_stats=True)
                stats["k"] = sk
            else:
                k = self.lca_k(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores.masked_fill(causal_mask(T, x.device), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = self.out(attn @ v)

        outputs = [out]
        if return_attn:
            outputs.append(attn)
        if return_stats:
            outputs.append(stats)
        return tuple(outputs) if len(outputs) > 1 else out


# ============================================================
# Model
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, attn, mlp_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = attn
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, max_len, attn_factory):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attn_factory()) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def build_model(condition, vocab_size, d_model, n_layers, max_len,
                n_heads=4, lca_iters=10, lca_lam=0.5):
    def attn_factory():
        if condition == "mha":
            return MultiHeadAttention(d_model, n_heads=n_heads)
        if condition == "sha":
            return SingleHeadAttention(d_model)
        if condition == "sha_lca":
            return SingleHeadAttention(d_model, lca_q=True, lca_k=True,
                                       lca_iters=lca_iters, lca_lam=lca_lam)
        raise ValueError(f"Unknown condition: {condition}")

    return TinyTransformer(vocab_size, d_model, n_layers, max_len, attn_factory)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Train / Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """Return mean val loss in nats/char."""
    model.eval()
    loss_sum, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        loss_sum += loss.item()
        count += y.numel()
    model.train()
    return loss_sum / max(count, 1)


@torch.no_grad()
def diagnostics(model, loader, device, n_batches=3):
    model.eval()
    ents, srs, sp_q, sp_k = [], [], [], []

    for bi, (x, _) in enumerate(loader):
        if bi >= n_batches:
            break
        x = x.to(device)
        B, T = x.shape
        pos = torch.arange(T, device=device)
        h = model.tok_emb(x) + model.pos_emb(pos)[None]

        for block in model.blocks:
            ln_x = block.ln1(h)
            attn_mod = block.attn

            if isinstance(attn_mod, MultiHeadAttention):
                out, attn = attn_mod(ln_x, return_attn=True)
            else:
                has_lca = attn_mod.lca_q is not None or attn_mod.lca_k is not None
                if has_lca:
                    out, attn, stats = attn_mod(ln_x, return_attn=True, return_stats=True)
                    if "q" in stats:
                        sp_q.append(stats["q"]["sparsity"])
                    if "k" in stats:
                        sp_k.append(stats["k"]["sparsity"])
                else:
                    out, attn = attn_mod(ln_x, return_attn=True)

            clamped = attn.clamp_min(1e-12)
            ent = -(clamped * clamped.log()).sum(-1).mean().item()
            ents.append(ent)

            # Stable rank: matrix_norm(ord=2) on MPS is flaky, do on CPU.
            A = attn.reshape(-1, T, T).detach().cpu()
            fro2 = (A ** 2).sum(dim=(-2, -1))
            specs = torch.linalg.matrix_norm(A, ord=2)
            sr = (fro2 / (specs ** 2).clamp_min(1e-12)).mean().item()
            srs.append(sr)

            h = h + out
            h = h + block.mlp(block.ln2(h))

    model.train()

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "attn_entropy": mean(ents),
        "attn_stable_rank": mean(srs),
        "sparsity_q": mean(sp_q),
        "sparsity_k": mean(sp_k),
    }


def train_one(condition, config, tokenizer, train_tokens, val_tokens, device, seed=0):
    """Train one model until val loss plateaus."""
    torch.manual_seed(seed)

    d_model = config["d_model"]
    n_layers = config["n_layers"]
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    max_steps = config["max_steps"]
    eval_interval = config["eval_interval"]
    patience = config["patience"]
    min_delta = config["min_delta"]
    warmup_steps = config["warmup_steps"]
    lr = config["lr"]

    train_ds = LMDataset(train_tokens, seq_len, random=True,
                         n_samples=10000, seed=seed)
    val_ds = LMDataset(val_tokens, seq_len, random=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        condition, tokenizer.vocab_size, d_model, n_layers,
        max_len=seq_len,
        n_heads=config["n_heads"],
        lca_iters=config["lca_iters"],
        lca_lam=config["lca_lam"],
    ).to(device)
    n_params = count_params(model)
    print(f"  condition={condition}, seed={seed}, params={n_params}, "
          f"vocab={tokenizer.vocab_size}, seq_len={seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    log = []
    t0 = time.time()

    best_val_loss = float("inf")
    best_step = 0
    evals_without_improvement = 0
    stop_reason = "max_steps"

    train_iter = iter(train_loader)
    step = 0
    while step < max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_interval == 0 or step == max_steps - 1:
            vl = evaluate(model, val_loader, device)
            bpc = vl / math.log(2)
            log.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": vl,
                "val_bpc": bpc,
            })
            improved = vl < best_val_loss - min_delta
            if improved:
                best_val_loss = vl
                best_step = step
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            marker = "*" if improved else " "
            print(f"    step {step:6d} {marker} train_loss={loss.item():.4f}  "
                  f"val_loss={vl:.4f}  bpc={bpc:.4f}  "
                  f"stale={evals_without_improvement}/{patience}")
            if evals_without_improvement >= patience:
                stop_reason = "plateau"
                break
        step += 1

    elapsed = time.time() - t0
    final_val_loss = evaluate(model, val_loader, device)
    diags = diagnostics(model, val_loader, device)
    return {
        "condition": condition,
        "seed": seed,
        "n_params": n_params,
        "final_val_loss": final_val_loss,
        "final_bpc": final_val_loss / math.log(2),
        "best_val_loss": best_val_loss,
        "best_bpc": best_val_loss / math.log(2),
        "best_step": best_step,
        "steps_trained": step + 1,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "log": log,
        "diagnostics": diags,
    }


# ============================================================
# Main
# ============================================================

DEFAULT_CONFIG = {
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "seq_len": 256,
    "batch_size": 32,
    "lr": 3e-4,
    "warmup_steps": 200,
    "max_steps": 25000,
    "eval_interval": 300,
    "patience": 60,
    "min_delta": 0.005,
    "lca_iters": 10,
    "lca_lam": 0.5,
    "n_train_stories": 20000,
    "n_val_stories": 500,
}

CONDITIONS = ["mha", "sha", "sha_lca"]
SEEDS = [0, 1]


def aggregate(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[r["condition"]].append(r)

    out = []
    for cond, rs in sorted(groups.items()):
        def stat(key, source=None):
            if source is None:
                vals = [r[key] for r in rs if r.get(key) is not None]
            else:
                vals = [r[source].get(key) for r in rs if r[source].get(key) is not None]
            if not vals:
                return None, None
            arr = np.array(vals, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0))

        mean_loss, std_loss = stat("best_val_loss")
        mean_bpc, std_bpc = stat("best_bpc")
        mean_steps, std_steps = stat("steps_trained")
        mean_ent, _ = stat("attn_entropy", source="diagnostics")
        mean_sr, _ = stat("attn_stable_rank", source="diagnostics")
        mean_spq, _ = stat("sparsity_q", source="diagnostics")
        mean_spk, _ = stat("sparsity_k", source="diagnostics")

        out.append({
            "condition": cond,
            "n_seeds": len(rs),
            "mean_val_loss": mean_loss,
            "std_val_loss": std_loss,
            "mean_bpc": mean_bpc,
            "std_bpc": std_bpc,
            "mean_steps": mean_steps,
            "std_steps": std_steps,
            "mean_attn_entropy": mean_ent,
            "mean_stable_rank": mean_sr,
            "mean_sparsity_q": mean_spq,
            "mean_sparsity_k": mean_spk,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Smaller model, fewer stories, shorter run.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="results_tinystories.json")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.quick:
        config["d_model"] = 64
        config["n_layers"] = 2
        config["seq_len"] = 128
        config["max_steps"] = 2000
        config["eval_interval"] = 100
        config["patience"] = 5
        config["warmup_steps"] = 100
        config["n_train_stories"] = 3000
        config["n_val_stories"] = 200

    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Load and tokenize once, reuse across all runs.
    train_texts, val_texts = load_tinystories(
        config["n_train_stories"], config["n_val_stories"]
    )
    print("Building tokenizer from training text")
    tokenizer = CharTokenizer(train_texts)
    print(f"  vocab_size={tokenizer.vocab_size}")

    print("Tokenizing")
    train_tokens = tokenize_and_concat(train_texts, tokenizer)
    val_tokens = tokenize_and_concat(val_texts, tokenizer)
    print(f"  train tokens: {len(train_tokens):,}, val tokens: {len(val_tokens):,}")

    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    seeds = args.seeds if args.seeds is not None else ([0] if args.quick else SEEDS)

    total_runs = len(conditions) * len(seeds)
    print(f"\nPlanned: {len(conditions)} conditions x {len(seeds)} seeds "
          f"= {total_runs} runs")

    runs = []
    run_idx = 0
    t_all = time.time()
    for cond in conditions:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] cond={cond} seed={seed}")
            res = train_one(cond, config, tokenizer, train_tokens, val_tokens,
                            device, seed=seed)
            runs.append(res)
            agg = aggregate(runs)
            with open(args.output, "w") as f:
                json.dump({
                    "config": config,
                    "vocab_size": tokenizer.vocab_size,
                    "runs": runs,
                    "aggregated": agg,
                }, f, indent=2)

    total_elapsed = time.time() - t_all
    agg = aggregate(runs)

    print("\n===== Aggregated Summary (mean +/- std over seeds) =====")
    header = (f"{'condition':<10} {'seeds':>5} "
              f"{'val_loss':>16} {'bpc':>16} {'steps':>12} "
              f"{'ent':>6} {'sr':>6} {'sp_q':>6} {'sp_k':>6}")
    print(header)

    def fmt(v, w=6, p=3):
        return f"{v:.{p}f}".rjust(w) if v is not None else "-".rjust(w)

    def fmt_mean_std(m, s, w=16, p=3):
        if m is None:
            return "-".rjust(w)
        return f"{m:.{p}f}+/-{s:.{p}f}".rjust(w)

    for a in agg:
        print(f"{a['condition']:<10} {a['n_seeds']:>5} "
              f"{fmt_mean_std(a['mean_val_loss'], a['std_val_loss'])} "
              f"{fmt_mean_std(a['mean_bpc'], a['std_bpc'])} "
              f"{fmt_mean_std(a['mean_steps'], a['std_steps'], p=0)} "
              f"{fmt(a['mean_attn_entropy'])} {fmt(a['mean_stable_rank'])} "
              f"{fmt(a['mean_sparsity_q'])} {fmt(a['mean_sparsity_k'])}")

    print(f"\nTotal wall-clock: {total_elapsed/60:.1f} min")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
