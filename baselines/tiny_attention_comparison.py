"""
tiny_attention_comparison.py

Compare three attention mechanisms at ~10-13M parameter scale:
  1. Standard softmax attention        - O(T^2) per layer, full recall
  2. Linear attention (ELU+1 kernel)   - O(T) per layer, running state
  3. Gated DeltaNet linear attention   - O(T) per layer, delta-rule state with gating

Runs on a single GPU. Expected total runtime: 5-15 minutes depending on hardware.
Will also run on CPU but much more slowly.

Outputs:
  results/training_curves.png       loss vs training step for each model
  results/inference_latency.png     latency vs context length for each model
  results/results.json              raw numbers

Dependencies:
  pip install torch tiktoken datasets matplotlib
"""

import math
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import tiktoken
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # Model
    vocab_size: int = 50257       # GPT-2 tokenizer
    dim: int = 256
    n_heads: int = 8              # head_dim = 32, keeps linear attention memory modest
    n_layers: int = 4
    mlp_ratio: int = 4

    # Training
    context_len: int = 512        # train length. Keep modest so GDN's Python loop is tolerable.
    batch_size: int = 16
    learning_rate: float = 3e-4
    n_steps: int = 800
    warmup_steps: int = 80

    # Eval
    eval_every: int = 100
    eval_steps: int = 20

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data
# ============================================================

class TokenDataset(Dataset):
    """Loads wikitext-2, tokenizes with GPT-2, chunks into fixed-length sequences."""

    def __init__(self, split: str, context_len: int, tokenizer):
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(d["text"] for d in ds if d["text"].strip())
        tokens = tokenizer.encode(text)

        # Chunk into sequences of length (context_len + 1) so we can form (x, y) pairs
        n_seqs = len(tokens) // (context_len + 1)
        tokens = tokens[: n_seqs * (context_len + 1)]
        self.data = torch.tensor(tokens, dtype=torch.long).view(n_seqs, context_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# ============================================================
# Attention variants
# ============================================================

class StandardAttention(nn.Module):
    """
    Softmax attention. O(T^2) cost per layer, perfect recall across the window.

    This is the baseline the two linear variants are trying to approximate with
    linear cost per token.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, Dh)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class LinearAttention(nn.Module):
    """
    Linear attention with ELU+1 feature map (Katharopoulos et al. 2020).

    Key idea: replace softmax(Q K^T) V with phi(Q) (phi(K)^T V), which lets us
    maintain a running state S = sum_i phi(k_i) v_i^T. Output at step t is
        (phi(q_t) @ S_t) / (phi(q_t) @ z_t)
    where z_t = sum_i phi(k_i) is a normalizer.

    Implemented here via causal cumsum. Memory is O(B H T Dh^2) for the cumsum
    tensor. We keep Dh=32 to stay reasonable.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        qkv = self.qkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)

        q = self.phi(q)
        k = self.phi(k)

        # Outer products phi(k_t) v_t^T, shape (B, H, T, Dh, Dh)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        kv_cumsum = kv.cumsum(dim=2)  # causal prefix sum of the state updates

        # Numerator: phi(q_t) @ S_t  ->  (B, H, T, Dh)
        num = torch.einsum("bhtd,bhtde->bhte", q, kv_cumsum)

        # Denominator for stability: phi(q_t) . sum_{i<=t} phi(k_i)
        k_cumsum = k.cumsum(dim=2)
        denom = torch.einsum("bhtd,bhtd->bht", q, k_cumsum).unsqueeze(-1).clamp_min(1e-6)

        out = num / denom
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class GatedDeltaNetAttention(nn.Module):
    """
    Simplified Gated DeltaNet (Yang, Kautz, Hatamizadeh 2024).

    Maintains a per-head state matrix S of shape (Dh, Dh) updated per step:

        S_t = alpha_t * S_{t-1} * (I - beta_t * k_t k_t^T) + beta_t * v_t k_t^T
        o_t = S_t @ q_t

    Interpretation:
      - beta_t (in (0,1)) is a per-step "learning rate" for the delta rule,
        which overwrites whatever was previously associated with k_t to v_t.
      - alpha_t (in (0,1)) is a forget gate on the prior state. alpha=1 recovers
        plain DeltaNet. alpha<1 lets the state decay, similar to Mamba's selective
        forgetting.

    This is a naive sequential implementation with a Python for-loop over time.
    Production implementations use a chunk-parallel form. At tiny scale the
    for-loop is OK but will be noticeably slower than the cumsum-based linear
    attention above for training.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gate_proj = nn.Linear(dim, n_heads, bias=True)   # alpha
        self.beta_proj = nn.Linear(dim, n_heads, bias=True)   # beta
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        qkv = self.qkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)

        # L2-normalize q and k for stability of the delta rule
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        alpha = torch.sigmoid(self.gate_proj(x)).permute(0, 2, 1)  # (B, H, T)
        beta = torch.sigmoid(self.beta_proj(x)).permute(0, 2, 1)   # (B, H, T)

        # Keep state in fp32 for numerical stability over long sequences
        S = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=torch.float32)
        outs = []

        for t in range(T):
            k_t = k[:, :, t, :].float()                                # (B, H, Dh)
            v_t = v[:, :, t, :].float()
            q_t = q[:, :, t, :].float()
            a_t = alpha[:, :, t].float().unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            b_t = beta[:, :, t].float().unsqueeze(-1).unsqueeze(-1)

            # S <- a * [S - b * (S k) k^T] + b * v k^T
            Sk = torch.einsum("bhij,bhj->bhi", S, k_t)                 # (B, H, Dh)
            S = (
                a_t * (S - b_t * Sk.unsqueeze(-1) * k_t.unsqueeze(-2))
                + b_t * v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
            )

            o_t = torch.einsum("bhij,bhj->bhi", S, q_t)                # (B, H, Dh)
            outs.append(o_t)

        out = torch.stack(outs, dim=2).to(x.dtype)        # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# ============================================================
# Model
# ============================================================

ATTENTION_CLASSES = {
    "standard": StandardAttention,
    "linear": LinearAttention,
    "gated_delta": GatedDeltaNetAttention,
}


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, attention_type: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = ATTENTION_CLASSES[attention_type](cfg.dim, cfg.n_heads)
        self.norm2 = nn.LayerNorm(cfg.dim)
        hidden = cfg.dim * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, cfg.dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config, attention_type: str):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        # Learned positional embedding, sized to max context we'll ever use
        self.pos_emb = nn.Embedding(max(cfg.context_len, 8192), cfg.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, attention_type) for _ in range(cfg.n_layers)]
        )
        self.norm_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # tie

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self, non_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n


# ============================================================
# Training and evaluation
# ============================================================

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.n_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def eval_loss(model, loader, cfg):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= cfg.eval_steps:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_one(attention_type: str, cfg: Config, train_ds, val_ds):
    torch.manual_seed(42)
    model = TinyLM(cfg, attention_type).to(cfg.device)
    print(f"[{attention_type}] non-embedding params: {model.num_params(True):,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    history = {"step": [], "train_loss": [], "val_loss": []}

    step = 0
    train_iter = iter(train_loader)
    t_start = time.time()

    while step < cfg.n_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(cfg.device), y.to(cfg.device)

        lr = get_lr(step, cfg)
        for pg in optim.param_groups:
            pg["lr"] = lr

        _, loss = model(x, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % cfg.eval_every == 0 or step == cfg.n_steps - 1:
            val = eval_loss(model, val_loader, cfg)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val)
            elapsed = time.time() - t_start
            print(f"  step {step:5d}  train {loss.item():.3f}  val {val:.3f}  ({elapsed:.1f}s)")
        step += 1

    wall = time.time() - t_start
    print(f"[{attention_type}] total training time: {wall:.1f}s\n")
    return model, history, wall


# ============================================================
# Benchmarking
# ============================================================

@torch.no_grad()
def benchmark_inference(attention_type: str, cfg: Config, seq_lens, n_warmup=2, n_trials=5):
    """Measure forward-pass (prefill) latency at various context lengths."""
    bench_cfg = Config(**{**asdict(cfg), "context_len": max(seq_lens)})
    model = TinyLM(bench_cfg, attention_type).to(cfg.device).eval()

    results = {}
    for T in seq_lens:
        x = torch.randint(0, cfg.vocab_size, (1, T), device=cfg.device)

        for _ in range(n_warmup):
            _ = model(x)
        if cfg.device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_trials):
            _ = model(x)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t_total = time.time() - t0

        ms_per_1k = (t_total / n_trials) * 1000 / (T / 1000)
        results[T] = ms_per_1k
        print(f"  [{attention_type}] T={T:5d}: {ms_per_1k:.2f} ms per 1k tokens")

    del model
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return results


# ============================================================
# Plotting
# ============================================================

def plot_results(histories, bench_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"standard": "C0", "linear": "C1", "gated_delta": "C2"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, hist in histories.items():
        ax.plot(
            hist["step"], hist["val_loss"],
            label=f"{name}", color=colors[name], marker="o", markersize=3,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Training curves (validation loss)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, res in bench_results.items():
        lens = sorted(res.keys())
        ax.plot(lens, [res[l] for l in lens], label=name, color=colors[name], marker="o")
    ax.set_xlabel("Context length")
    ax.set_ylabel("Latency (ms per 1K tokens)")
    ax.set_title("Inference latency vs context length")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "inference_latency.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()
    out_dir = Path("./results")

    print(f"Device: {cfg.device}")
    print(f"Config: {asdict(cfg)}\n")

    print("Loading wikitext-2...")
    tok = tiktoken.get_encoding("gpt2")
    class TokWrapper:
        def encode(self, s):
            return tok.encode(s, allowed_special="all")
    tok_wrapper = TokWrapper()
    train_ds = TokenDataset("train", cfg.context_len, tok_wrapper)
    val_ds = TokenDataset("validation", cfg.context_len, tok_wrapper)
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}\n")

    histories = {}
    train_times = {}
    for atype in ["standard", "linear", "gated_delta"]:
        print(f"=== Training {atype} ===")
        _, hist, wt = train_one(atype, cfg, train_ds, val_ds)
        histories[atype] = hist
        train_times[atype] = wt

    print("\n=== Inference benchmark ===")
    bench_results = {}
    seq_lens = [512, 1024, 2048, 4096]
    for atype in ["standard", "linear", "gated_delta"]:
        print(f"--- {atype} ---")
        bench_results[atype] = benchmark_inference(atype, cfg, seq_lens)

    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {
                "histories": histories,
                "train_times": train_times,
                "benchmark": {
                    k: {str(kk): vv for kk, vv in v.items()}
                    for k, v in bench_results.items()
                },
                "config": asdict(cfg),
            },
            f, indent=2,
        )
    plot_results(histories, bench_results, out_dir)

    print("\n=== Summary ===")
    print(f"{'Method':<16} {'Final val loss':<16} {'Train time (s)':<16}")
    for atype in ["standard", "linear", "gated_delta"]:
        final_loss = histories[atype]["val_loss"][-1]
        print(f"{atype:<16} {final_loss:<16.3f} {train_times[atype]:<16.1f}")


if __name__ == "__main__":
    main()
