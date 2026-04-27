"""
Compare standard cross-entropy training vs. mistake-only backprop training
for a small transformer language model on character-level TinyShakespeare,
with an optional Gaussian embedding-noise pretraining phase applied
identically to both conditions before main training begins.

Pipeline:
    1. Build GPT with fixed seed.
    2. (Optional) Noise pretrain: inject Gaussian noise (std=1.0 by default)
       in place of the token embedding output, pair with random token targets,
       train with CE. The tied output head means tok_emb still receives gradient.
    3. Snapshot the resulting state_dict.
    4. For each condition (baseline CE, mistake-only CE):
           - reload the snapshot
           - fresh AdamW
           - identical batch stream
           - train for N_STEPS on TinyShakespeare with the condition's loss
           - evaluate on held-out data

Usage:
    python mistake_only_experiment.py
    python mistake_only_experiment.py --quick
    python mistake_only_experiment.py --noise_steps 0   # disable noise init
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Logging tee --------------------
class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


# -------------------- Config & reproducibility --------------------
SEED = 42
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    n_steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20


@dataclass
class NoiseConfig:
    n_steps: int = 500
    noise_std: float = 1.0
    log_every: int = 100


# -------------------- Data --------------------
def get_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def make_batch_sampler(data, block_size, batch_size, seed):
    """Deterministic batch sampler so both runs see identical batches."""
    gen = torch.Generator().manual_seed(seed)

    def sample():
        ix = torch.randint(
            len(data) - block_size - 1, (batch_size,), generator=gen
        )
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    return sample


# -------------------- Model --------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout_p = cfg.dropout
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout_p, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def _core(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        return self._core(x)

    def forward_from_embeddings(self, emb):
        """
        Bypass the token embedding lookup and feed a (B, T, n_embd) tensor
        directly into the transformer stack. Positional embeddings are still
        added. Used for Gaussian-noise pretraining.
        """
        B, T, C = emb.shape
        assert T <= self.cfg.block_size and C == self.cfg.n_embd
        pos = torch.arange(T, device=emb.device).unsqueeze(0)
        x = self.drop(emb + self.pos_emb(pos))
        return self._core(x)


# -------------------- Loss functions --------------------
def standard_ce_loss(logits, targets):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )


def mistake_only_loss(logits, targets):
    """CE averaged over positions where argmax(logits) != target."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    with torch.no_grad():
        preds = flat_logits.argmax(dim=-1)
        mask = (preds != flat_targets).float()
    n_mistakes = mask.sum()
    if n_mistakes.item() == 0:
        # keep the graph intact so .backward() works; contributes zero gradient
        return flat_logits.sum() * 0.0
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_token * mask).sum() / n_mistakes


def mistake_fraction(logits, targets):
    with torch.no_grad():
        preds = logits.view(-1, logits.size(-1)).argmax(dim=-1)
        return (preds != targets.view(-1)).float().mean().item()


# -------------------- Noise pretraining --------------------
def pretrain_on_embedding_noise(model, ncfg: NoiseConfig, batch_size, lr, device):
    """
    Gaussian-noise-at-embedding pretraining with random token targets.
    Adapted from the emb_noise condition of the reference script.

    For each step:
      emb ~ N(0, noise_std^2) with shape (B, T, n_embd)
      y   ~ Uniform(0, vocab_size) with shape (B, T)
      loss = CE(model.forward_from_embeddings(emb), y)
    """
    torch.manual_seed(SEED + 7)  # independent seed for noise phase
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    V = model.cfg.vocab_size
    C = model.cfg.n_embd
    T = model.cfg.block_size
    chance = math.log(V)
    t0 = time.time()
    history = []

    for step in range(1, ncfg.n_steps + 1):
        emb = torch.randn(batch_size, T, C, device=device) * ncfg.noise_std
        y = torch.randint(0, V, (batch_size, T), device=device)
        logits = model.forward_from_embeddings(emb)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % ncfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            history.append({"step": step, "loss": loss.item(), "elapsed_s": elapsed})
            print(
                f"  [noise-pretrain] step {step:5d}/{ncfg.n_steps} "
                f"| loss {loss.item():6.4f} (chance={chance:.3f}) "
                f"| std={ncfg.noise_std:.2f} | {elapsed:.1f}s"
            )
    return history


# -------------------- Evaluation --------------------
@torch.no_grad()
def evaluate(model, sampler, n_batches, device, n_bins=15):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    brier_sum = 0.0
    entropy_sum = 0.0
    conf_when_correct_sum = 0.0
    conf_when_wrong_sum = 0.0
    n_correct = 0
    n_wrong = 0

    all_conf = []
    all_is_correct = []

    for _ in range(n_batches):
        x, y = sampler()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = y.view(-1)

        loss = F.cross_entropy(flat_logits, flat_targets, reduction="sum")
        total_loss += loss.item()

        probs = F.softmax(flat_logits, dim=-1)
        preds = probs.argmax(dim=-1)
        is_correct = preds == flat_targets

        top_conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
        all_conf.append(top_conf.cpu())
        all_is_correct.append(is_correct.cpu())

        onehot = F.one_hot(flat_targets, num_classes=probs.size(-1)).float()
        brier_sum += ((probs - onehot) ** 2).sum().item()

        entropy_sum += (
            -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).sum().item()
        )

        correct += is_correct.sum().item()
        conf_when_correct_sum += top_conf[is_correct].sum().item()
        conf_when_wrong_sum += top_conf[~is_correct].sum().item()
        n_correct += is_correct.sum().item()
        n_wrong += (~is_correct).sum().item()

        total_tokens += flat_targets.numel()

    model.train()

    confidences = torch.cat(all_conf)
    correctness = torch.cat(all_is_correct).float()

    # ECE with equal-width bins
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    N = confidences.numel()
    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        cnt = int(in_bin.sum().item())
        if cnt > 0:
            acc_b = correctness[in_bin].mean().item()
            conf_b = confidences[in_bin].mean().item()
            ece += (cnt / N) * abs(acc_b - conf_b)
            reliability.append((lo, hi, cnt, conf_b, acc_b))
        else:
            reliability.append((lo, hi, 0, 0.0, 0.0))

    avg_loss = total_loss / total_tokens
    vocab_size = probs.size(-1)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "accuracy": correct / total_tokens,
        "brier": brier_sum / total_tokens,
        "entropy": entropy_sum / total_tokens,
        "entropy_uniform_ref": math.log(vocab_size),
        "ece": ece,
        "conf_when_correct": conf_when_correct_sum / max(n_correct, 1),
        "conf_when_wrong": conf_when_wrong_sum / max(n_wrong, 1),
        "reliability": reliability,
    }


def print_reliability(reliability, title):
    print(f"\n  {title}")
    print(f"  {'bin':>14} {'count':>8} {'conf':>8} {'acc':>8}  gap")
    for lo, hi, cnt, c, a in reliability:
        if cnt == 0:
            continue
        print(f"    [{lo:.2f},{hi:.2f}] {cnt:8d} {c:8.3f} {a:8.3f}  {a-c:+.3f}")


# -------------------- Main training loop --------------------
def train_model(loss_fn, label, init_state, mcfg, tcfg, train_data, val_data, device):
    model = GPT(mcfg).to(device)
    model.load_state_dict(init_state)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        betas=(0.9, 0.95),
        weight_decay=tcfg.weight_decay,
    )

    train_sampler = make_batch_sampler(
        train_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 1
    )

    history = []
    t0 = time.time()
    running_miss = 0.0
    model.train()

    for step in range(1, tcfg.n_steps + 1):
        x, y = train_sampler()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        mf = mistake_fraction(logits.detach(), y)
        running_miss = 0.98 * running_miss + 0.02 * mf if step > 1 else mf

        if step % tcfg.eval_every == 0 or step == 1:
            # Probe losses: compute both full-CE and mistake-only-CE on the
            # current training batch regardless of which condition is running.
            # This gives an apples-to-apples train signal across conditions.
            # In the baseline run, probe_full == loss.item(). In the
            # mistake-only run, probe_mistake == loss.item().
            with torch.no_grad():
                probe_full = standard_ce_loss(logits, y).item()
                probe_mistake = mistake_only_loss(logits, y).item()

            val_sampler_eval = make_batch_sampler(
                val_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 2
            )
            metrics = evaluate(model, val_sampler_eval, tcfg.eval_batches, device)
            entry = {
                "step": step,
                "train_full_loss": probe_full,
                "train_mistake_loss": probe_mistake,
                "mistake_frac_ema": running_miss,
                "elapsed_s": time.time() - t0,
                **{k: v for k, v in metrics.items() if k != "reliability"},
            }
            history.append(entry)
            print(
                f"[{label:>13}] step {step:5d} | "
                f"tr_full {probe_full:7.4f} | tr_mis {probe_mistake:7.4f} | "
                f"val_ce {metrics['loss']:6.4f} | ppl {metrics['perplexity']:6.2f} | "
                f"acc {metrics['accuracy']:.4f} | ece {metrics['ece']:.4f} | "
                f"H {metrics['entropy']:.3f} | miss {running_miss:.3f}"
            )

    # final reliability table for inspection
    val_sampler_eval = make_batch_sampler(
        val_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 2
    )
    final_metrics = evaluate(model, val_sampler_eval, tcfg.eval_batches, device)
    print_reliability(final_metrics["reliability"], f"Reliability [{label}]")

    return history, final_metrics


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short run")
    parser.add_argument("--out", default="results.json")
    parser.add_argument(
        "--log",
        default="mistake_only_experiment.txt",
        help="Path for plain-text log mirroring console output.",
    )
    parser.add_argument(
        "--noise_steps",
        type=int,
        default=500,
        help="Gaussian embedding-noise pretraining steps. 0 to disable.",
    )
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=3000,
        help="Number of main training steps per condition.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--condition",
        choices=["both", "baseline", "mistake"],
        default="both",
        help="Which condition(s) to run.",
    )
    args = parser.parse_args()

    tee = Tee(args.log)
    sys.stdout = tee
    print("$ " + " ".join(sys.argv))
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data, vocab_size = get_data()
    print(
        f"Vocab: {vocab_size} | train tokens: {len(train_data):,} | "
        f"val tokens: {len(val_data):,}"
    )

    mcfg = ModelConfig(vocab_size=vocab_size)
    tcfg = TrainConfig(n_steps=args.n_steps, lr=args.lr)
    ncfg = NoiseConfig(n_steps=args.noise_steps, noise_std=args.noise_std)
    if args.quick:
        tcfg.n_steps = 400
        tcfg.eval_every = 100
        tcfg.eval_batches = 10
        ncfg.n_steps = min(ncfg.n_steps, 150)
        ncfg.log_every = 50

    print(f"\nModel config: {asdict(mcfg)}")
    print(f"Train config: {asdict(tcfg)}")
    print(f"Noise config: {asdict(ncfg)}\n")

    # ---- Build model and (optionally) noise-pretrain once ----
    torch.manual_seed(SEED)
    model = GPT(mcfg).to(device)

    noise_history = []
    if ncfg.n_steps > 0:
        print("=" * 70)
        print(f"Phase 0: Gaussian embedding-noise pretraining (std={ncfg.noise_std})")
        print("=" * 70)
        noise_history = pretrain_on_embedding_noise(
            model, ncfg, tcfg.batch_size, tcfg.lr, device
        )
        print("  done.\n")
    else:
        print("(noise pretraining skipped)\n")

    init_state = copy.deepcopy(model.state_dict())

    run_baseline = args.condition in ("both", "baseline")
    run_mistake = args.condition in ("both", "mistake")

    hist_baseline, final_baseline = [], None
    hist_mistake, final_mistake = [], None

    # ---- Phase 1: baseline CE ----
    if run_baseline:
        print("=" * 70)
        print("Phase 1: Baseline (standard cross-entropy)")
        print("=" * 70)
        hist_baseline, final_baseline = train_model(
            standard_ce_loss, "baseline", init_state, mcfg, tcfg, train_data, val_data, device
        )

    # ---- Phase 2: mistake-only ----
    if run_mistake:
        print("\n" + "=" * 70)
        print("Phase 2: Mistake-only backprop")
        print("=" * 70)
        hist_mistake, final_mistake = train_model(
            mistake_only_loss, "mistake-only", init_state, mcfg, tcfg, train_data, val_data, device
        )

    # ---- Final comparison (only when both ran) ----
    if run_baseline and run_mistake:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        b, m = hist_baseline[-1], hist_mistake[-1]
        fields = [
            ("val_ce (loss)", "loss"),
            ("perplexity", "perplexity"),
            ("top-1 accuracy", "accuracy"),
            ("Brier score", "brier"),
            ("entropy (nats)", "entropy"),
            ("ECE", "ece"),
            ("confidence|correct", "conf_when_correct"),
            ("confidence|wrong", "conf_when_wrong"),
        ]
        print(f"{'Metric':<22} {'Baseline':>12} {'Mistake-only':>14} {'Delta':>12}")
        print("-" * 62)
        for label, key in fields:
            delta = m[key] - b[key]
            print(f"{label:<22} {b[key]:>12.4f} {m[key]:>14.4f} {delta:>+12.4f}")
        print(f"\nUniform-entropy reference for vocab={vocab_size}: {math.log(vocab_size):.3f} nats")

    results = {
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "noise_config": asdict(ncfg),
        "noise_history": noise_history,
    }
    if run_baseline:
        results["baseline"] = hist_baseline
        results["final_baseline_reliability"] = final_baseline["reliability"]
    if run_mistake:
        results["mistake_only"] = hist_mistake
        results["final_mistake_reliability"] = final_mistake["reliability"]

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote full history to {args.out}")
    print(f"Wrote log to {args.log}")
    tee.close()


if __name__ == "__main__":
    main()
