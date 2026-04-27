"""
Random-noise pretraining for uncertainty calibration, adapted to a causal LM.

Replication target:
    Cheon & Paik (2025), "Pretraining with random noise for uncertainty calibration"
    https://arxiv.org/abs/2412.17411

Three conditions:
    (A) baseline       : train a small GPT directly on TinyStories
    (B) token_noise    : pretrain on unpaired random token IDs, then train on TinyStories
                         (noise enters through the normal embedding layer)
    (C) emb_noise      : pretrain on Gaussian noise injected AFTER the embedding
                         (bypasses tok_emb on the input side), with random token
                         targets at the output. This is the closer literal analog
                         of the paper's "Gaussian noise input + random label."

All three conditions use the same model, optimizer, LR, and number of DATA steps.
The only difference is whether (B) or (C) got N extra steps of noise pretraining first.

Main evaluation metric: Expected Calibration Error (ECE) on held-out TinyStories,
computed over next-token predictions (confidence = max softmax, correct = argmax == target).

Usage:
    pip install torch datasets transformers
    python noise_pretrain_tinystories.py

Scale it to your compute by adjusting --noise_steps, --data_steps, and --batch_size.
On a single modern GPU the defaults below run in ~15 min and are enough to see
a meaningful ECE gap. CPU runs work but will be slow.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# ------------------------------- Model ------------------------------------- #

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                 .view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _forward_core(self, x, targets=None):
        """Shared post-embedding path: transformer blocks + final norm + head."""
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        return self._forward_core(x, targets)

    def forward_from_embeddings(self, emb, targets=None):
        """
        Forward pass that bypasses tok_emb and takes a (B, T, n_embd) tensor
        directly. Positional embeddings are still added. Used for injecting
        Gaussian noise at the embedding level during pretraining.
        """
        B, T, C = emb.shape
        assert T <= self.cfg.block_size and C == self.cfg.n_embd
        pos = torch.arange(T, device=emb.device).unsqueeze(0)
        x = self.drop(emb + self.pos_emb(pos))
        return self._forward_core(x, targets)


# ------------------------------- Data -------------------------------------- #

def build_tinystories_loader(tokenizer, split: str, block_size: int, batch_size: int):
    """
    Streaming loader: tokenize TinyStories on the fly and yield fixed-length
    (input, target) pairs for next-token prediction.
    """
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    class StoryStream(IterableDataset):
        def __iter__(self):
            buf: List[int] = []
            for ex in ds:
                toks = tokenizer.encode(ex["text"]) + [eos]
                buf.extend(toks)
                while len(buf) >= block_size + 1:
                    chunk = buf[: block_size + 1]
                    buf = buf[block_size:]  # non-overlapping windows
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

    return DataLoader(StoryStream(), batch_size=batch_size)


def random_noise_batch(vocab_size: int, block_size: int, batch_size: int, device):
    """
    Unpaired random inputs and random next-token targets.
    This is the LM analog of "Gaussian noise input + random label from uniform"
    used in the paper.
    """
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y


# ------------------------------- Training ---------------------------------- #

def pretrain_on_noise(model: GPT, steps: int, batch_size: int, lr: float, device):
    """Pretrain with random TOKEN IDs as input + random token IDs as targets."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    t0 = time.time()
    for step in range(steps):
        x, y = random_noise_batch(model.cfg.vocab_size, model.cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            chance = math.log(model.cfg.vocab_size)
            print(f"[tok_noise] {step:5d}/{steps}  loss={loss.item():.3f}  chance={chance:.3f}  "
                  f"t={time.time()-t0:.1f}s")


def pretrain_on_embedding_noise(model: GPT, steps: int, batch_size: int,
                                lr: float, device, noise_std: float):
    """
    Pretrain with GAUSSIAN noise injected at the embedding level.
    This is closer to the paper's literal setup: Gaussian noise input + random target.
    The token embedding table gets no input-side gradient here (it's bypassed), but
    because the output head is tied to it, tok_emb still receives output-side gradient
    from the random-label cross-entropy.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    t0 = time.time()
    B, T, C = batch_size, model.cfg.block_size, model.cfg.n_embd
    V = model.cfg.vocab_size
    for step in range(steps):
        emb = torch.randn(B, T, C, device=device) * noise_std
        y = torch.randint(0, V, (B, T), device=device)
        _, loss = model.forward_from_embeddings(emb, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            chance = math.log(V)
            print(f"[emb_noise] {step:5d}/{steps}  loss={loss.item():.3f}  chance={chance:.3f}  "
                  f"std={noise_std:.3f}  t={time.time()-t0:.1f}s")


def train_on_data(model: GPT, loader, steps: int, lr: float, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    it = iter(loader)
    t0 = time.time()
    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[data]  {step:5d}/{steps}  loss={loss.item():.3f}  "
                  f"t={time.time()-t0:.1f}s")


# ---------------------------- Calibration ---------------------------------- #

@torch.no_grad()
def evaluate_calibration(model: GPT, loader, device, max_tokens: int = 200_000,
                         n_bins: int = 15) -> Tuple[float, float, float, List[Tuple]]:
    """
    Returns (ece, mean_conf, accuracy, bins) where bins is a list of
    (lo, hi, count, avg_conf, accuracy) per bin.
    """
    model.eval()
    confs, corrects = [], []
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        confs.append(conf.flatten().cpu())
        corrects.append((pred == y).flatten().cpu())
        total += y.numel()
        if total >= max_tokens:
            break

    conf = torch.cat(confs).numpy()
    corr = torch.cat(corrects).numpy().astype(float)

    edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    N = len(conf)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append((lo, hi, 0, 0.0, 0.0))
            continue
        acc_bin = float(corr[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += (cnt / N) * abs(acc_bin - avg_conf)
        bins.append((lo, hi, cnt, avg_conf, acc_bin))
    return float(ece), float(conf.mean()), float(corr.mean()), bins


def print_reliability(bins, title):
    print(f"\n{title}")
    print(f"{'bin':>14} {'count':>8} {'conf':>8} {'acc':>8}  gap")
    for lo, hi, cnt, c, a in bins:
        if cnt == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f}] {cnt:8d} {c:8.3f} {a:8.3f}  {a-c:+.3f}")


# -------------------------------- Main ------------------------------------- #

def run_condition(label, cfg, tokenizer, args, device, noise_mode: str,
                  emb_noise_std: float = 0.02):
    """
    noise_mode:
        'none'      -> train on TinyStories only
        'token'     -> random-token-ID pretraining, then TinyStories
        'embedding' -> Gaussian-noise-at-embedding pretraining, then TinyStories
                       (uses emb_noise_std as the Gaussian std)
    """
    print(f"\n===== Condition: {label} (noise_mode={noise_mode}) =====")
    torch.manual_seed(args.seed)
    model = GPT(cfg).to(device)

    if noise_mode == "token":
        pretrain_on_noise(model, args.noise_steps, args.batch_size, args.lr, device)
    elif noise_mode == "embedding":
        pretrain_on_embedding_noise(
            model, args.noise_steps, args.batch_size, args.lr, device, emb_noise_std
        )
    elif noise_mode != "none":
        raise ValueError(f"unknown noise_mode: {noise_mode}")

    train_loader = build_tinystories_loader(tokenizer, "train", args.block_size, args.batch_size)
    train_on_data(model, train_loader, args.data_steps, args.lr, device)

    val_loader = build_tinystories_loader(tokenizer, "validation", args.block_size, args.batch_size)
    ece, mean_conf, acc, bins = evaluate_calibration(model, val_loader, device)
    print(f"\n[{label}] ECE={ece:.4f}  mean_conf={mean_conf:.3f}  top1_acc={acc:.3f}")
    print_reliability(bins, f"Reliability diagram [{label}]")
    return {"label": label, "ece": ece, "mean_conf": mean_conf, "acc": acc, "bins": bins}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_steps", type=int, default=2000,
                        help="Steps of random-noise pretraining for condition B.")
    parser.add_argument("--data_steps", type=int, default=5000,
                        help="Steps of TinyStories training for both conditions.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emb_noise_stds", type=float, nargs="+", default=[0.02, 0.1, 1.0],
                        help="One or more stds for Gaussian embedding-level noise pretraining. "
                             "One emb_noise condition is run per value. "
                             "Default sweeps 0.02 (init-scale), 0.1 (trained-embedding-scale), "
                             "1.0 (paper's literal value from input-space noise).")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the no-pretraining baseline condition.")
    parser.add_argument("--skip_token_noise", action="store_true",
                        help="Skip the random-token-ID pretraining condition.")
    args = parser.parse_args()

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"model: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"vocab={cfg.vocab_size} block={cfg.block_size}")
    n_params = sum(p.numel() for p in GPT(cfg).parameters())
    print(f"param count (incl. tied head): ~{n_params/1e6:.2f}M")

    results = []
    if not args.skip_baseline:
        results.append(run_condition(
            "baseline", cfg, tok, args, device, noise_mode="none"))
    if not args.skip_token_noise:
        results.append(run_condition(
            "token_noise", cfg, tok, args, device, noise_mode="token"))
    for std in args.emb_noise_stds:
        label = f"emb_noise_std={std:g}"
        results.append(run_condition(
            label, cfg, tok, args, device, noise_mode="embedding", emb_noise_std=std))

    print("\n================ Summary ================")
    for r in results:
        print(f"  {r['label']:>22}: ECE={r['ece']:.4f}  mean_conf={r['mean_conf']:.3f}  "
              f"top1_acc={r['acc']:.3f}")


if __name__ == "__main__":
    main()
