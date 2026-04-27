"""
Compare standard cross-entropy training vs. mistake-only backprop training
for a small transformer language model on word-level WikiText-2.

Extended to save model checkpoints (baseline.pt, mistake_only.pt) after
training and generate text samples from each trained model at multiple
temperatures.

Use --only_generate to skip training and just run the generation comparison
if baseline.pt and mistake_only.pt already exist.

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
           - train for N_STEPS on WikiText-2 with the condition's loss
           - evaluate on held-out data
    5. Save final weights; run generation comparison.

Notes on word-level dynamics:
    * Vocab ~33k, so chance CE is ~10.4 nats (vs ~4.2 at char level).
    * Early training has near-100% miss rate; mistake-only and baseline
      behave almost identically until accuracy climbs above a few percent.
    * Recommended n_steps: >=10000 to see meaningful divergence.

Usage:
    python mistake_only_wikitext.py
    python mistake_only_wikitext.py --quick
    python mistake_only_wikitext.py --n_steps 20000
    python mistake_only_wikitext.py --only_generate
"""

import argparse
import copy
import json
import math
import os
import time
import shutil
import urllib.request
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Config & reproducibility --------------------
SEED = 42
_PQ_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-2-v1"
DATA_DIR = "wikitext-2"


@dataclass
class ModelConfig:
    vocab_size: int = 33278   # overwritten from data at runtime
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


@dataclass
class TrainConfig:
    n_steps: int = 10000
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


# -------------------- Data (WikiText-2, word-level) --------------------
def get_data():
    """
    Download WikiText-2 (v1, with <unk> for OOV), tokenize by whitespace,
    build a vocab from training tokens, and return train/val tensors plus
    the string<->int maps.
    """
    if not os.path.isdir(DATA_DIR):
        print("Downloading WikiText-2...")
        import pandas as pd
        os.makedirs(DATA_DIR, exist_ok=True)
        for split, fname in [("train", "wiki.train.tokens"), ("validation", "wiki.valid.tokens")]:
            url = f"{_PQ_BASE}/{split}/0000.parquet"
            pq_path = os.path.join(DATA_DIR, f"{split}.parquet")
            with urllib.request.urlopen(url) as resp:
                with open(pq_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
            df = pd.read_parquet(pq_path)
            with open(os.path.join(DATA_DIR, fname), "w", encoding="utf-8") as f:
                for text in df["text"]:
                    f.write(text + "\n")
            os.remove(pq_path)

    def read_tokens(split):
        path = os.path.join(DATA_DIR, f"wiki.{split}.tokens")
        with open(path, "r", encoding="utf-8") as f:
            # wikitext-2-v1 is already whitespace-tokenized; <unk> present
            return f.read().split()

    train_tokens = read_tokens("train")
    val_tokens = read_tokens("valid")

    vocab = sorted(set(train_tokens))
    if "<unk>" not in vocab:
        vocab.append("<unk>")
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    unk_id = stoi["<unk>"]

    train_ids = torch.tensor([stoi[t] for t in train_tokens], dtype=torch.long)
    val_ids = torch.tensor(
        [stoi.get(t, unk_id) for t in val_tokens], dtype=torch.long
    )
    return train_ids, val_ids, len(vocab), stoi, itos


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


def hybrid_ce_loss_fn(lam):
    """
    Returns a closure: loss = mean_CE_over_mistakes + lam * mean_CE_over_correct.

    lam = 0.0  -> mistake-only (identical to mistake_only_loss up to numerics).
    lam -> inf -> correct-only.
    At miss rate m, 'equivalent to standard CE' corresponds to lam = (1-m)/m;
    for our converged miss ~0.67 that's lam ~= 0.5. Values below 0.5 are more
    mistake-weighted than standard CE; above, more correct-weighted.
    """
    def _loss(logits, targets):
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        with torch.no_grad():
            preds = flat_logits.argmax(dim=-1)
            mis_mask = (preds != flat_targets).float()
            cor_mask = 1.0 - mis_mask
        n_mis = mis_mask.sum()
        n_cor = cor_mask.sum()
        # Empty-set guards; keep graph intact for .backward().
        mis_term = (
            (per_token * mis_mask).sum() / n_mis
            if n_mis.item() > 0 else flat_logits.sum() * 0.0
        )
        cor_term = (
            (per_token * cor_mask).sum() / n_cor
            if n_cor.item() > 0 else flat_logits.sum() * 0.0
        )
        return mis_term + lam * cor_term
    return _loss


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

        onehot_probs = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        # Brier = sum_k (p_k - y_k)^2 = sum_k p_k^2 - 2 p_true + 1  (one-hot y)
        # This avoids materializing a dense one-hot tensor (huge for large vocab).
        brier_sum += ((probs ** 2).sum(dim=-1) - 2 * onehot_probs + 1).sum().item()

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
    ema_full = 0.0
    ema_mis = 0.0
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

        # Measure both losses on this batch regardless of which was used for the gradient
        with torch.no_grad():
            det_logits = logits.detach()
            tr_full_val = float(standard_ce_loss(det_logits, y).item())
            tr_mis_val = float(mistake_only_loss(det_logits, y).item())
        mf = mistake_fraction(logits.detach(), y)
        if step == 1:
            running_miss = mf
            ema_full = tr_full_val
            ema_mis = tr_mis_val
        else:
            running_miss = 0.98 * running_miss + 0.02 * mf
            ema_full = 0.98 * ema_full + 0.02 * tr_full_val
            ema_mis = 0.98 * ema_mis + 0.02 * tr_mis_val

        if step % tcfg.eval_every == 0 or step == 1:
            val_sampler_eval = make_batch_sampler(
                val_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 2
            )
            metrics = evaluate(model, val_sampler_eval, tcfg.eval_batches, device)
            entry = {
                "step": step,
                "train_loss": float(loss.item()),
                "tr_full": ema_full,
                "tr_mis": ema_mis,
                "mistake_frac_ema": running_miss,
                "elapsed_s": time.time() - t0,
                **{k: v for k, v in metrics.items() if k != "reliability"},
            }
            history.append(entry)
            print(
                f"[{label:>13}] step {step:5d} | "
                f"tr_full {ema_full:7.4f} | tr_mis {ema_mis:7.4f} | "
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

    # checkpoint the final weights so generation comparison can load them
    ckpt_path = f"{label.strip().replace('-', '_').replace(' ', '_')}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  saved checkpoint to {ckpt_path}")

    return history, final_metrics


# -------------------- Generation & distribution inspection --------------------
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, device="cpu", seed=42):
    """Autoregressively generate tokens. Deterministic given the seed."""
    model.eval()
    idx = prompt_ids.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits = model(idx_cond)[:, -1, :]

        if temperature < 1e-6:
            idx_next = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1, generator=gen)

        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx


@torch.no_grad()
def next_token_distribution(model, prompt_ids, itos, top_k=8, device="cpu"):
    """Return (top_k (token, prob) pairs, entropy in nats) for the next token."""
    model.eval()
    idx = prompt_ids.to(device)
    idx_cond = idx[:, -model.cfg.block_size:]
    logits = model(idx_cond)[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_probs, top_idx = torch.topk(probs, top_k)
    H = -(probs * probs.clamp(min=1e-12).log()).sum().item()
    model.train()
    return (
        [(itos[i.item()], p.item()) for p, i in zip(top_probs[0], top_idx[0])],
        H,
    )


def _fmt_tok(tok, width=14):
    """Truncate/escape token for side-by-side display."""
    s = tok.replace("\n", "\\n")
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s


def compare_generations(mcfg, itos, stoi, device, prompts=None, temperatures=(0.0, 0.7, 1.0), max_new_tokens=80, extra_models=None):
    """
    Load checkpoints and print next-token distributions + samples.

    Loads baseline.pt and mistake_only.pt by default. If extra_models is
    provided (list of (label, path) tuples), those are appended to the
    comparison so hybrid sweep runs can be included.
    """
    if prompts is None:
        prompts = [
            "The history of",
            "In the 19th century ,",
            "The novel was published in",
            "Scientists have discovered that",
        ]

    ckpt_list = [("baseline", "baseline.pt"), ("mistake-only", "mistake_only.pt")]
    if extra_models:
        ckpt_list += list(extra_models)

    models = []  # preserve order
    for label, path in ckpt_list:
        if not os.path.exists(path):
            print(f"  missing checkpoint {path}; skipping {label}")
            continue
        m = GPT(mcfg).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        models.append((label, m))
    if len(models) < 2:
        print("  need at least 2 checkpoints; skipping generation comparison")
        return

    unk_id = stoi.get("<unk>", 0)

    for prompt in prompts:
        # Word-level: split by whitespace. Replace OOV with <unk>.
        prompt_tokens = prompt.split()
        prompt_ids_list = [stoi.get(t, unk_id) for t in prompt_tokens]
        missing = [t for t in prompt_tokens if t not in stoi]
        if missing:
            print(f"\n  (prompt uses {missing!r}, mapped to <unk>)")
        prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long)

        print("\n" + "=" * 74)
        print(f"PROMPT: {prompt!r}")
        print("=" * 74)

        # Next-token distributions. Stack vertically per condition so this
        # scales cleanly beyond two models.
        print("\n  Next-token distribution (top 8):")
        for label, m in models:
            dist, H = next_token_distribution(m, prompt_ids, itos, top_k=8, device=device)
            print(f"    [{label}] H = {H:.2f} nats")
            for tok, p in dist:
                print(f"      {_fmt_tok(tok):>14} {p:6.3f}")

        # generations at each temperature, same seed so RNG is shared
        for T in temperatures:
            print(f"\n  --- temperature {T} ---")
            for label, m in models:
                out = generate(m, prompt_ids, max_new_tokens=max_new_tokens, temperature=T, device=device, seed=42)
                ids = out[0].tolist()
                prompt_text = " ".join(itos[i] for i in ids[: len(prompt_ids_list)])
                gen_text = " ".join(itos[i] for i in ids[len(prompt_ids_list):])
                print(f"\n  [{label}] >>>{prompt_text}<<< {gen_text}")


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short run")
    parser.add_argument("--out", default="results.json")
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
    parser.add_argument(
        "--only_generate",
        action="store_true",
        help="Skip training and just run generation comparison "
             "(requires baseline.pt and mistake_only.pt to exist).",
    )
    parser.add_argument(
        "--lambdas",
        default="",
        help="Comma-separated list of hybrid lambdas to sweep, e.g. '0.1,0.3,1.0'. "
             "Each runs an additional training phase with loss = "
             "mean_CE_mistakes + lam * mean_CE_correct. "
             "Empty string (default) skips the sweep.",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip the baseline and mistake-only training phases. Useful when "
             "you already have results for those and only want to run a new "
             "hybrid sweep from the same noise-pretrained init. Requires "
             "--lambdas to be non-empty.",
    )
    args = parser.parse_args()

    # Parse lambda sweep values early so we can validate before any work.
    lambdas = []
    if args.lambdas.strip():
        try:
            lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
        except ValueError as e:
            parser.error(f"invalid --lambdas: {e}")
        for lam in lambdas:
            if lam < 0:
                parser.error(f"--lambdas must be non-negative; got {lam}")
    if args.skip_baselines and not lambdas:
        parser.error("--skip_baselines requires --lambdas to be non-empty")
    if args.skip_baselines and os.path.exists(args.out):
        print(
            f"WARNING: --skip_baselines is set and {args.out} already exists. "
            f"It will be overwritten with results that contain only hybrid runs. "
            f"Consider passing --out <new_path> to preserve prior results."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data, vocab_size, stoi, itos = get_data()
    print(
        f"Vocab: {vocab_size} | train tokens: {len(train_data):,} | "
        f"val tokens: {len(val_data):,}"
    )

    mcfg = ModelConfig(vocab_size=vocab_size)
    tcfg = TrainConfig(n_steps=args.n_steps)
    ncfg = NoiseConfig(n_steps=args.noise_steps, noise_std=args.noise_std)
    if args.quick:
        tcfg.n_steps = 400
        tcfg.eval_every = 100
        tcfg.eval_batches = 10
        ncfg.n_steps = min(ncfg.n_steps, 150)
        ncfg.log_every = 50

    if args.only_generate:
        print("\nSkipping training; running generation comparison only.")
        print("=" * 70)
        print("GENERATION COMPARISON")
        print("=" * 70)
        extra_ckpts = [
            (f"hybrid-lam{lam:g}", f"hybrid_lam{lam:g}.pt") for lam in lambdas
        ]
        compare_generations(mcfg, itos, stoi, device, extra_models=extra_ckpts)
        return

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

    # ---- Phase 1: baseline CE ----
    hist_baseline, final_baseline = None, None
    if not args.skip_baselines:
        print("=" * 70)
        print("Phase 1: Baseline (standard cross-entropy)")
        print("=" * 70)
        hist_baseline, final_baseline = train_model(
            standard_ce_loss, "baseline", init_state, mcfg, tcfg, train_data, val_data, device
        )
    else:
        print("Phase 1 (baseline): skipped via --skip_baselines")

    # ---- Phase 2: mistake-only ----
    hist_mistake, final_mistake = None, None
    if not args.skip_baselines:
        print("\n" + "=" * 70)
        print("Phase 2: Mistake-only backprop")
        print("=" * 70)
        hist_mistake, final_mistake = train_model(
            mistake_only_loss, "mistake-only", init_state, mcfg, tcfg, train_data, val_data, device
        )
    else:
        print("Phase 2 (mistake-only): skipped via --skip_baselines")

    # ---- Phase 3+: hybrid sweep over lambda ----
    # Each run is identical in every respect (init, batches, optimizer, eval)
    # except for the loss-fn's lambda parameter. Labels encode lambda so the
    # saved .pt checkpoints have distinct filenames.
    hybrid_runs = []
    for lam in lambdas:
        label = f"hybrid-lam{lam:g}"
        print("\n" + "=" * 70)
        print(f"Phase 3+: Hybrid (lambda={lam:g})  "
              f"[loss = CE_mistakes + {lam:g} * CE_correct]")
        print("=" * 70)
        hist, final = train_model(
            hybrid_ce_loss_fn(lam), label,
            init_state, mcfg, tcfg, train_data, val_data, device,
        )
        hybrid_runs.append({"lambda": lam, "label": label, "history": hist, "final": final})

    # ---- Final comparison ----
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
    if hist_baseline is not None and hist_mistake is not None:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        b, m = hist_baseline[-1], hist_mistake[-1]
        print(f"{'Metric':<22} {'Baseline':>12} {'Mistake-only':>14} {'Delta':>12}")
        print("-" * 62)
        for label, key in fields:
            delta = m[key] - b[key]
            print(f"{label:<22} {b[key]:>12.4f} {m[key]:>14.4f} {delta:>+12.4f}")
        print(f"\nUniform-entropy reference for vocab={vocab_size}: {math.log(vocab_size):.3f} nats")

    # Extra table: all conditions side by side (include baselines if present).
    if hybrid_runs:
        print("\n" + "=" * 70)
        print("HYBRID SWEEP (all final metrics)")
        print("=" * 70)
        cols = []
        if hist_baseline is not None:
            cols.append(("Baseline", hist_baseline[-1]))
        if hist_mistake is not None:
            cols.append(("Mistake-only", hist_mistake[-1]))
        cols += [(f"lam={r['lambda']:g}", r["history"][-1]) for r in hybrid_runs]
        col_widths = [max(12, len(name) + 1) for name, _ in cols]
        header = f"{'Metric':<22}"
        for (name, _), w in zip(cols, col_widths):
            header += f" {name:>{w}}"
        print(header)
        print("-" * len(header))
        for label, key in fields:
            row = f"{label:<22}"
            for (_, entry), w in zip(cols, col_widths):
                row += f" {entry[key]:>{w}.4f}"
            print(row)

    # ---- JSON output ----
    out_json = {
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "noise_config": asdict(ncfg),
        "noise_history": noise_history,
    }
    if hist_baseline is not None:
        out_json["baseline"] = hist_baseline
        out_json["final_baseline_reliability"] = final_baseline["reliability"]
    if hist_mistake is not None:
        out_json["mistake_only"] = hist_mistake
        out_json["final_mistake_reliability"] = final_mistake["reliability"]
    if hybrid_runs:
        out_json["hybrid_runs"] = [
            {
                "lambda": r["lambda"],
                "label": r["label"],
                "history": r["history"],
                "final_reliability": r["final"]["reliability"],
            }
            for r in hybrid_runs
        ]
    with open(args.out, "w") as fh:
        json.dump(out_json, fh, indent=2)
    print(f"\nWrote full history to {args.out}")

    # -------- Generation comparison --------
    print("\n" + "=" * 70)
    print("GENERATION COMPARISON")
    print("=" * 70)
    extra_ckpts = [(r["label"], f"{r['label'].replace('-', '_')}.pt") for r in hybrid_runs]
    compare_generations(mcfg, itos, stoi, device, extra_models=extra_ckpts)


if __name__ == "__main__":
    main()
