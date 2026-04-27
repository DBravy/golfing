"""
2x2 experiment: standard attention vs XSA vs sparsemax-on-K vs both,
on word-level WikiText-2.

Four conditions, each trained from an identical init with an identical
batch stream so the only axis of variation is the attention variant:

    baseline          : standard causal self-attention
    xsa               : remove projection of attn output onto self value v_i
    sparsemax_k       : replace softmax on attn scores unchanged, but apply
                        sparsemax along the feature axis of K before scoring
    xsa_sparsemax_k   : both

XSA follows Zhai 2026: after y = softmax(QK^T / sqrt(d)) V, compute
    z = y - (y . v_hat) v_hat,  v_hat = v / ||v||
per head, per token. See Algorithm 1 in the XSA paper.

Sparsemax-on-K applies sparsemax (Martins & Astudillo 2016) along the
d_head axis of K, per head, per token. This produces a non-negative,
L1-normalized, sparse key vector -- the "DG-analog" orthogonalization
of the stored addresses.

Notes:
  * The 1/sqrt(d_head) attention score scaling is kept in all conditions
    for simplicity. With sparsemax(K) the scores will start smaller
    than baseline, but W_k adapts during training. If you want a
    tighter control, add a learnable per-head temperature (not done here
    to keep parameter counts identical across conditions).
  * Vocab ~33k, so chance CE is ~10.4 nats.
  * Recommended n_steps: >=10000 to see meaningful divergence.

Usage:
    python xsa_sparsemax_k_2x2.py
    python xsa_sparsemax_k_2x2.py --quick
    python xsa_sparsemax_k_2x2.py --n_steps 20000
    python xsa_sparsemax_k_2x2.py --conditions baseline,xsa
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

ALL_CONDITIONS = ["baseline", "xsa", "sparsemax_k", "xsa_sparsemax_k"]


@dataclass
class ModelConfig:
    vocab_size: int = 33278   # overwritten from data at runtime
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    attention_mode: str = "baseline"


@dataclass
class TrainConfig:
    n_steps: int = 10000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20


# -------------------- Data (WikiText-2, word-level) --------------------
def get_data():
    """
    Download WikiText-2 (v1, with <unk> for OOV), tokenize by whitespace,
    build a vocab from training tokens, return train/val tensors + maps.
    """
    if not os.path.isdir(DATA_DIR):
        print("Downloading WikiText-2...")
        import pandas as pd
        os.makedirs(DATA_DIR, exist_ok=True)
        for split, fname in [("train", "wiki.train.tokens"),
                             ("validation", "wiki.valid.tokens")]:
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
    """Deterministic batch sampler so all runs see identical batches."""
    gen = torch.Generator().manual_seed(seed)

    def sample():
        ix = torch.randint(
            len(data) - block_size - 1, (batch_size,), generator=gen
        )
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        return x, y

    return sample


# -------------------- Sparsemax --------------------
def sparsemax(z, dim=-1):
    """
    Sparsemax (Martins & Astudillo 2016). Projects z onto the probability
    simplex in a way that admits sparse solutions. Fully autograd-compatible:
    the clamp-at-zero and gather have well-defined gradients in PyTorch.

    Args:
        z: tensor of scores
        dim: axis to apply sparsemax over
    Returns:
        p: tensor same shape as z, non-negative, sum-to-1 along `dim`,
           typically sparse.
    """
    # Move target dim to the last position for convenience.
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    d = orig_shape[-1]
    z_flat = z.reshape(-1, d)

    # Sort descending.
    z_sorted, _ = torch.sort(z_flat, dim=-1, descending=True)
    z_cumsum = z_sorted.cumsum(dim=-1)

    # rho(z) = max { j in [1..d] : 1 + j * z_(j) > sum_{i<=j} z_(i) }
    k = torch.arange(1, d + 1, device=z.device, dtype=z.dtype).unsqueeze(0)
    support = (1 + k * z_sorted) > z_cumsum            # bool, (N, d)
    rho = support.to(z.dtype).sum(dim=-1, keepdim=True)  # (N, 1)

    # tau(z) = (sum_{i<=rho} z_(i) - 1) / rho
    rho_idx = (rho.long() - 1).clamp(min=0)            # (N, 1)
    z_cumsum_at_rho = torch.gather(z_cumsum, -1, rho_idx)
    tau = (z_cumsum_at_rho - 1) / rho.clamp(min=1)

    p_flat = torch.clamp(z_flat - tau, min=0)
    p = p_flat.reshape(orig_shape).transpose(dim, -1)
    return p


# -------------------- Model --------------------
class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with one of four modes:
        "baseline"         -- standard softmax attention
        "xsa"              -- baseline + XSA output projection
        "sparsemax_k"      -- sparsemax(K, dim=d_head) + baseline output
        "xsa_sparsemax_k"  -- both

    Parameter count is identical across modes.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.attention_mode in ALL_CONDITIONS, \
            f"unknown attention_mode {cfg.attention_mode}"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.mode = cfg.attention_mode
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
        # (B, H, T, d_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # ----- Sparsemax on K (DG-analog) -----
        if self.mode in ("sparsemax_k", "xsa_sparsemax_k"):
            # Sparsemax along the feature axis of each key, per head.
            # Forces each stored address to be a non-negative, sparse,
            # L1-normalized code.
            k = sparsemax(k, dim=-1)

        # ----- Standard scaled-dot-product attention -----
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout_p, training=self.training)
        y = att @ v  # (B, H, T, d_head)

        # ----- XSA (CA1-analog): remove y's component along v_hat -----
        if self.mode in ("xsa", "xsa_sparsemax_k"):
            v_hat = F.normalize(v, dim=-1)
            # Project y onto v_hat, then subtract.
            proj_coef = (y * v_hat).sum(dim=-1, keepdim=True)
            y = y - proj_coef * v_hat

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

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


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


# -------------------- Training --------------------
def train_model(condition, init_state, mcfg, tcfg, train_data, val_data, device):
    # Override attention_mode for this run.
    run_cfg = ModelConfig(**{**asdict(mcfg), "attention_mode": condition})
    model = GPT(run_cfg).to(device)
    # Load the shared init (parameter shapes are identical across conditions).
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
    ema_loss = 0.0
    model.train()

    for step in range(1, tcfg.n_steps + 1):
        x, y = train_sampler()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        lv = float(loss.item())
        ema_loss = lv if step == 1 else 0.98 * ema_loss + 0.02 * lv

        if step % tcfg.eval_every == 0 or step == 1:
            val_sampler = make_batch_sampler(
                val_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 2
            )
            metrics = evaluate(model, val_sampler, tcfg.eval_batches, device)
            elapsed = time.time() - t0
            entry = {
                "step": step,
                "train_ema": ema_loss,
                "val_loss": metrics["loss"],
                "val_ppl": metrics["perplexity"],
                "val_acc": metrics["accuracy"],
                "val_entropy": metrics["entropy"],
                "val_ece": metrics["ece"],
                "elapsed_s": elapsed,
            }
            history.append(entry)
            print(
                f"  [{condition:>18}] step {step:5d}/{tcfg.n_steps} "
                f"| train_ema {ema_loss:6.4f} "
                f"| val {metrics['loss']:.4f} "
                f"| ppl {metrics['perplexity']:7.2f} "
                f"| acc {metrics['accuracy']:.3f} "
                f"| {elapsed:5.1f}s"
            )

    # Final full eval
    val_sampler = make_batch_sampler(
        val_data, mcfg.block_size, tcfg.batch_size, seed=SEED + 2
    )
    final = evaluate(model, val_sampler, tcfg.eval_batches * 2, device)

    ckpt_path = f"{condition}.pt"
    torch.save({"config": asdict(run_cfg), "state_dict": model.state_dict()},
               ckpt_path)
    print(f"  [{condition}] saved checkpoint to {ckpt_path}")

    return history, final


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--quick", action="store_true",
                        help="Short sanity run (400 steps each).")
    parser.add_argument("--out", default="xsa_sparsemax_k_2x2_results.json")
    parser.add_argument(
        "--conditions", default=",".join(ALL_CONDITIONS),
        help=f"Comma-separated subset of {ALL_CONDITIONS}. "
             "Default runs all four.",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            parser.error(f"unknown condition {c!r}; expected {ALL_CONDITIONS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data, vocab_size, stoi, itos = get_data()
    print(
        f"Vocab: {vocab_size} | train tokens: {len(train_data):,} | "
        f"val tokens: {len(val_data):,}"
    )

    mcfg = ModelConfig(vocab_size=vocab_size)
    tcfg = TrainConfig(n_steps=args.n_steps)
    if args.quick:
        tcfg.n_steps = 400
        tcfg.eval_every = 100
        tcfg.eval_batches = 10

    print(f"\nModel config: {asdict(mcfg)}")
    print(f"Train config: {asdict(tcfg)}")
    print(f"Conditions: {conditions}\n")

    # ---- Build a SHARED init once ----
    # All conditions inherit the exact same initial weights. Attention
    # variants only differ in forward-pass behaviour, not in parameter
    # shape or count, so one state_dict is valid for all of them.
    torch.manual_seed(SEED)
    # Build with baseline mode to produce the state_dict; the attention_mode
    # flag is a forward-pass switch only.
    init_cfg = ModelConfig(**{**asdict(mcfg), "attention_mode": "baseline"})
    init_model = GPT(init_cfg).to(device)
    init_state = copy.deepcopy(init_model.state_dict())
    del init_model

    # ---- Run each condition ----
    results = {}
    for condition in conditions:
        print("=" * 70)
        print(f"Condition: {condition}")
        print("=" * 70)
        hist, final = train_model(
            condition, init_state, mcfg, tcfg, train_data, val_data, device
        )
        results[condition] = {"history": hist, "final": final}

    # ---- Final comparison ----
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
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
    col_w = max(14, max(len(c) for c in conditions) + 2)
    header = f"{'Metric':<22}"
    for c in conditions:
        header += f" {c:>{col_w}}"
    print(header)
    print("-" * len(header))
    for label, key in fields:
        row = f"{label:<22}"
        for c in conditions:
            row += f" {results[c]['final'][key]:>{col_w}.4f}"
        print(row)
    print(f"\nUniform-entropy reference for vocab={vocab_size}: "
          f"{math.log(vocab_size):.3f} nats")

    # ---- 2x2 summary: deltas relative to baseline ----
    if "baseline" in results:
        print("\n" + "=" * 70)
        print("DELTAS vs baseline  (negative val_loss / ppl is better)")
        print("=" * 70)
        base = results["baseline"]["final"]
        print(f"{'Condition':<22} {'dLoss':>10} {'dPPL':>10} "
              f"{'dAcc':>10} {'dECE':>10}")
        print("-" * 64)
        for c in conditions:
            if c == "baseline":
                continue
            f_c = results[c]["final"]
            print(f"{c:<22} "
                  f"{f_c['loss']-base['loss']:>+10.4f} "
                  f"{f_c['perplexity']-base['perplexity']:>+10.4f} "
                  f"{f_c['accuracy']-base['accuracy']:>+10.4f} "
                  f"{f_c['ece']-base['ece']:>+10.4f}")

        # Super-additivity check: does xsa + sparsemax_k beat the sum of
        # individual gains? Positive "super_additive_loss_gain" means the
        # combined intervention gives more loss reduction than the sum of
        # the two individual loss reductions.
        if all(k in results for k in
               ["xsa", "sparsemax_k", "xsa_sparsemax_k"]):
            d_xsa   = base["loss"] - results["xsa"]["final"]["loss"]
            d_spk   = base["loss"] - results["sparsemax_k"]["final"]["loss"]
            d_both  = base["loss"] - results["xsa_sparsemax_k"]["final"]["loss"]
            super_add = d_both - (d_xsa + d_spk)
            print("\nSuper-additivity test (loss-drop relative to baseline):")
            print(f"  xsa alone             : {d_xsa:+.4f}")
            print(f"  sparsemax_k alone     : {d_spk:+.4f}")
            print(f"  both                  : {d_both:+.4f}")
            print(f"  both - (xsa + sparsemax_k)  = {super_add:+.4f}")
            if super_add > 0:
                print("  --> super-additive (both > sum of parts)")
            elif super_add < 0:
                print("  --> sub-additive (overlap between interventions)")
            else:
                print("  --> exactly additive")

    # ---- JSON dump ----
    out = {
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "conditions": conditions,
        "results": {
            c: {
                "history": r["history"],
                "final": {k: v for k, v in r["final"].items()
                          if k != "reliability"},
                "reliability": r["final"]["reliability"],
            }
            for c, r in results.items()
        },
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote full history to {args.out}")


if __name__ == "__main__":
    main()
