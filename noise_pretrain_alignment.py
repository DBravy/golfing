"""
Does random-noise pretraining induce cross-layer MLP alignment?

This script combines two papers:

  - Cheon & Paik (2025), "Pretraining with random noise for uncertainty
    calibration" -- provides the random-noise pretraining procedure.
  - Bray (2026), "Communication Before Computation" -- provides the metric:
    adjacent-layer top-k subspace overlap of the composed MLP product
    W_down @ W_up.

It builds directly on the reference script noise_pretrain_tinystories.py,
reusing its GPT model and noise-pretraining loops, and adds the alignment
measurement at checkpoints throughout pretraining.

Research question
-----------------
Bray shows that during real training of transformer LMs, adjacent layers'
top-k MLP singular vector subspaces go from random-baseline overlap at
init (~sqrt(k/d)) to strongly aligned within the first ~10^3 steps. Does
noise pretraining, which has no linguistic signal at all, produce the
same alignment? If yes, alignment is a product of gradient flow through
the architecture rather than of the content of the training signal.

Measurement
-----------
For each transformer block l, the composed MLP product is
    C_l = W_down_l @ W_up_l              shape (n_embd, n_embd)
Top-k left singular vectors of C_l form a subspace U_l of the residual
stream. Mean adjacent-layer overlap is
    mean over l of  mean( svdvals( U_l.T @ U_{l+1} ) )
which equals the mean cosine of the top-k principal angles, averaged
over adjacent-layer pairs. Compared to an empirical random baseline from
pairs of random k-dim subspaces in R^{n_embd}.

Usage
-----
    python noise_pretrain_alignment.py                    # token-ID noise
    python noise_pretrain_alignment.py --noise_mode embedding
    python noise_pretrain_alignment.py --n_layer 8 --noise_steps 3000
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Model ------------------------------------- #
# Copied verbatim from noise_pretrain_tinystories.py so the script is
# self-contained. The only change is n_layer=6 default and dropout=0.0 (we
# are measuring weights directly and want no stochasticity in the forward pass).

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0


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
        # IMPORTANT: indices into self.mlp are relied on below:
        #   mlp[0] = W_up  (Linear: n_embd -> 4*n_embd)
        #   mlp[2] = W_down (Linear: 4*n_embd -> n_embd)
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
        B, T, C = emb.shape
        assert T <= self.cfg.block_size and C == self.cfg.n_embd
        pos = torch.arange(T, device=emb.device).unsqueeze(0)
        x = self.drop(emb + self.pos_emb(pos))
        return self._forward_core(x, targets)


# ------------------ Cross-layer alignment measurement ---------------------- #

@torch.no_grad()
def composed_mlp_product(block: Block) -> torch.Tensor:
    """
    Return W_down @ W_up, a (n_embd, n_embd) matrix in the residual stream.

    This is Bray's "composed product" whose singular vectors identify the
    directions along which the MLP reads from and writes to the residual
    stream. Note: this ignores the GELU nonlinearity, matching Bray's
    weight-level analysis. For a GELU model this is justified because the
    nonlinearity is fixed and the communicative geometry is carried by the
    two linear projections.
    """
    w_up = block.mlp[0].weight     # (4*n_embd, n_embd)
    w_down = block.mlp[2].weight   # (n_embd, 4*n_embd)
    return w_down @ w_up           # (n_embd, n_embd)


@torch.no_grad()
def top_k_subspace(composed: torch.Tensor, k: int,
                   which: str = "left") -> torch.Tensor:
    """Top-k singular vector subspace of the composed product.

    which='left'  -> write directions (columns of U)
    which='right' -> read directions (columns of V)

    Both live in R^{n_embd}. Returns a (n_embd, k) matrix with orthonormal
    columns.
    """
    U, S, Vh = torch.linalg.svd(composed, full_matrices=False)
    if which == "left":
        return U[:, :k]
    elif which == "right":
        return Vh[:k, :].T
    else:
        raise ValueError(f"which must be 'left' or 'right', got {which}")


@torch.no_grad()
def mean_principal_cosine(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Mean cosine of the principal angles between two subspaces.

    U1, U2: (d, k) matrices with orthonormal columns. The singular values of
    U1.T @ U2 are the cosines of the principal angles between span(U1) and
    span(U2). Perfect alignment = 1.0, orthogonal = 0.0.
    """
    M = U1.T @ U2  # (k, k)
    s = torch.linalg.svdvals(M)
    return float(s.mean().item())


@torch.no_grad()
def random_subspace_baseline(d: int, k: int, n_samples: int = 50,
                             device="cpu") -> float:
    """Empirical expected mean top-k principal cosine between two random
    k-dim subspaces in R^d. Serves as Bray's "random baseline" dotted line.
    Sampled via QR of Gaussian matrices (uniform on the Stiefel manifold).
    """
    overlaps = []
    for _ in range(n_samples):
        A = torch.randn(d, k, device=device)
        B = torch.randn(d, k, device=device)
        QA, _ = torch.linalg.qr(A)
        QB, _ = torch.linalg.qr(B)
        overlaps.append(mean_principal_cosine(QA, QB))
    return sum(overlaps) / len(overlaps)


@torch.no_grad()
def measure_alignment(model: GPT, k: int, which: str = "left") -> dict:
    """Adjacent-layer top-k subspace overlap for every MLP block pair."""
    subspaces = [top_k_subspace(composed_mlp_product(b), k, which)
                 for b in model.blocks]
    adj = [mean_principal_cosine(subspaces[i], subspaces[i + 1])
           for i in range(len(subspaces) - 1)]
    return {
        "per_boundary": adj,
        "mean": sum(adj) / len(adj),
    }


# ------------------- Noise pretraining with measurements ------------------- #

def random_token_batch(vocab_size, block_size, batch_size, device):
    """Unpaired random token IDs as input and as next-token targets."""
    x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    return x, y


def pretrain_with_measurements(model: GPT, steps: int, batch_size: int,
                               lr: float, device, noise_mode: str,
                               checkpoint_steps, k: int, which: str,
                               baseline: float, emb_noise_std: float = 1.0):
    """Run noise pretraining; measure adjacent-layer alignment at checkpoints.

    noise_mode='token'     -> random token IDs, the LM analog of Cheon & Paik.
    noise_mode='embedding' -> Gaussian noise after the embedding layer
                              (bypasses tok_emb on the input side), closer to
                              the paper's literal "Gaussian input + random label."
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            betas=(0.9, 0.999), weight_decay=0.01)
    V = model.cfg.vocab_size
    T = model.cfg.block_size
    C = model.cfg.n_embd

    history = []

    def snapshot(step, loss_val):
        model.eval()
        a = measure_alignment(model, k=k, which=which)
        history.append({"step": step, "loss": loss_val,
                        "mean": a["mean"], "per_boundary": a["per_boundary"]})
        ratio = a["mean"] / baseline if baseline > 0 else float("nan")
        boundaries_str = " ".join(f"{v:.3f}" for v in a["per_boundary"])
        loss_str = f"loss={loss_val:.3f}" if loss_val is not None else "loss=--"
        print(f"[step {step:5d}] {loss_str}  "
              f"mean_adj={a['mean']:.3f}  x{ratio:.2f}_over_baseline  "
              f"per_boundary=[{boundaries_str}]")
        model.train()

    # step-0 measurement before any training
    snapshot(0, None)

    checkpoint_set = set(checkpoint_steps)
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        if noise_mode == "token":
            x, y = random_token_batch(V, T, batch_size, device)
            _, loss = model(x, y)
        elif noise_mode == "embedding":
            emb = torch.randn(batch_size, T, C, device=device) * emb_noise_std
            y = torch.randint(0, V, (batch_size, T), device=device)
            _, loss = model.forward_from_embeddings(emb, y)
        else:
            raise ValueError(f"unknown noise_mode: {noise_mode}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step in checkpoint_set:
            snapshot(step, float(loss.item()))

    if steps not in checkpoint_set:
        snapshot(steps, float(loss.item()))

    print(f"total pretraining time: {time.time() - t0:.1f}s")
    return history


# --------------------------------- Main ----------------------------------- #

def log_spaced_checkpoints(max_step: int) -> List[int]:
    """Log-ish spacing so we see the early phase densely and later steps sparsely,
    matching the checkpoint philosophy used in Bray's paper."""
    candidates = [10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    return [c for c in candidates if c <= max_step]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_mode", type=str, default="token",
                        choices=["token", "embedding"],
                        help="'token' = random-token-ID noise through the embedding, "
                             "'embedding' = Gaussian noise injected post-embedding")
    parser.add_argument("--emb_noise_std", type=float, default=1.0,
                        help="Std for embedding-level Gaussian noise (Cheon & Paik use 1.0).")
    parser.add_argument("--noise_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Only used to size the embedding for token-mode noise.")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k subspace dimension (Bray uses 10 for Pythia).")
    parser.add_argument("--which", type=str, default="left",
                        choices=["left", "right"],
                        help="Compare left (write) or right (read) singular vectors.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"model:  n_layer={cfg.n_layer}  n_embd={cfg.n_embd}  "
          f"n_head={cfg.n_head}  block={cfg.block_size}")
    print(f"params: ~{n_params/1e6:.2f}M (including tied head)")
    print(f"noise:  mode={args.noise_mode}  steps={args.noise_steps}  "
          f"batch={args.batch_size}  lr={args.lr}")
    print(f"metric: top-{args.k} {args.which} singular vector subspaces\n")

    # Random baseline: expected overlap of two random k-dim subspaces in R^{n_embd}.
    baseline = random_subspace_baseline(cfg.n_embd, args.k,
                                        n_samples=100, device=device)
    print(f"random baseline (empirical, R^{cfg.n_embd}, k={args.k}): {baseline:.3f}\n")

    checkpoints = log_spaced_checkpoints(args.noise_steps)
    print(f"measurement checkpoints: [0] + {checkpoints} + [{args.noise_steps}]\n")

    history = pretrain_with_measurements(
        model=model,
        steps=args.noise_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        noise_mode=args.noise_mode,
        checkpoint_steps=checkpoints,
        k=args.k,
        which=args.which,
        baseline=baseline,
        emb_noise_std=args.emb_noise_std,
    )

    # -------------- Summary -------------- #
    print("\n=========== Summary: mean adjacent-layer top-k overlap ===========")
    print(f"{'step':>8}  {'loss':>7}  {'mean_adj':>9}  {'ratio_over_baseline':>20}")
    for h in history:
        loss_str = f"{h['loss']:.3f}" if h["loss"] is not None else "   --"
        ratio = h["mean"] / baseline if baseline > 0 else float("nan")
        print(f"{h['step']:>8}  {loss_str:>7}  {h['mean']:>9.3f}  {ratio:>19.2f}x")
    print(f"random baseline: {baseline:.3f}")

    init_mean = history[0]["mean"]
    final_mean = history[-1]["mean"]
    print(f"\ninit mean adjacent overlap:  {init_mean:.3f} "
          f"({init_mean/baseline:.2f}x baseline)")
    print(f"final mean adjacent overlap: {final_mean:.3f} "
          f"({final_mean/baseline:.2f}x baseline)")
    print(f"change: {final_mean - init_mean:+.3f}  "
          f"(multiplicative: {final_mean/max(init_mean,1e-9):.2f}x)")

    if final_mean > init_mean * 1.5 and final_mean > baseline * 1.5:
        print("\n==> noise pretraining INDUCED cross-layer alignment "
              "(consistent with Bray's early-training surge)")
    elif final_mean > baseline * 1.2:
        print("\n==> noise pretraining produced modest alignment above baseline")
    else:
        print("\n==> noise pretraining did NOT produce clear cross-layer alignment")


if __name__ == "__main__":
    main()
