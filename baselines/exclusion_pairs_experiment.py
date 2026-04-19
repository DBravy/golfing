"""
Toy model experiment: mutual exclusion pairs vs. independent features.

Based on Henighan et al. (2023) "Superposition, Memorization, and Double Descent."
https://transformer-circuits.pub/2023/toy-double-descent/

Modification from the paper: instead of every feature firing independently,
features are grouped into mutually exclusive pairs. When a pair fires, exactly
one of its two members is selected. This tests whether exclusion structure
lets the model pack more features per hidden dimension (by placing paired
features on anti-parallel directions), and whether that shifts the
memorization-to-generalization transition toward smaller T.

Run two sweeps (independent baseline and exclusion pairs), then compare:
  - Final training loss vs T
  - Distribution of cos(W[:, 2k], W[:, 2k+1]) for each pair k
  - Mean pair cosine vs T

Prediction:
  - Independent baseline: pair cosines centered near 0 (no reason for the
    model to align arbitrary feature pairs).
  - Exclusion pairs: cosines should concentrate near -1 once the model
    reaches the generalizing regime, since anti-parallel directions let two
    mutually exclusive features share hidden capacity at zero interference.
"""

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------- Config ----------

@dataclass
class Config:
    n: int = 1000                  # number of features (input dim)
    m: int = 4                     # hidden dim (bottleneck). Paper uses 2-6
                                   # for clean superposition. Must be small
                                   # enough to force packing choices.
    p_active: float = 0.01         # per-feature activation prob in the
                                   # independent baseline; used to match
                                   # expected active count in pair mode
    T_sweep: tuple = (10, 30, 100, 300, 1000, 3000, 10000, 30000)
    steps: int = 20_000
    lr: float = 1e-3
    warmup_steps: int = 1_000
    weight_decay: float = 1e-2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Data generation ----------

def sample_dataset(T: int, mode: Literal["independent", "exclusion_pairs"],
                   cfg: Config, rng: torch.Generator) -> torch.Tensor:
    """
    Return X of shape (T, n), row-normalized to unit length.
    Non-negative sparse features.

    independent:
        each of n features fires at p_active, independently.

    exclusion_pairs:
        n must be even. Features (2k, 2k+1) form a pair. The pair fires with
        probability 2 * p_active (to keep expected active count per input
        matched to the independent baseline). When the pair fires, exactly
        one member is chosen uniformly and assigned a U[0,1] magnitude.
    """
    n = cfg.n

    if mode == "independent":
        mask = (torch.rand(T, n, generator=rng) < cfg.p_active).float()
        mag = torch.rand(T, n, generator=rng)
        X = mask * mag

    elif mode == "exclusion_pairs":
        assert n % 2 == 0, "n must be even for exclusion_pairs mode"
        num_pairs = n // 2
        pair_fires = (torch.rand(T, num_pairs, generator=rng)
                      < 2 * cfg.p_active).float()
        # which member fires when the pair is active (0 -> even, 1 -> odd)
        member = (torch.rand(T, num_pairs, generator=rng) < 0.5).float()
        mag = torch.rand(T, num_pairs, generator=rng)

        X = torch.zeros(T, n)
        X[:, 0::2] = pair_fires * (1.0 - member) * mag   # even index
        X[:, 1::2] = pair_fires * member * mag           # odd index

    else:
        raise ValueError(f"unknown mode: {mode}")

    # Row-wise unit normalization. Clamp to avoid divide-by-zero for the rare
    # all-zero row.
    norms = X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X = X / norms
    return X.to(cfg.device)


# ---------- Model ----------

class ReLUOutputModel(nn.Module):
    """
    h      = W x
    x_hat  = ReLU(W^T h + b)
    Tied weights: the same W is used to compress and to reconstruct.
    """
    def __init__(self, n: int, m: int):
        super().__init__()
        W = torch.empty(m, n)
        nn.init.xavier_uniform_(W)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.T                         # (batch, m)
        x_hat = F.relu(h @ self.W + self.b)      # (batch, n)
        return x_hat


# ---------- LR schedule ----------

def lr_at(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return cfg.lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# ---------- Training ----------

def train_one(T: int, mode: str, cfg: Config, seed: int = 0):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    X = sample_dataset(T, mode, cfg, rng)  # (T, n), fixed across all steps
    model = ReLUOutputModel(cfg.n, cfg.m).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    loss_log = []
    for step in range(cfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, cfg)

        x_hat = model(X)
        loss = ((X - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == cfg.steps - 1:
            loss_log.append((step, float(loss.item())))

    return model, loss_log


# ---------- Diagnostics ----------

def pair_cosine(model: ReLUOutputModel) -> np.ndarray:
    """
    Cosine similarity between W[:, 2k] and W[:, 2k+1] for every pair k.
    Returns shape (n/2,).
    """
    W = model.W.detach().cpu().numpy()   # (m, n)
    even = W[:, 0::2]
    odd = W[:, 1::2]
    num = (even * odd).sum(axis=0)
    den = (np.linalg.norm(even, axis=0)
           * np.linalg.norm(odd, axis=0)).clip(min=1e-12)
    return num / den


# ---------- Sweep ----------

def run_sweep(mode: str, cfg: Config):
    out = {}
    print(f"\n=== sweep: {mode} ===")
    for T in cfg.T_sweep:
        model, log = train_one(T, mode, cfg, seed=cfg.seed)
        cp = pair_cosine(model)
        out[T] = {
            "final_loss": log[-1][1],
            "loss_log": log,
            "cos_pair": cp,
            "W": model.W.detach().cpu().numpy(),
        }
        print(f"  T={T:>6d}  "
              f"final_loss={log[-1][1]:.5f}  "
              f"mean_cos={cp.mean():+.3f}  "
              f"median_cos={np.median(cp):+.3f}")
    return out


# ---------- Plotting ----------

def plot_results(res_ind, res_exc, cfg, save_path="exclusion_pairs_results.png"):
    T_vals = list(cfg.T_sweep)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Final loss vs T
    axes[0].plot(T_vals, [res_ind[T]["final_loss"] for T in T_vals],
                 "o-", label="independent")
    axes[0].plot(T_vals, [res_exc[T]["final_loss"] for T in T_vals],
                 "s-", label="exclusion pairs")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("T (training set size)")
    axes[0].set_ylabel("final train loss (MSE)")
    axes[0].set_title("Loss vs dataset size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Pair-cosine histogram at a representative T (middle of the sweep)
    T_ref = T_vals[len(T_vals) // 2]
    axes[1].hist(res_ind[T_ref]["cos_pair"], bins=50, alpha=0.5,
                 label="independent", density=True)
    axes[1].hist(res_exc[T_ref]["cos_pair"], bins=50, alpha=0.5,
                 label="exclusion pairs", density=True)
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].axvline(-1, color="r", lw=0.5, ls="--")
    axes[1].set_xlabel(f"cos(W[:, 2k], W[:, 2k+1])  at T={T_ref}")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"Pair-cosine distribution at T={T_ref}")
    axes[1].legend()

    # 3. Mean pair cosine vs T
    mean_ind = [res_ind[T]["cos_pair"].mean() for T in T_vals]
    mean_exc = [res_exc[T]["cos_pair"].mean() for T in T_vals]
    axes[2].plot(T_vals, mean_ind, "o-", label="independent")
    axes[2].plot(T_vals, mean_exc, "s-", label="exclusion pairs")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].axhline(-1, color="r", lw=0.5, ls="--",
                    label="perfect anti-parallel")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("mean cos(pair)")
    axes[2].set_title("Mean pair cosine vs T")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\nSaved plot: {save_path}")
    plt.show()


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Superposition / exclusion-pairs toy experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=1000,
                   help="input feature dimension")
    p.add_argument("--m", type=int, default=4,
                   help="hidden dim (bottleneck). Small values (2-6) force "
                        "superposition choices; large values remove the "
                        "packing pressure entirely.")
    p.add_argument("--p-active", type=float, default=0.01,
                   help="per-feature activation probability (independent "
                        "mode). Pair mode matches expected active count.")
    p.add_argument("--T-sweep", type=str,
                   default="10,30,100,300,1000,3000,10000,30000",
                   help="comma-separated list of dataset sizes")
    p.add_argument("--steps", type=int, default=20_000,
                   help="number of full-batch optimizer steps per T")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="peak learning rate (after warmup)")
    p.add_argument("--warmup-steps", type=int, default=1_000,
                   help="linear warmup steps")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="AdamW weight decay. Low values favor double descent.")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for data and model init")
    p.add_argument("--mode", choices=["independent", "exclusion_pairs", "both"],
                   default="both",
                   help="which condition(s) to run")
    p.add_argument("--output", type=str, default="exclusion_pairs_results.png",
                   help="path to save the comparison plot (only used when "
                        "mode=both)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu. Auto-detect if unset.")
    return p.parse_args()


def cfg_from_args(args) -> Config:
    T_sweep = tuple(int(x.strip()) for x in args.T_sweep.split(",") if x.strip())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Config(
        n=args.n,
        m=args.m,
        p_active=args.p_active,
        T_sweep=T_sweep,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )


# ---------- Main ----------

if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_from_args(args)

    print(f"device: {cfg.device}")
    print(f"config: n={cfg.n}, m={cfg.m}, p_active={cfg.p_active}, "
          f"steps={cfg.steps}, weight_decay={cfg.weight_decay}, seed={cfg.seed}")
    print(f"T_sweep: {cfg.T_sweep}")
    print(f"mode: {args.mode}")

    res_ind = None
    res_exc = None

    if args.mode in ("independent", "both"):
        res_ind = run_sweep("independent", cfg)
    if args.mode in ("exclusion_pairs", "both"):
        res_exc = run_sweep("exclusion_pairs", cfg)

    if args.mode == "both":
        plot_results(res_ind, res_exc, cfg, save_path=args.output)
    else:
        print("\nSingle-mode run; skipping comparison plot. "
              "Re-run with --mode both for the overlay.")
