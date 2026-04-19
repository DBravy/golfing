"""
Latent-space feature experiment.

Each feature i has a fixed latent position v_i in R^d. To generate an input:
    1. Sample context vector c ~ N(0, I_d)
    2. Compute affinity a_i = <c, v_i> for every feature
    3. Select top-K features by affinity
    4. Assign magnitudes via softmax over the K chosen affinities
    5. Unit-normalize the resulting sparse input

This preserves sparsity (exactly K features active per input) while giving every
feature a relationship with every other feature: they all compete for the same K
slots against a shared context. Closer features in v-space tend to co-fire;
opposite features rarely do. This generalizes exclusion pairs (pairs = v_i on
antipodal points) and gives the continuous gradation we want for "language-like"
interaction structure.

Two conditions compared by default:
    circle: v_i evenly spaced on unit circle (d=2), maximally structured
    random: v_i random unit vectors in d dimensions, diffuse structure

Main diagnostic: does W's column geometry match v's geometry? Measured via
Pearson correlation between the pairwise cosine-distance matrices of v_alive
and W_alive. High r = the model learned the latent structure. Near 0 = it
didn't.

Imports exclusion_pairs_experiment.py for shared model and optimizer infra.

Usage:
    python latent_experiment.py
    python latent_experiment.py --n 200 --m 8 --K 15
    python latent_experiment.py --structures circle --d 2
    python latent_experiment.py --structures circle,random --temperature 2.0
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from baselines.exclusion_pairs_experiment import Config, ReLUOutputModel, lr_at


DEAD_THRESHOLD = 0.01


# ---------- Latent-space generation ----------

def generate_v(n: int, d: int, structure: str,
               rng: torch.Generator) -> torch.Tensor:
    """
    Fixed latent positions v_i in R^d, one per feature.

    circle: evenly spaced on unit circle. Only meaningful for d=2.
            For d>2, falls back to random unit vectors.
    random: iid random unit vectors in R^d.
    """
    if structure == "circle" and d == 2:
        angles = torch.linspace(0, 2 * np.pi, n + 1)[:-1]
        v = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    elif structure == "circle":
        # d != 2; fall back to random unit sphere
        v = torch.randn(n, d, generator=rng)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    elif structure == "random":
        v = torch.randn(n, d, generator=rng)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        raise ValueError(f"unknown structure: {structure}")
    return v


def sample_latent_dataset(T: int, K: int, v: torch.Tensor, temperature: float,
                          rng: torch.Generator, device: str) -> torch.Tensor:
    """
    Sample T inputs from the latent-space distribution.

    v: (n, d) fixed latent positions
    K: exact number of features active per input
    temperature: scales affinities before softmax. 1.0 = neutral. Higher =
                 more peaked magnitudes on the best-match feature.

    Returns X of shape (T, n), row-normalized to unit length.
    """
    n, d = v.shape
    C = torch.randn(T, d, generator=rng)           # (T, d)
    A = C @ v.T                                    # (T, n) affinities
    topk_vals, topk_idx = A.topk(K, dim=1)         # (T, K)
    weights = F.softmax(topk_vals * temperature, dim=1)   # (T, K)

    X = torch.zeros(T, n)
    X.scatter_(1, topk_idx, weights)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return X.to(device)


# ---------- Training ----------

def train_one(cfg: Config, K: int, d: int, structure: str,
              T: int, temperature: float, seed: int):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    v = generate_v(cfg.n, d, structure, rng)       # (n, d), CPU
    X = sample_latent_dataset(T, K, v, temperature, rng, cfg.device)

    model = ReLUOutputModel(cfg.n, cfg.m).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    log_every = max(1, cfg.steps // 20)
    traj = []
    for step in range(cfg.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, cfg)
        x_hat = model(X)
        loss = ((X - x_hat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == cfg.steps - 1:
            traj.append((step, float(loss.item())))

    return model, traj, v.cpu().numpy()


# ---------- Diagnostic ----------

def latent_recovery(model: ReLUOutputModel, v: np.ndarray):
    """
    Compare the pairwise cosine-distance matrix of v's rows (features in
    latent space) to the pairwise cosine-distance matrix of W's columns
    (features in hidden space). High correlation = structure recovered.

    Computed over alive features only.
    """
    W = model.W.detach().cpu().numpy()             # (m, n)
    col_norms = np.linalg.norm(W, axis=0)
    alive = col_norms >= DEAD_THRESHOLD
    n_alive = int(alive.sum())

    result = {
        "n_alive": n_alive,
        "frac_dead": float((~alive).mean()),
        "pearson_r": None,
    }
    if n_alive < 3:
        return result

    W_a = W[:, alive].T                            # (n_alive, m)
    v_a = v[alive]                                  # (n_alive, d)

    W_u = W_a / np.linalg.norm(W_a, axis=1, keepdims=True).clip(min=1e-12)
    v_u = v_a / np.linalg.norm(v_a, axis=1, keepdims=True).clip(min=1e-12)

    D_W = 1 - W_u @ W_u.T
    D_v = 1 - v_u @ v_u.T

    iu = np.triu_indices(n_alive, k=1)
    result["pearson_r"] = float(np.corrcoef(D_W[iu], D_v[iu])[0, 1])
    return result


# ---------- Plotting ----------

def project_to_2d(W: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """
    Project W columns (shape (m, n)) to 2D. Uses the first two principal
    components of the alive columns; falls back gracefully if m <= 2.
    """
    m, n = W.shape
    if m == 2:
        return W.T.copy()
    if alive.sum() < 2:
        return np.zeros((n, 2))
    Wa = W[:, alive]
    mean = Wa.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Wa - mean, full_matrices=False)
    return ((W - mean).T @ U[:, :2])


def plot_geometry(v: np.ndarray, model: ReLUOutputModel,
                  structure: str, recovery_r, save_path: str):
    W = model.W.detach().cpu().numpy()
    col_norms = np.linalg.norm(W, axis=0)
    alive = col_norms >= DEAD_THRESHOLD

    # Color features by their angular position in v (for circle structure this
    # makes the 1D ordering visible in both plots).
    if v.shape[1] >= 2:
        hue = (np.arctan2(v[:, 1], v[:, 0]) + np.pi) / (2 * np.pi)
    else:
        hue = np.linspace(0, 1, v.shape[0])

    W_2d = project_to_2d(W, alive)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # v positions (first two dims)
    v_2d = v[:, :2] if v.shape[1] >= 2 else np.column_stack([v[:, 0], np.zeros(len(v))])
    axes[0].scatter(v_2d[~alive, 0], v_2d[~alive, 1], c="lightgray",
                    s=15, alpha=0.5, label="dead")
    axes[0].scatter(v_2d[alive, 0], v_2d[alive, 1], c=hue[alive],
                    cmap="hsv", s=40, alpha=0.8, label="alive")
    axes[0].set_title(f"Latent v (first 2 dims)")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # W 2D projection
    axes[1].scatter(W_2d[~alive, 0], W_2d[~alive, 1], c="lightgray",
                    s=15, alpha=0.5)
    axes[1].scatter(W_2d[alive, 0], W_2d[alive, 1], c=hue[alive],
                    cmap="hsv", s=40, alpha=0.8)
    r_str = f"{recovery_r:+.2f}" if recovery_r is not None else "n/a"
    axes[1].set_title(f"Learned W (2D PCA, m={W.shape[0]}) — r={r_str}")
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # Cosine matrix of W columns, reordered by v's angular position
    if v.shape[1] >= 2 and alive.sum() >= 2:
        order = np.argsort(np.arctan2(v[:, 1], v[:, 0]))
        # Restrict to alive, then reorder
        alive_order = np.array([i for i in order if alive[i]])
        Wa = W[:, alive_order]
        Wa_u = Wa / np.linalg.norm(Wa, axis=0, keepdims=True).clip(min=1e-12)
        cos_mat = Wa_u.T @ Wa_u
        im = axes[2].imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[2].set_title("cos(W cols), ordered by angle of v")
        plt.colorbar(im, ax=axes[2], fraction=0.046)
    else:
        axes[2].axis("off")

    fig.suptitle(f"structure = {structure}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"  saved plot: {save_path}")
    plt.close()


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Latent-space feature experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--m", type=int, default=4)
    p.add_argument("--K", type=int, default=10,
                   help="features active per input (exact)")
    p.add_argument("--d", type=int, default=2,
                   help="latent dimension for v")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="multiplies affinities before softmax. Higher = more"
                        " peaked magnitudes on the best-match feature.")
    p.add_argument("--T", type=int, default=10_000)
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--structures", type=str, default="circle,random",
                   help="comma-separated list of latent structures to run")
    p.add_argument("--output-json", type=str, default="latent_results.json")
    p.add_argument("--plot-prefix", type=str, default="latent",
                   help="prefix for saved plots (one per structure)")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    structures = [s.strip() for s in args.structures.split(",") if s.strip()]

    cfg = Config(
        n=args.n, m=args.m,
        steps=args.steps, lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    print(f"device: {device}")
    print(f"config: n={cfg.n}, m={cfg.m}, K={args.K}, d={args.d}, "
          f"temperature={args.temperature}")
    print(f"        T={args.T}, steps={cfg.steps}, wd={cfg.weight_decay}")
    print(f"structures: {structures}")
    print()

    results = {}
    for structure in structures:
        print(f"=== {structure} ===")
        t0 = time.time()
        model, traj, v = train_one(cfg, args.K, args.d, structure,
                                    args.T, args.temperature, args.seed)
        elapsed = time.time() - t0
        rec = latent_recovery(model, v)

        final_loss = traj[-1][1]
        trivial = 1.0 / args.n

        entry = {
            "config": {
                "n": cfg.n, "m": cfg.m, "K": args.K, "d": args.d,
                "T": args.T, "steps": cfg.steps,
                "temperature": args.temperature,
                "structure": structure, "seed": args.seed,
                "weight_decay": cfg.weight_decay,
            },
            "timing_sec": round(elapsed, 2),
            "final_loss": final_loss,
            "trivial_loss": trivial,
            "normalized_loss": final_loss / trivial,
            "recovery": rec,
            "loss_trajectory": [[int(s), float(l)] for s, l in traj],
        }
        results[structure] = entry

        r_str = f"{rec['pearson_r']:+.3f}" if rec["pearson_r"] is not None else "n/a"
        print(f"  time: {elapsed:.1f}s")
        print(f"  final_loss: {final_loss:.5f}  "
              f"normalized: {entry['normalized_loss']:.3f}")
        print(f"  n_alive: {rec['n_alive']}/{cfg.n}  "
              f"(frac_dead: {rec['frac_dead']:.2f})")
        print(f"  recovery r (W-dist vs v-dist): {r_str}")

        plot_path = f"{args.plot_prefix}_{structure}.png"
        plot_geometry(v, model, structure, rec["pearson_r"], plot_path)
        print()

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved JSON: {args.output_json}")

    print("\n=== summary ===")
    print(f"{'structure':<10}  {'recovery_r':>11}  {'norm_loss':>10}  {'frac_dead':>10}")
    for s, r in results.items():
        pr = r["recovery"]["pearson_r"]
        pr_s = f"{pr:+.3f}" if pr is not None else "n/a"
        print(f"{s:<10}  {pr_s:>11}  "
              f"{r['normalized_loss']:>10.3f}  "
              f"{r['recovery']['frac_dead']:>10.3f}")


if __name__ == "__main__":
    main()
