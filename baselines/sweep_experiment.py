"""
2D sweep over (n, m) × seed × mode for the exclusion-pairs experiment.

Holds expected active features per input constant at e_active=10 by scaling
p_active = e_active / n. This fixes the confound from the previous 1D sweep,
where fixed p_active caused E[active] to drift from 0.8 to 20 across n.

Compact JSON output: distributions are stored as 20-bin histograms rather
than raw value arrays. Live-pair statistics (pairs where both W columns have
norm >= threshold) are computed in-line since that's the correct denominator.

Imports exclusion_pairs_experiment.py, must run from the same directory.

Usage:
    python sweep_experiment.py
    python sweep_experiment.py --n-values 100,400 --m-values 4,8
    python sweep_experiment.py --seeds 0,1,2
"""

import argparse
import json
import time
from itertools import product

import numpy as np
import torch

from exclusion_pairs_experiment import (
    Config, sample_dataset, ReLUOutputModel, lr_at, pair_cosine,
)


DEAD_THRESHOLD = 0.01        # ||W[:, i]|| below this = dead feature
COS_BINS = 20                # histogram resolution for cosine distributions
NORM_BINS = 20               # histogram resolution for column norms
TRAJ_POINTS = 10             # loss trajectory samples


# ---------- Metrics ----------

def hist_as_list(values, bin_edges):
    """Return counts only. Bin edges are stored once at top level."""
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts.tolist()


def compute_metrics(model, cos_bin_edges, norm_bin_edges):
    W = model.W.detach().cpu().numpy()            # (m, n)
    m, n = W.shape

    col_norms = np.linalg.norm(W, axis=0)         # (n,)
    alive = col_norms >= DEAD_THRESHOLD

    cos_pair = pair_cosine(model)                 # (n/2,)
    even_alive = alive[0::2]
    odd_alive = alive[1::2]
    both_alive = even_alive & odd_alive
    live_cos = cos_pair[both_alive]

    # Inter-pair interference
    W_unit = W / col_norms.clip(min=1e-12)
    cos_matrix = W_unit.T @ W_unit
    pair_idx = np.arange(n) // 2
    same_pair = pair_idx[:, None] == pair_idx[None, :]
    diag = np.eye(n, dtype=bool)
    off_pair_mask = ~same_pair & ~diag
    off_abs = np.abs(cos_matrix[off_pair_mask])

    result = {
        "col_norms": {
            "mean": float(col_norms.mean()),
            "std": float(col_norms.std()),
            "max": float(col_norms.max()),
            "frac_dead": float((~alive).mean()),
            "hist": hist_as_list(col_norms, norm_bin_edges),
        },
        "pair_survival": {
            "both_alive": float(both_alive.mean()),
            "one_alive": float((even_alive ^ odd_alive).mean()),
            "both_dead": float((~even_alive & ~odd_alive).mean()),
        },
        "pair_cosine_all": {
            "n": int(len(cos_pair)),
            "mean": float(cos_pair.mean()),
            "median": float(np.median(cos_pair)),
            "std": float(cos_pair.std()),
            "hist": hist_as_list(cos_pair, cos_bin_edges),
        },
        "pair_cosine_live": {
            "n": int(both_alive.sum()),
            "mean": float(live_cos.mean()) if len(live_cos) else None,
            "median": float(np.median(live_cos)) if len(live_cos) else None,
            "std": float(live_cos.std()) if len(live_cos) else None,
            "frac_below_neg_half": (float((live_cos < -0.5).mean())
                                    if len(live_cos) else None),
            "frac_above_pos_half": (float((live_cos > 0.5).mean())
                                    if len(live_cos) else None),
            "hist": (hist_as_list(live_cos, cos_bin_edges)
                     if len(live_cos) else None),
        },
        "inter_pair_interference": {
            "mean_abs_cos": float(off_abs.mean()),
            "p95_abs_cos": float(np.percentile(off_abs, 95)),
        },
    }
    return result


def train_with_log(T, mode, cfg, seed, n_log_points):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    X = sample_dataset(T, mode, cfg, rng)
    model = ReLUOutputModel(cfg.n, cfg.m).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    log_every = max(1, cfg.steps // n_log_points)
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
            traj.append([step, float(loss.item())])  # list, saves bytes

    return model, traj


def run_condition(n, m, mode, seed, T, e_active, base_cfg,
                  cos_bin_edges, norm_bin_edges):
    p_active = e_active / n
    cfg = Config(
        n=n, m=m, p_active=p_active,
        T_sweep=(T,),
        steps=base_cfg.steps, lr=base_cfg.lr,
        warmup_steps=base_cfg.warmup_steps,
        weight_decay=base_cfg.weight_decay,
        seed=seed, device=base_cfg.device,
    )

    t0 = time.time()
    model, traj = train_with_log(T, mode, cfg, seed, TRAJ_POINTS)
    elapsed = time.time() - t0

    metrics = compute_metrics(model, cos_bin_edges, norm_bin_edges)
    final_loss = traj[-1][1]
    trivial_loss = 1.0 / n

    del model
    if base_cfg.device == "cuda":
        torch.cuda.empty_cache()

    return {
        "config": {
            "n": n, "m": m, "T": T, "mode": mode, "seed": seed,
            "p_active": p_active, "e_active": e_active,
        },
        "timing_sec": round(elapsed, 2),
        "final_loss": final_loss,
        "normalized_loss": final_loss / trivial_loss,
        "loss_trajectory": traj,
        "metrics": metrics,
    }


# ---------- Aggregation ----------

def compute_aggregates(results, n_values, m_values, modes):
    aggregates = []
    for n, m, mode in product(n_values, m_values, modes):
        matching = [r for r in results
                    if r["config"]["n"] == n
                    and r["config"]["m"] == m
                    and r["config"]["mode"] == mode]
        if not matching:
            continue

        def avg_path(path):
            vals = []
            for r in matching:
                v = r["metrics"]
                for k in path:
                    v = v[k]
                if v is not None:
                    vals.append(v)
            if not vals:
                return (None, None)
            return (float(np.mean(vals)), float(np.std(vals)))

        pcl_mean_avg, pcl_mean_std = avg_path(["pair_cosine_live", "mean"])
        pcl_med_avg, pcl_med_std = avg_path(["pair_cosine_live", "median"])
        fd_avg, fd_std = avg_path(["col_norms", "frac_dead"])
        ba_avg, ba_std = avg_path(["pair_survival", "both_alive"])

        norm_losses = [r["normalized_loss"] for r in matching]

        aggregates.append({
            "n": n, "m": m, "mode": mode,
            "n_seeds": len(matching),
            "normalized_loss": [float(np.mean(norm_losses)),
                                float(np.std(norm_losses))],
            "frac_dead": [fd_avg, fd_std],
            "frac_both_alive": [ba_avg, ba_std],
            "cos_live_mean": [pcl_mean_avg, pcl_mean_std],
            "cos_live_median": [pcl_med_avg, pcl_med_std],
        })
    return aggregates


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="2D (n x m) sweep with compact JSON output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-values", type=str, default="100,200,400,800")
    p.add_argument("--m-values", type=str, default="2,3,4,6,8,12")
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--modes", type=str,
                   default="independent,exclusion_pairs")
    p.add_argument("--T", type=int, default=10_000)
    p.add_argument("--e-active", type=float, default=10.0,
                   help="target expected active features per input")
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--output", type=str, default="sweep_results.json")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = Config(
        n=n_values[0], m=m_values[0], p_active=0.01,
        steps=args.steps, lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        device=device,
    )

    cos_bin_edges = np.linspace(-1, 1, COS_BINS + 1)
    norm_bin_edges = np.linspace(0, 2, NORM_BINS + 1)

    conditions = list(product(n_values, m_values, modes, seeds))
    total = len(conditions)

    print(f"device: {device}")
    print(f"total conditions: {total}")
    print(f"n values: {n_values}  (p_active = {args.e_active}/n)")
    print(f"m values: {m_values}")
    print(f"seeds: {seeds}")
    print(f"modes: {modes}")
    print(f"T={args.T}  steps={args.steps}  wd={args.weight_decay}")
    print()

    results = []
    t_start = time.time()
    for i, (n, m, mode, seed) in enumerate(conditions, 1):
        print(f"[{i:>3}/{total}] n={n:>4} m={m:>2} "
              f"mode={mode[:3]} seed={seed}",
              end="  ", flush=True)
        res = run_condition(n, m, mode, seed, args.T, args.e_active,
                            base_cfg, cos_bin_edges, norm_bin_edges)
        results.append(res)
        pcl = res["metrics"]["pair_cosine_live"]
        cn = res["metrics"]["col_norms"]
        cos_str = (f"cos_live={pcl['mean']:+.3f}"
                   if pcl["mean"] is not None else "cos_live=n/a")
        print(f"nloss={res['normalized_loss']:.3f}  "
              f"dead={cn['frac_dead']:.2f}  "
              f"n_live={pcl['n']:>4}  "
              f"{cos_str}  "
              f"t={res['timing_sec']:.1f}s")

    total_time = time.time() - t_start

    aggregates = compute_aggregates(results, n_values, m_values, modes)

    output = {
        "sweep_config": {
            "n_values": n_values,
            "m_values": m_values,
            "seeds": seeds,
            "modes": modes,
            "T": args.T,
            "e_active": args.e_active,
            "steps": args.steps,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "device": device,
        },
        "hist_bin_edges": {
            "cosine": cos_bin_edges.tolist(),
            "col_norm": norm_bin_edges.tolist(),
        },
        "total_sweep_time_sec": round(total_time, 1),
        "num_conditions": len(results),
        "aggregates": aggregates,
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, separators=(",", ":"))  # no whitespace

    import os
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\ntotal sweep time: {total_time:.1f}s")
    print(f"saved: {args.output} ({size_kb:.0f} KB)")

    # Print summary grids
    print("\n=== exclusion MINUS independent  live-pair mean cos ===")
    print(f"{'':>9}", end="")
    for m in m_values:
        print(f"  m={m:>2}", end="")
    print()
    for n in n_values:
        print(f"  n={n:>5}", end="")
        for m in m_values:
            exc = next((a for a in aggregates
                       if a["n"] == n and a["m"] == m
                       and a["mode"] == "exclusion_pairs"), None)
            ind = next((a for a in aggregates
                       if a["n"] == n and a["m"] == m
                       and a["mode"] == "independent"), None)
            e_val = exc["cos_live_mean"][0] if exc else None
            i_val = ind["cos_live_mean"][0] if ind else None
            if e_val is not None and i_val is not None:
                diff = e_val - i_val
                print(f"  {diff:+5.2f}", end="")
            else:
                print(f"   n/a", end="")
        print()

    print("\n=== frac_dead  (exclusion condition) ===")
    print(f"{'':>9}", end="")
    for m in m_values:
        print(f"  m={m:>2}", end="")
    print()
    for n in n_values:
        print(f"  n={n:>5}", end="")
        for m in m_values:
            exc = next((a for a in aggregates
                       if a["n"] == n and a["m"] == m
                       and a["mode"] == "exclusion_pairs"), None)
            if exc and exc["frac_dead"][0] is not None:
                print(f"  {exc['frac_dead'][0]:>5.2f}", end="")
            else:
                print(f"   n/a", end="")
        print()


if __name__ == "__main__":
    main()
