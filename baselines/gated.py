"""
Correctness-gated training. Simple version.

The only difference from baseline.py: tokens that the model already predicts
correctly (top-1 by default) are masked out of the loss before the backward
pass. No memory system, no spacing. Everything else identical.

The intervention implements the cognitive-science intuition: don't keep
drilling things you already get right. Once the easy tasks are solved, they
stop pulling on the representational geometry, which (we hypothesize) leaves
more room for the hard tasks to shape representations during their own
critical learning windows.

Usage:
    # Single run
    python gated.py

    # Compare across seeds, same config as baseline sweep
    python gated.py --seeds 0,1,2,3 --log_root runs/gated_seeds

    # Ablation: gate off (should reproduce baseline)
    python gated.py --gate_off --seeds 0,1,2,3 --log_root runs/gate_off_sanity

    # Sweep top_k (stricter vs looser gate)
    python gated.py --seeds 0,1,2 --sweep_top_k 1,3,5 --log_root runs/topk_sweep

Key diagnostic: per-task `grad_frac` in the log. This is the fraction of
answer tokens that actually contributed gradient at each eval window. For the
gate to be doing anything, grad_frac should drop over training as tasks are
learned. If the hard-task grad_frac stays near 1.0 while easy-task grad_frac
drops toward 0, the mechanism is working as intended.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines.baseline import (
    Config,
    TASK_NAMES,
    VOCAB_SIZE,
    make_batch,
    TinyGPT,
    count_params,
    evaluate,
    get_lr,
    auto_name,
    crossing_step,
    _parse_int_list,
)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class GatedConfig(Config):
    # Token counts as "correct" (and is gated out) if the true token is in the
    # top-k predictions. top_k=1 is the strict version. Higher values treat
    # entropic tokens as correct when the true token is just near-top.
    top_k: int = 1
    # If False, this script trains identically to baseline. Used for ablation.
    gate_enabled: bool = True


def auto_name_gated(cfg: GatedConfig) -> str:
    """Prefix the baseline auto-name with gate info so outputs are unambiguous."""
    base = auto_name(cfg)
    prefix = f"gated_k{cfg.top_k}" if cfg.gate_enabled else "nogate"
    return f"{prefix}_{base}"


# =============================================================================
# TRAINING (with correctness gate)
# =============================================================================

def train_gated(cfg: GatedConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device} | gate_enabled={cfg.gate_enabled} | top_k={cfg.top_k}")

    # Probe max_len (same as baseline).
    from baselines.baseline import sample_task
    probe_rng = random.Random(0)
    max_len = 0
    for _ in range(200):
        for name in TASK_NAMES:
            ids, _ = sample_task(probe_rng, name, cfg)
            max_len = max(max_len, len(ids))
    max_len = max_len + 4
    print(f"max_len: {max_len}")

    model = TinyGPT(cfg, max_len=max_len).to(device)
    n_params = count_params(model)
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Log structure extends baseline's by adding grad_frac per task.
    log = {
        "step": [],
        "loss": [],
        "per_task_acc": {name: [] for name in TASK_NAMES},
        "per_task_loss": {name: [] for name in TASK_NAMES},
        "per_task_grad_frac": {name: [] for name in TASK_NAMES},
        "n_params": n_params,
        "gate_enabled": cfg.gate_enabled,
        "top_k": cfg.top_k,
    }

    rng = random.Random(cfg.seed + 1)
    model.train()
    t_start = time.time()
    running_loss = 0.0
    running_per_task_loss = {name: 0.0 for name in TASK_NAMES}
    running_per_task_grad_frac = {name: 0.0 for name in TASK_NAMES}
    running_per_task_count = {name: 0 for name in TASK_NAMES}

    for step in range(cfg.total_steps):
        input_ids, loss_mask, task_ids, _ = make_batch(rng, cfg)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        # Per-token loss over all positions (will mask afterwards).
        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        # ============================================================
        # THE INTERVENTION
        # ============================================================
        # Compute which answer tokens the model already gets right, and gate
        # them out of the loss. No grad contribution from "already correct"
        # positions.
        with torch.no_grad():
            if cfg.gate_enabled:
                if cfg.top_k == 1:
                    preds = logits.argmax(dim=-1)
                    correct = (preds == targets).float()
                else:
                    topk_idx = logits.topk(cfg.top_k, dim=-1).indices  # (B, T, k)
                    correct = (topk_idx == targets.unsqueeze(-1)).any(dim=-1).float()
                # gate == 1 for tokens we want to learn from, 0 for correct ones.
                gate = 1.0 - correct
            else:
                gate = torch.ones_like(mask)

        effective_mask = mask * gate
        # Normalize by kept tokens, not all answer tokens. This keeps the per-
        # token loss scale stable as more tokens get gated out over training.
        total_kept = effective_mask.sum().clamp(min=1.0)
        loss = (loss_per_tok * effective_mask).sum() / total_kept

        # ============================================================
        # DIAGNOSTICS
        # ============================================================
        # We log per-task loss on ALL answer tokens (not just kept ones) so
        # the per_task_loss curve is directly comparable to baseline. And we
        # log per-task grad_frac: the fraction of answer tokens that actually
        # contributed gradient this step.
        with torch.no_grad():
            for t_idx, name in enumerate(TASK_NAMES):
                task_filter = (task_ids == t_idx).float().unsqueeze(-1) * mask
                tf = task_filter.sum()
                if tf > 0:
                    # Mean loss on all this task's answer tokens.
                    tl = (loss_per_tok * task_filter).sum() / tf
                    running_per_task_loss[name] += tl.item()
                    # Fraction of this task's answer tokens that weren't gated.
                    kept = (task_filter * gate).sum()
                    running_per_task_grad_frac[name] += (kept / tf).item()
                    running_per_task_count[name] += 1

        lr_now = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % cfg.eval_every == 0 or step == 0:
            denom = cfg.eval_every if step > 0 else 1
            avg_loss = running_loss / max(1, denom)
            running_loss = 0.0

            accs = evaluate(model, cfg, device)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed

            log["step"].append(step + 1)
            log["loss"].append(avg_loss)
            for name in TASK_NAMES:
                log["per_task_acc"][name].append(accs[name])
                cnt = max(1, running_per_task_count[name])
                log["per_task_loss"][name].append(running_per_task_loss[name] / cnt)
                log["per_task_grad_frac"][name].append(running_per_task_grad_frac[name] / cnt)
                running_per_task_loss[name] = 0.0
                running_per_task_grad_frac[name] = 0.0
                running_per_task_count[name] = 0

            # Compact log line: include grad_frac so we can see the gate working.
            acc_str = " ".join(f"{name[:4]}:{accs[name]:.2f}" for name in TASK_NAMES)
            gf_str = " ".join(
                f"{name[:4]}:{log['per_task_grad_frac'][name][-1]:.2f}"
                for name in TASK_NAMES
            )
            print(
                f"step {step+1:6d} | loss {avg_loss:.4f} | lr {lr_now:.2e} | "
                f"{rate:.1f} st/s | acc [{acc_str}] | gfrac [{gf_str}]"
            )

            with open(log_dir / "log.json", "w") as f:
                json.dump(log, f, indent=2)

    torch.save(
        {"model_state": model.state_dict(), "config": asdict(cfg), "final_log": log},
        log_dir / "final.pt",
    )

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Final per-task accuracy:")
    for name in TASK_NAMES:
        print(f"  {name:10s}: {log['per_task_acc'][name][-1]:.4f}")

    return log


# =============================================================================
# ANALYSIS / SWEEP
# =============================================================================

def summarize_run_gated(log, cfg: GatedConfig):
    """Same shape as baseline's summarize_run but adds grad_frac summary.

    final_grad_frac: grad_frac at the last eval window. Tells you how selective
                     the gate ended up being per task.
    mean_grad_frac:  time-averaged grad_frac. Integrates over training.
    """
    s = {
        "run_name": auto_name_gated(cfg),
        "config": asdict(cfg),
        "n_params": log["n_params"],
        "gate_enabled": log.get("gate_enabled", True),
        "top_k": log.get("top_k", 1),
        "final_acc": {name: log["per_task_acc"][name][-1] for name in TASK_NAMES},
        "best_acc": {name: max(log["per_task_acc"][name]) for name in TASK_NAMES},
        "crossing_step_50": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.5)
            for name in TASK_NAMES
        },
        "crossing_step_90": {
            name: crossing_step(log["step"], log["per_task_acc"][name], 0.9)
            for name in TASK_NAMES
        },
        "final_grad_frac": {
            name: log["per_task_grad_frac"][name][-1] for name in TASK_NAMES
        },
        "mean_grad_frac": {
            name: float(np.mean(log["per_task_grad_frac"][name]))
            for name in TASK_NAMES
        },
    }
    return s


def print_summary_table_gated(summaries):
    if not summaries:
        return
    print("\n" + "=" * 90)
    print("SWEEP SUMMARY (gated)")
    print("=" * 90)

    header = f"{'run':55s}  " + "  ".join(f"{n[:6]:>6s}" for n in TASK_NAMES)

    print("\nFinal accuracy:")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = "  ".join(f"{s['final_acc'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:55s}  {vals}")

    print("\nStep at which task first reached 50% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_50"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:55s}  " + "  ".join(vals))

    print("\nStep at which task first reached 90% accuracy (None = never):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = []
        for n in TASK_NAMES:
            v = s["crossing_step_90"][n]
            vals.append(f"{v:>6d}" if v is not None else "  none")
        print(f"{s['run_name']:55s}  " + "  ".join(vals))

    print("\nMean grad_frac (fraction of answer tokens contributing gradient, avg over training):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = "  ".join(f"{s['mean_grad_frac'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:55s}  {vals}")

    print("\nFinal grad_frac (at last eval window):")
    print(header)
    print("-" * len(header))
    for s in summaries:
        vals = "  ".join(f"{s['final_grad_frac'][n]:>6.3f}" for n in TASK_NAMES)
        print(f"{s['run_name']:55s}  {vals}")
    print()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Four-task trainer with correctness gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run, top-1 gate
  python gated.py

  # Four seeds to compare against the baseline seed sweep
  python gated.py --seeds 0,1,2,3 --log_root runs/gated_seeds

  # Ablation: gate disabled (should match baseline within noise)
  python gated.py --gate_off --seeds 0,1,2,3 --log_root runs/gate_off_sanity

  # Top-k sweep: how permissive should "correct" be?
  python gated.py --seeds 0,1,2 --sweep_top_k 1,3,5 --log_root runs/topk

  # Compare capacity too
  python gated.py --seeds 0,1,2 --sweep_d_model 32,64 --log_root runs/gated_width
""",
    )
    # Baseline-identical args.
    p.add_argument("--n_layers", type=int, default=GatedConfig.n_layers)
    p.add_argument("--d_model", type=int, default=GatedConfig.d_model)
    p.add_argument("--n_heads", type=int, default=GatedConfig.n_heads)
    p.add_argument("--total_steps", type=int, default=GatedConfig.total_steps)
    p.add_argument("--batch_size", type=int, default=GatedConfig.batch_size)
    p.add_argument("--lr", type=float, default=GatedConfig.lr)
    p.add_argument("--eval_every", type=int, default=GatedConfig.eval_every)
    p.add_argument("--device", type=str, default=GatedConfig.device, choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_dir", type=str, default="runs/gated", help="Output dir for single runs.")
    p.add_argument("--log_root", type=str, default="runs/gated_sweep", help="Parent dir for sweeps.")
    p.add_argument("--seed", type=int, default=GatedConfig.seed)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--sweep_d_model", type=str, default=None)
    p.add_argument("--sweep_n_layers", type=str, default=None)
    p.add_argument("--sweep_lr", type=str, default=None)
    # Gate-specific args.
    p.add_argument("--top_k", type=int, default=GatedConfig.top_k,
                   help="Token counts as correct if true target is in top-k predictions.")
    p.add_argument("--sweep_top_k", type=str, default=None,
                   help="Comma-separated top_k values to sweep.")
    p.add_argument("--gate_off", action="store_true",
                   help="Disable the gate (train as baseline). Ablation control.")
    return p.parse_args()


def build_configs(args):
    seeds = _parse_int_list(args.seeds) if args.seeds else [args.seed]
    d_models = _parse_int_list(args.sweep_d_model) if args.sweep_d_model else [args.d_model]
    n_layers_list = _parse_int_list(args.sweep_n_layers) if args.sweep_n_layers else [args.n_layers]
    lrs = [float(x) for x in args.sweep_lr.split(",")] if args.sweep_lr else [args.lr]
    top_ks = _parse_int_list(args.sweep_top_k) if args.sweep_top_k else [args.top_k]

    is_sweep = (
        len(seeds) > 1
        or len(d_models) > 1
        or len(n_layers_list) > 1
        or len(lrs) > 1
        or len(top_ks) > 1
    )

    configs = []
    for n_layers in n_layers_list:
        for d_model in d_models:
            for lr in lrs:
                for top_k in top_ks:
                    for seed in seeds:
                        cfg = GatedConfig(
                            n_layers=n_layers,
                            d_model=d_model,
                            n_heads=args.n_heads,
                            total_steps=args.total_steps,
                            batch_size=args.batch_size,
                            lr=lr,
                            eval_every=args.eval_every,
                            device=args.device,
                            seed=seed,
                            top_k=top_k,
                            gate_enabled=not args.gate_off,
                        )
                        if is_sweep:
                            cfg.log_dir = str(Path(args.log_root) / auto_name_gated(cfg))
                        else:
                            cfg.log_dir = args.log_dir
                        configs.append(cfg)

    for cfg in configs:
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"n_heads={cfg.n_heads} must divide d_model={cfg.d_model}."
            )

    return configs, is_sweep


def main():
    args = parse_args()
    configs, is_sweep = build_configs(args)

    if is_sweep:
        log_root = Path(args.log_root)
        log_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep mode: {len(configs)} runs under {log_root}")
        for i, cfg in enumerate(configs):
            print(f"  [{i+1}/{len(configs)}] {auto_name_gated(cfg)}")
        print()

        summaries = []
        t0 = time.time()
        for i, cfg in enumerate(configs):
            print("=" * 90)
            print(f"RUN {i+1}/{len(configs)}: {auto_name_gated(cfg)}")
            print("=" * 90)
            log = train_gated(cfg)
            summaries.append(summarize_run_gated(log, cfg))
            with open(log_root / "summary.json", "w") as f:
                json.dump(summaries, f, indent=2)

        elapsed = time.time() - t0
        print(f"\nSweep complete: {len(configs)} runs in {elapsed/60:.1f} min "
              f"(avg {elapsed/len(configs)/60:.1f} min/run)")
        print_summary_table_gated(summaries)
        print(f"\nPer-run logs:      {log_root}/<run_name>/log.json")
        print(f"Aggregate summary: {log_root}/summary.json")
    else:
        cfg = configs[0]
        log = train_gated(cfg)
        summary = summarize_run_gated(log, cfg)
        with open(Path(cfg.log_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
