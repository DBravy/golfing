"""
Residual-stream RNN probe, Dreamer-v3 style.

Trains a small recurrent model that observes the transformer's layer-1 residual
stream (detached) and, for each token position, predicts:
    (a) the layer-2 residual at that same position, and
    (b) the transformer's per-token cross-entropy loss at that position.

Architecture per position t (along the sequence):
    inputs : r1_t  (layer-1 residual, detached)
             h_{t-1} (previous hidden state)
    encode : z_t = sparse_latent(concat(r1_t, h_{t-1}))
                   8 categoricals x 8 classes, straight-through one-hot
    update : h_t = GRUCell(z_t, h_{t-1})
    heads  : feature = concat(z_t, h_t)
             r2_hat_t   = decoder(feature)        # predicts r2_t (MSE)
             loss_hat_t = loss_head(feature)      # predicts CE loss (MSE)

The transformer is trained normally. The RNN is a parasitic observer: residual
tensors are detached before entering the RNN, so RNN gradients do not flow
back into the transformer.

Run: python rnn_probe.py
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
import torch.nn as nn
import torch.nn.functional as F

import baselines.baseline as baseline
from baselines.baseline import (
    Config,
    TinyGPT,
    VOCAB_SIZE,
    PAD_ID,
    TASK_NAMES,
    make_batch,
    evaluate,
    get_lr,
    count_params,
    auto_name,
)


# =============================================================================
# RNN PROBE CONFIG
# =============================================================================

@dataclass
class ProbeConfig:
    # Sparse latent: n_categoricals categoricals, each with n_classes classes.
    # Total latent dim is n_categoricals * n_classes (one-hot concatenated).
    n_categoricals: int = 8
    n_classes: int = 8

    # Hidden state dimensionality of the GRU.
    d_hidden: int = 64

    # Width of the encoder / decoder MLPs.
    d_mlp: int = 128

    # Loss weights.
    w_recon: float = 1.0   # residual reconstruction
    w_loss: float = 1.0    # transformer-loss prediction

    # Optimiser (separate from the transformer's).
    probe_lr: float = 3e-4
    probe_weight_decay: float = 0.0

    # Which transformer layer index feeds the RNN (input side).
    # Layer 0 output == layer 1 input. For n_layers=2 we want input=0, target=1.
    source_layer: int = 0   # residual after block[source_layer]
    target_layer: int = 1   # residual after block[target_layer]

    # If True, probe gradients flow back into the transformer, turning the
    # probe's objective into an auxiliary regulariser. The source residual is
    # no longer detached, and the target residual is also left attached so
    # the transformer is pushed to make layer-(target_layer) predictable from
    # layer-(source_layer) via the sparse-RNN bottleneck.
    # When False (default), probe is a passive observer.
    attach: bool = False
    # When attach=True, scales how strongly the probe loss affects the
    # transformer. 1.0 = equal weight with the CE loss. Tune down if the
    # auxiliary objective dominates.
    w_probe_to_xform: float = 1.0


# =============================================================================
# TRANSFORMER WITH RESIDUAL CAPTURE
# =============================================================================
# We need the per-layer residual stream. TinyGPT.forward only returns logits,
# so we subclass it and expose the intermediate residuals.

class TinyGPTWithResiduals(TinyGPT):
    """Same as TinyGPT but forward() also returns the residual after each block.

    residuals[i] = hidden state after block i (pre final LayerNorm).
    residuals has length n_layers. residuals[-1] is what feeds ln_f.
    """

    def forward_with_residuals(self, ids):
        B, T = ids.shape
        pos = self._pos_ids[:, :T]
        x = self.tok_emb(ids) + self.pos_emb(pos)
        residuals = []
        for blk in self.blocks:
            x = blk(x)
            residuals.append(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T
        return logits, residuals


# =============================================================================
# SPARSE LATENT (Dreamer v3 style categorical with straight-through)
# =============================================================================

class SparseLatent(nn.Module):
    """Produces n_categoricals x n_classes one-hot latent.

    Forward returns a flat tensor of shape (..., n_categoricals * n_classes).
    Uses straight-through: forward is argmax one-hot, backward is softmax.

    Dreamer v3 mixes the softmax with a uniform for robustness. We include
    that (unimix=0.01) but keep it small.
    """

    def __init__(self, d_in: int, n_categoricals: int, n_classes: int,
                 d_hidden: int = 128, unimix: float = 0.01):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.unimix = unimix
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_categoricals * n_classes),
        )

    @property
    def out_dim(self):
        return self.n_categoricals * self.n_classes

    def forward(self, x):
        # x: (..., d_in)
        logits = self.net(x)
        shape = logits.shape[:-1] + (self.n_categoricals, self.n_classes)
        logits = logits.reshape(shape)

        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0:
            uniform = torch.ones_like(probs) / self.n_classes
            probs = (1 - self.unimix) * probs + self.unimix * uniform

        # Straight-through: hard one-hot in forward, soft probs in backward.
        idx = probs.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0)
        z = hard + probs - probs.detach()

        return z.reshape(*shape[:-2], self.n_categoricals * self.n_classes)


# =============================================================================
# RESIDUAL RNN
# =============================================================================

class ResidualRNN(nn.Module):
    """Scans a batch of residual sequences left-to-right, one step per token.

    At each step:
        z_t = SparseLatent(concat(r1_t, h_{t-1}))
        h_t = GRUCell(z_t, h_{t-1})
        feature_t = concat(z_t, h_t)
        r2_hat_t  = decoder(feature_t)
        loss_hat_t = loss_head(feature_t)   # scalar
    """

    def __init__(self, d_model: int, pcfg: ProbeConfig):
        super().__init__()
        self.pcfg = pcfg
        self.d_model = d_model
        self.d_hidden = pcfg.d_hidden

        self.latent = SparseLatent(
            d_in=d_model + pcfg.d_hidden,
            n_categoricals=pcfg.n_categoricals,
            n_classes=pcfg.n_classes,
            d_hidden=pcfg.d_mlp,
        )
        d_z = self.latent.out_dim

        self.cell = nn.GRUCell(input_size=d_z, hidden_size=pcfg.d_hidden)

        feat_dim = d_z + pcfg.d_hidden
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, pcfg.d_mlp),
            nn.GELU(),
            nn.Linear(pcfg.d_mlp, d_model),
        )
        self.loss_head = nn.Sequential(
            nn.Linear(feat_dim, pcfg.d_mlp),
            nn.GELU(),
            nn.Linear(pcfg.d_mlp, 1),
        )

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(batch_size, self.d_hidden, device=device)

    def forward(self, r1_seq):
        """r1_seq: (B, T, d_model) detached layer-1 residuals.

        Returns:
            r2_hat:   (B, T, d_model)
            loss_hat: (B, T)
            z_seq:    (B, T, d_z)   # for optional logging
        """
        B, T, _ = r1_seq.shape
        h = self.init_hidden(B, r1_seq.device)
        r2_hats, loss_hats, zs = [], [], []
        for t in range(T):
            r1_t = r1_seq[:, t, :]
            inp = torch.cat([r1_t, h], dim=-1)
            z = self.latent(inp)
            h = self.cell(z, h)
            feat = torch.cat([z, h], dim=-1)
            r2_hats.append(self.decoder(feat))
            loss_hats.append(self.loss_head(feat).squeeze(-1))
            zs.append(z)
        r2_hat = torch.stack(r2_hats, dim=1)
        loss_hat = torch.stack(loss_hats, dim=1)
        z_seq = torch.stack(zs, dim=1)
        return r2_hat, loss_hat, z_seq


# =============================================================================
# TRAINING
# =============================================================================

def train(cfg: Config, pcfg: ProbeConfig, log_name: str = "log"):
    """Train transformer + probe.

    log_name: base filename (without .json) for the running log. Default 'log'
    writes to {log_dir}/log.json. Sweep driver uses e.g. 'detached_log' to
    match the existing folder/file naming convention.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # Probe max_len exactly as baseline does.
    probe_rng = random.Random(0)
    max_len = 0
    for _ in range(200):
        for name in TASK_NAMES:
            ids, _ = baseline.sample_task(probe_rng, name, cfg)
            max_len = max(max_len, len(ids))
    max_len = max_len + 4
    print(f"max_len: {max_len}")

    # Require the source / target layers to be valid.
    assert 0 <= pcfg.source_layer < cfg.n_layers, "source_layer out of range"
    assert 0 <= pcfg.target_layer < cfg.n_layers, "target_layer out of range"

    model = TinyGPTWithResiduals(cfg, max_len=max_len).to(device)
    probe = ResidualRNN(d_model=cfg.d_model, pcfg=pcfg).to(device)

    n_params = count_params(model)
    n_probe_params = count_params(probe)
    print(f"Transformer params: {n_params:,}")
    print(f"Probe params:       {n_probe_params:,}")

    # Separate optimisers so the probe can't touch the transformer and vice
    # versa. Matches baseline.train() for the transformer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    probe_optimizer = torch.optim.AdamW(
        probe.parameters(), lr=pcfg.probe_lr, weight_decay=pcfg.probe_weight_decay,
        betas=(0.9, 0.95),
    )

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.json", "w") as f:
        json.dump({"baseline": asdict(cfg), "probe": asdict(pcfg)}, f, indent=2)

    log = {
        "step": [],
        "loss": [],               # transformer CE on answer tokens
        "probe_recon": [],        # MSE of r2_hat vs r2 (answer tokens)
        "probe_loss_mse": [],     # MSE of loss_hat vs per-tok CE (answer tokens)
        "probe_explained_var": [],  # 1 - MSE/Var(target); ~R^2 on r2 (answer tokens)
        "per_task_acc": {name: [] for name in TASK_NAMES},
        "per_task_loss": {name: [] for name in TASK_NAMES},
        "n_params": n_params,
        "n_probe_params": n_probe_params,
    }

    rng = random.Random(cfg.seed + 1)
    model.train()
    probe.train()
    t_start = time.time()
    running_loss = 0.0
    running_recon = 0.0
    running_loss_mse = 0.0
    running_explained_var = 0.0
    running_per_task_loss = {name: 0.0 for name in TASK_NAMES}
    running_per_task_count = {name: 0 for name in TASK_NAMES}

    for step in range(cfg.total_steps):
        input_ids, loss_mask, task_ids, _ = make_batch(rng, cfg)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        # Forward through transformer with residual capture.
        # We need residuals aligned with the *input* positions [:, :-1] so
        # they match the logits and per-token losses.
        logits, residuals = model.forward_with_residuals(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        masked = loss_per_tok * mask
        total_masked = mask.sum().clamp(min=1.0)
        xform_loss = masked.sum() / total_masked

        # Per-task transformer loss (logging only).
        with torch.no_grad():
            for t_idx, name in enumerate(TASK_NAMES):
                task_mask = (task_ids == t_idx).float().unsqueeze(-1) * mask
                tm_sum = task_mask.sum()
                if tm_sum > 0:
                    tl = (loss_per_tok * task_mask).sum() / tm_sum
                    running_per_task_loss[name] += tl.item()
                    running_per_task_count[name] += 1

        # --- probe forward ---
        # In detached mode (default), probe gradients cannot reach the
        # transformer. In attach mode, gradients flow from the probe's loss
        # back through source and target residuals into the transformer,
        # making the probe's objective an auxiliary regulariser.
        if pcfg.attach:
            r_src = residuals[pcfg.source_layer]
            r_tgt = residuals[pcfg.target_layer]
            # Loss-prediction target stays detached: we don't want the
            # transformer gaming its own CE by making it predictable.
            # The representation-level pressure comes from recon alone.
            loss_target = loss_per_tok.detach()
        else:
            r_src = residuals[pcfg.source_layer].detach()
            r_tgt = residuals[pcfg.target_layer].detach()
            loss_target = loss_per_tok.detach()

        r2_hat, loss_hat, _ = probe(r_src)

        # Reconstruction + loss-prediction losses, averaged over answer tokens.
        answer_mask = mask  # (B, T)
        denom = answer_mask.sum().clamp(min=1.0)

        recon_per_tok = ((r2_hat - r_tgt) ** 2).mean(dim=-1)  # (B, T)
        recon_loss = (recon_per_tok * answer_mask).sum() / denom

        # Diagnostic: fraction of r_tgt variance captured by the probe.
        # recon_loss above is per-element MSE (averaged over d_model and
        # answer tokens). The matching target variance is per-element too:
        # mean over d_model of r_tgt variance computed across answer tokens.
        # We subtract the per-element mean of r_tgt (masked) before squaring
        # so this is actual variance, not second moment, matching standard R^2.
        with torch.no_grad():
            r_tgt_det = r_tgt.detach()
            # Per-element mean of r_tgt across answer tokens: shape (d_model,)
            tgt_sum = (r_tgt_det * answer_mask.unsqueeze(-1)).sum(dim=(0, 1))
            tgt_mean = tgt_sum / denom  # (d_model,)
            # Per-token squared deviation, averaged over d_model -> (B, T)
            tgt_var_per_tok = ((r_tgt_det - tgt_mean) ** 2).mean(dim=-1)
            tgt_var = (tgt_var_per_tok * answer_mask).sum() / denom
            # Explained variance = 1 - MSE / Var(target). Clamped to [-1, 1]
            # for stability early in training when both quantities are tiny.
            explained_var = 1.0 - (recon_loss.detach() / tgt_var.clamp(min=1e-8))
            explained_var = explained_var.clamp(min=-1.0, max=1.0)

        loss_mse_per_tok = (loss_hat - loss_target) ** 2  # (B, T)
        loss_mse = (loss_mse_per_tok * answer_mask).sum() / denom

        probe_loss = pcfg.w_recon * recon_loss + pcfg.w_loss * loss_mse

        # --- backward + step ---
        lr_now = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        if pcfg.attach:
            # Single combined backward. The transformer gets CE + scaled
            # probe loss; the probe gets its own full loss through the same
            # graph. Both optimisers step on their own param groups.
            total_loss = xform_loss + pcfg.w_probe_to_xform * probe_loss
            optimizer.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
            optimizer.step()
            probe_optimizer.step()
        else:
            # Detached: two independent backwards, probe cannot touch the
            # transformer (verified).
            optimizer.zero_grad(set_to_none=True)
            xform_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            probe_optimizer.zero_grad(set_to_none=True)
            probe_loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), cfg.grad_clip)
            probe_optimizer.step()

        running_loss += xform_loss.item()
        running_recon += recon_loss.item()
        running_loss_mse += loss_mse.item()
        running_explained_var += explained_var.item()

        if (step + 1) % cfg.eval_every == 0 or step == 0:
            denom_log = cfg.eval_every if step > 0 else 1
            avg_loss = running_loss / denom_log
            avg_recon = running_recon / denom_log
            avg_loss_mse = running_loss_mse / denom_log
            avg_explained_var = running_explained_var / denom_log
            running_loss = 0.0
            running_recon = 0.0
            running_loss_mse = 0.0
            running_explained_var = 0.0

            accs = evaluate(model, cfg, device)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed

            log["step"].append(step + 1)
            log["loss"].append(avg_loss)
            log["probe_recon"].append(avg_recon)
            log["probe_loss_mse"].append(avg_loss_mse)
            log["probe_explained_var"].append(avg_explained_var)
            for name in TASK_NAMES:
                log["per_task_acc"][name].append(accs[name])
                if running_per_task_count[name] > 0:
                    ptl = running_per_task_loss[name] / running_per_task_count[name]
                else:
                    ptl = float("nan")
                log["per_task_loss"][name].append(ptl)
                running_per_task_loss[name] = 0.0
                running_per_task_count[name] = 0

            acc_str = " ".join(f"{name[:4]}:{accs[name]:.2f}" for name in TASK_NAMES)
            print(f"step {step+1:6d} | loss {avg_loss:.4f} | recon {avg_recon:.4f} "
                  f"| EV {avg_explained_var:+.3f} | lossMSE {avg_loss_mse:.4f} | "
                  f"lr {lr_now:.2e} | {rate:.1f} steps/s | {acc_str}")

            with open(log_dir / f"{log_name}.json", "w") as f:
                json.dump(log, f, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "probe_state": probe.state_dict(),
        "config": asdict(cfg),
        "probe_config": asdict(pcfg),
        "final_log": log,
    }, log_dir / "final.pt")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Final per-task accuracy:")
    for name in TASK_NAMES:
        print(f"  {name:10s}: {log['per_task_acc'][name][-1]:.4f}")

    return log


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Residual-stream RNN probe (Dreamer v3 style).")
    # Model
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    # Training
    p.add_argument("--total_steps", type=int, default=Config.total_steps)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--eval_every", type=int, default=Config.eval_every)
    p.add_argument("--device", type=str, default=Config.device, choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_dir", type=str, default="runs/rnn_probe")
    p.add_argument("--seed", type=int, default=Config.seed)
    # Probe
    p.add_argument("--n_categoricals", type=int, default=ProbeConfig.n_categoricals)
    p.add_argument("--n_classes", type=int, default=ProbeConfig.n_classes)
    p.add_argument("--d_hidden", type=int, default=ProbeConfig.d_hidden)
    p.add_argument("--probe_lr", type=float, default=ProbeConfig.probe_lr)
    p.add_argument("--w_recon", type=float, default=ProbeConfig.w_recon)
    p.add_argument("--w_loss", type=float, default=ProbeConfig.w_loss)
    p.add_argument("--source_layer", type=int, default=ProbeConfig.source_layer)
    p.add_argument("--target_layer", type=int, default=ProbeConfig.target_layer)
    p.add_argument("--attach", action="store_true",
                   help="Let probe gradients flow into the transformer, turning "
                        "the probe's objective into an auxiliary regulariser.")
    p.add_argument("--w_probe_to_xform", type=float, default=ProbeConfig.w_probe_to_xform,
                   help="Scale of the probe loss when fed back into the transformer "
                        "(only used when --attach is set).")
    # Sweep
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds, e.g. '0,1,2'. Overrides --seed and "
                        "runs once per seed.")
    p.add_argument("--conditions", type=str, default=None,
                   help="Comma-separated conditions to sweep. Supported: "
                        "'detached', 'weak_attached' (w=0.1), 'attached' (w=1.0). "
                        "Each condition runs once per seed in --seeds.")
    p.add_argument("--log_root", type=str, default="runs",
                   help="Parent directory for sweep runs. Each run writes to "
                        "{log_root}/rnn_probe_{condition}_s{seed}/.")
    return p.parse_args()


# Condition registry for sweep mode. Each entry returns a ProbeConfig override
# dict. Keeps the condition definitions in one place so naming and settings
# can't drift apart.
CONDITIONS = {
    "detached":      {"attach": False, "w_probe_to_xform": 0.0},
    "weak_attached": {"attach": True,  "w_probe_to_xform": 0.1},
    "attached":      {"attach": True,  "w_probe_to_xform": 1.0},
}


def build_pcfg(args, overrides=None):
    """Construct a ProbeConfig from CLI args, applying optional overrides."""
    overrides = overrides or {}
    return ProbeConfig(
        n_categoricals=args.n_categoricals,
        n_classes=args.n_classes,
        d_hidden=args.d_hidden,
        probe_lr=args.probe_lr,
        w_recon=args.w_recon,
        w_loss=args.w_loss,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        attach=overrides.get("attach", args.attach),
        w_probe_to_xform=overrides.get("w_probe_to_xform", args.w_probe_to_xform),
    )


def build_cfg(args, seed, log_dir):
    """Construct a baseline Config with a specific seed and log_dir."""
    cfg = Config(
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        device=args.device,
        seed=seed,
        log_dir=log_dir,
    )
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"n_heads={cfg.n_heads} must divide d_model={cfg.d_model}")
    return cfg


def main():
    args = parse_args()

    # Sweep mode: --seeds and --conditions both provided.
    if args.seeds and args.conditions:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(
                f"Unknown condition(s): {unknown}. "
                f"Supported: {list(CONDITIONS.keys())}"
            )

        log_root = Path(args.log_root)
        log_root.mkdir(parents=True, exist_ok=True)

        runs = [(c, s) for c in conditions for s in seeds]
        print(f"Sweep: {len(runs)} runs under {log_root}")
        for c, s in runs:
            print(f"  rnn_probe_{c}_s{s}")
        print()

        t_sweep_start = time.time()
        for i, (cond, seed) in enumerate(runs):
            run_name = f"rnn_probe_{cond}_s{seed}"
            log_dir = str(log_root / run_name)
            print("=" * 80)
            print(f"RUN {i+1}/{len(runs)}: {run_name}")
            print("=" * 80)
            cfg = build_cfg(args, seed=seed, log_dir=log_dir)
            pcfg = build_pcfg(args, overrides=CONDITIONS[cond])
            # Log file basename matches the condition so the resulting files
            # follow the convention established by the pilot runs:
            # rnn_probe_detached_s0/detached_log.json, etc.
            train(cfg, pcfg, log_name=f"{cond}_log")

        elapsed = time.time() - t_sweep_start
        print(f"\nSweep complete: {len(runs)} runs in {elapsed/60:.1f} min "
              f"(avg {elapsed/len(runs)/60:.1f} min/run)")
        print(f"Logs under: {log_root}/rnn_probe_<condition>_s<seed>/<condition>_log.json")
        return

    # Single-run mode (unchanged behaviour).
    cfg = build_cfg(args, seed=args.seed, log_dir=args.log_dir)
    pcfg = build_pcfg(args)
    train(cfg, pcfg)


if __name__ == "__main__":
    main()
