"""
RNN steering: prefrontal-style control of transformer computation.

Single forward pass per training step. The RNN watches the transformer's
source-layer residual stream, produces a steering vector s_t per position,
and the steering vector is added to the residual at the configured site.
The transformer trains on the steered CE loss; the RNN trains on steered
CE plus optional auxiliary objectives (residual reconstruction and per-token
CE prediction).

ARCHITECTURE, ONE FORWARD PASS:

    ids -> [block 0] -> r_src ----+--> [RNN] -> s, r2_hat, loss_hat
                                  |              |
                                  +------> (r_src + s) --> [block 1] --> logits -> CE

GRADIENT FLOW:

    CE backpropagates through block 1 into (r_src + s), splitting into two
    paths:
        direct path: CE -> r_src -> block 0. Always flows; trains the
                     transformer on the steered loss.
        RNN-routed:  CE -> s -> RNN -> (r_src if attached). Trains the
                     steering head (and RNN) on CE.

    The attach flag controls whether `r_src` is detached where the RNN reads
    it. If detached (default), all RNN-originating gradient flow stops at
    the RNN's input boundary, meaning neither steering-CE nor aux losses
    reach the transformer's earlier layers. If attached, all three (CE
    through s, recon, loss-pred) push on earlier layers, turning the RNN's
    objectives into auxiliary regularisers on the transformer.

    Aux targets (r_tgt for recon, per-token CE for loss-pred) are always
    detached, so the auxiliary objective shapes the predictor, not the
    predicted.

AUX HEADS (optional):

    r2_hat   = decoder(z_t, h_t)          trained to predict r_tgt (MSE)
    loss_hat = loss_head(z_t, h_t)        trained to predict per-token CE (MSE)

    Controlled by --w_recon and --w_loss_pred. Both default to 0.0, which
    means the heads exist but contribute no gradient. Set positive weights
    to turn them on.

EVALUATION:

    Every eval_every steps, paired forward passes (with vs without steering,
    no grad) measure delta = loss_steered - loss_unsteered, overall and
    per-task. Also a shuffled-context version that permutes the source
    residuals across the batch before the RNN scan, to test whether the RNN
    is using per-sequence context or just learning a batch-independent bias.

CONDITION PRESETS (for --conditions sweep mode):

    steer_only        : w_recon=0, w_loss_pred=0, attach=False
                        Pure steering, no aux, RNN can't shape transformer.
    steer_aux         : w_recon=1, w_loss_pred=1, attach=False
                        Aux heads on, still detached. Tests whether the
                        probe objective gives the RNN a better representation.
    steer_aux_attach  : w_recon=1, w_loss_pred=1, attach=True
                        Full integration. RNN's objectives regularise the
                        transformer.

Run:
    python rnn_steer.py
    python rnn_steer.py --w_recon 1.0 --w_loss_pred 1.0
    python rnn_steer.py --w_recon 1.0 --w_loss_pred 1.0 --attach
    python rnn_steer.py --conditions steer_only,steer_aux,steer_aux_attach --seeds 0,1,2
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

import baseline
from baseline import (
    Config,
    VOCAB_SIZE,
    PAD_ID,
    TASK_NAMES,
    make_batch,
    evaluate,
    get_lr,
    count_params,
)
from rnn_probe import TinyGPTWithResiduals, SparseLatent


# =============================================================================
# STEERING CONFIG
# =============================================================================

@dataclass
class SteerConfig:
    # RNN architecture (mirrors ProbeConfig so results are comparable).
    n_categoricals: int = 8
    n_classes: int = 8
    d_hidden: int = 64
    d_mlp: int = 128

    # Where steering is injected. steer_layer is the residual index *after*
    # which s_t is added. steer_layer=0 means: add s_t to block[0]'s output,
    # so block[1] sees the steered residual. source_layer is what the RNN
    # reads. Default: RNN reads layer-0 output and steers the same residual.
    source_layer: int = 0
    steer_layer: int = 0
    # Layer whose residual the aux recon head targets. Set to -1 to default
    # to n_layers-1 at build time (i.e. predict the final-block residual).
    target_layer: int = -1

    # Optimiser for RNN + steering head + aux heads.
    steer_lr: float = 3e-4
    steer_weight_decay: float = 0.0

    # Steering magnitude. The head output is multiplied by this before being
    # added. 1.0 is the natural scale; zero-init of the final linear means
    # initial magnitude is 0 regardless.
    steer_scale: float = 1.0

    # Auxiliary objective weights. 0.0 disables (the heads still exist but
    # contribute no gradient).
    w_recon: float = 0.0
    w_loss_pred: float = 0.0

    # Attach flag. False (default): source residual is detached where the
    # RNN reads it, so no RNN-originating gradient reaches the transformer's
    # earlier layers. True: source residual is not detached, and the RNN's
    # CE / recon / loss_pred gradients all flow into the transformer.
    attach: bool = False

    # Evaluation knobs.
    delta_eval_batches: int = 4
    shuffle_h_eval: bool = True


# =============================================================================
# STEERING RNN
# =============================================================================

class SteerRNN(nn.Module):
    """RNN with three heads: steering, residual reconstruction, loss prediction.

    All three share the same (z_t, h_t) feature. The steering head is the
    only one active at forward time (its output mutates the transformer's
    residual); recon and loss_pred heads are pure predictors and their
    outputs are only used to compute auxiliary losses.

    The steering head's final linear is zero-initialised so the steered
    forward equals the unsteered forward at step 0.
    """

    def __init__(self, d_model: int, scfg: SteerConfig):
        super().__init__()
        self.scfg = scfg
        self.d_model = d_model
        self.d_hidden = scfg.d_hidden

        self.latent = SparseLatent(
            d_in=d_model + scfg.d_hidden,
            n_categoricals=scfg.n_categoricals,
            n_classes=scfg.n_classes,
            d_hidden=scfg.d_mlp,
        )
        d_z = self.latent.out_dim

        self.cell = nn.GRUCell(input_size=d_z, hidden_size=scfg.d_hidden)

        feat_dim = d_z + scfg.d_hidden

        # Steering head. Zero-init the final linear so the steered forward
        # equals the unsteered forward at init.
        self.steer_head = nn.Sequential(
            nn.Linear(feat_dim, scfg.d_mlp),
            nn.GELU(),
            nn.Linear(scfg.d_mlp, d_model),
        )
        nn.init.zeros_(self.steer_head[-1].weight)
        nn.init.zeros_(self.steer_head[-1].bias)

        # Aux: residual reconstruction.
        self.recon_head = nn.Sequential(
            nn.Linear(feat_dim, scfg.d_mlp),
            nn.GELU(),
            nn.Linear(scfg.d_mlp, d_model),
        )

        # Aux: per-token CE prediction.
        self.loss_head = nn.Sequential(
            nn.Linear(feat_dim, scfg.d_mlp),
            nn.GELU(),
            nn.Linear(scfg.d_mlp, 1),
        )

    def init_hidden(self, batch_size: int, device):
        return torch.zeros(batch_size, self.d_hidden, device=device)

    def scan(self, r_src, want_aux: bool = True, h0=None):
        """Left-to-right scan over the sequence.

        Args:
            r_src:    (B, T, d_model) source-layer residuals. Detach before
                      calling if you want to block gradients to the transformer.
            want_aux: if False, recon and loss_pred heads are skipped.
                      Useful for the paired eval to save compute.
            h0:       optional initial hidden state. Default zeros.

        Returns:
            s:        (B, T, d_model) steering vectors.
            r2_hat:   (B, T, d_model) recon head outputs, or None.
            loss_hat: (B, T) loss-prediction outputs, or None.
        """
        B, T, _ = r_src.shape
        h = self.init_hidden(B, r_src.device) if h0 is None else h0
        ss, r2s, lhs = [], [], []
        for t in range(T):
            inp = torch.cat([r_src[:, t, :], h], dim=-1)
            z = self.latent(inp)
            h = self.cell(z, h)
            feat = torch.cat([z, h], dim=-1)
            ss.append(self.steer_head(feat))
            if want_aux:
                r2s.append(self.recon_head(feat))
                lhs.append(self.loss_head(feat).squeeze(-1))
        s = torch.stack(ss, dim=1)
        r2_hat = torch.stack(r2s, dim=1) if want_aux else None
        loss_hat = torch.stack(lhs, dim=1) if want_aux else None
        return s, r2_hat, loss_hat


# =============================================================================
# SINGLE-PASS FORWARD WITH STEERING
# =============================================================================

def forward_with_steering(model: TinyGPTWithResiduals,
                          ids: torch.Tensor,
                          rnn: SteerRNN,
                          scfg: SteerConfig,
                          target_layer: int,
                          want_aux: bool = True):
    """Single forward pass through the transformer with RNN steering applied.

    Produces:
        logits:   (B, T, vocab)
        s:        (B, T, d_model) steering vector
        r_src:    (B, T, d_model) residual where RNN reads (post-detach if
                  attach=False)
        r_tgt:    (B, T, d_model) residual at target_layer, for aux recon.
                  Always returned detached (used only as a target).
        r2_hat, loss_hat: aux head outputs, or None if want_aux=False.
    """
    B, T = ids.shape
    pos = model._pos_ids[:, :T]
    x = model.tok_emb(ids) + model.pos_emb(pos)

    r_src = None
    r_tgt = None
    s = None

    for i, blk in enumerate(model.blocks):
        x = blk(x)

        # Capture source residual and compute steering before the next block.
        if i == scfg.source_layer:
            r_src_in = x if scfg.attach else x.detach()
            r_src = r_src_in
            s, r2_hat, loss_hat = rnn.scan(r_src_in, want_aux=want_aux)

        # Inject steering at the steer_layer boundary.
        if i == scfg.steer_layer:
            # If steer_layer < source_layer, s is not yet defined. Guard it.
            if s is None:
                raise ValueError(
                    f"steer_layer ({scfg.steer_layer}) must be >= source_layer "
                    f"({scfg.source_layer}); steering vector is not available yet."
                )
            x = x + scfg.steer_scale * s

        # Capture target residual for aux recon. Always detached because it
        # is used only as a target.
        if i == target_layer:
            r_tgt = x.detach()

    x = model.ln_f(x)
    logits = x @ model.tok_emb.weight.T
    return logits, s, r_src, r_tgt, r2_hat, loss_hat


def forward_unsteered(model: TinyGPTWithResiduals, ids: torch.Tensor):
    """Plain forward without any steering. Used only during eval."""
    B, T = ids.shape
    pos = model._pos_ids[:, :T]
    x = model.tok_emb(ids) + model.pos_emb(pos)
    for blk in model.blocks:
        x = blk(x)
    x = model.ln_f(x)
    logits = x @ model.tok_emb.weight.T
    return logits


def forward_with_external_steering(model: TinyGPTWithResiduals,
                                   ids: torch.Tensor,
                                   steer_vec: torch.Tensor,
                                   steer_layer: int):
    """Forward pass with a pre-computed steering vector. Used during eval."""
    B, T = ids.shape
    pos = model._pos_ids[:, :T]
    x = model.tok_emb(ids) + model.pos_emb(pos)
    for i, blk in enumerate(model.blocks):
        x = blk(x)
        if i == steer_layer:
            x = x + steer_vec
    x = model.ln_f(x)
    logits = x @ model.tok_emb.weight.T
    return logits


# =============================================================================
# LOSS HELPERS
# =============================================================================

def ce_masked(logits, targets, mask):
    """Per-token masked CE averaged over non-pad answer tokens.

    Returns (mean_loss, per_token_loss_tensor).
    """
    per_tok = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    denom = mask.sum().clamp(min=1.0)
    return (per_tok * mask).sum() / denom, per_tok


# =============================================================================
# EVALUATION: STEERING DELTA
# =============================================================================

@torch.no_grad()
def eval_steering_delta(model, rnn, cfg, scfg, device, rng, n_batches):
    """Paired forward passes (with vs without steering) to measure delta.

    Also runs a shuffled-context variant where the RNN's input residuals are
    permuted across the batch. If shuffled delta is near zero or positive
    while true delta is negative, the RNN is using per-sequence context.
    """
    model.eval()
    rnn.eval()

    total = {"u": 0.0, "s": 0.0, "s_shuf": 0.0, "n": 0}
    per_task = {name: {"u": 0.0, "s": 0.0, "s_shuf": 0.0, "n": 0} for name in TASK_NAMES}

    for _ in range(n_batches):
        input_ids, loss_mask, task_ids, _ = make_batch(rng, cfg)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        ids_in = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        # Unsteered pass.
        logits_u = forward_unsteered(model, ids_in)
        _, per_tok_u = ce_masked(logits_u, targets, mask)

        # Build r_src for RNN. We always detach at eval; attach mode is a
        # training-time concept.
        B, T = ids_in.shape
        pos = model._pos_ids[:, :T]
        x = model.tok_emb(ids_in) + model.pos_emb(pos)
        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i == scfg.source_layer:
                r_src = x.detach()
                break

        s, _, _ = rnn.scan(r_src, want_aux=False)
        s = scfg.steer_scale * s
        logits_s = forward_with_external_steering(model, ids_in, s, scfg.steer_layer)
        _, per_tok_s = ce_masked(logits_s, targets, mask)

        # Shuffled-context variant.
        if scfg.shuffle_h_eval:
            perm = torch.randperm(B, device=device)
            s_shuf, _, _ = rnn.scan(r_src[perm], want_aux=False)
            s_shuf = scfg.steer_scale * s_shuf
            logits_shuf = forward_with_external_steering(model, ids_in, s_shuf, scfg.steer_layer)
            _, per_tok_shuf = ce_masked(logits_shuf, targets, mask)
        else:
            per_tok_shuf = per_tok_u

        m_sum = mask.sum().item()
        total["u"] += (per_tok_u * mask).sum().item()
        total["s"] += (per_tok_s * mask).sum().item()
        total["s_shuf"] += (per_tok_shuf * mask).sum().item()
        total["n"] += m_sum

        for t_idx, name in enumerate(TASK_NAMES):
            tmask = (task_ids == t_idx).float().unsqueeze(-1) * mask
            tm_sum = tmask.sum().item()
            if tm_sum > 0:
                per_task[name]["u"] += (per_tok_u * tmask).sum().item()
                per_task[name]["s"] += (per_tok_s * tmask).sum().item()
                per_task[name]["s_shuf"] += (per_tok_shuf * tmask).sum().item()
                per_task[name]["n"] += tm_sum

    model.train()
    rnn.train()

    def safediv(a, b):
        return a / b if b > 0 else float("nan")

    out = {
        "loss_unsteered": safediv(total["u"], total["n"]),
        "loss_steered": safediv(total["s"], total["n"]),
        "loss_steered_shuf": safediv(total["s_shuf"], total["n"]),
    }
    out["delta"] = out["loss_steered"] - out["loss_unsteered"]
    out["delta_shuf"] = out["loss_steered_shuf"] - out["loss_unsteered"]
    out["per_task"] = {}
    for name in TASK_NAMES:
        d = per_task[name]
        u = safediv(d["u"], d["n"])
        s = safediv(d["s"], d["n"])
        sh = safediv(d["s_shuf"], d["n"])
        out["per_task"][name] = {
            "loss_unsteered": u,
            "loss_steered": s,
            "delta": s - u,
            "delta_shuf": sh - u,
        }
    return out


# =============================================================================
# TRAINING
# =============================================================================

def train(cfg: Config, scfg: SteerConfig, log_name: str = "log"):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # max_len setup same as baseline / rnn_probe.
    probe_rng = random.Random(0)
    max_len = 0
    for _ in range(200):
        for name in TASK_NAMES:
            ids, _ = baseline.sample_task(probe_rng, name, cfg)
            max_len = max(max_len, len(ids))
    max_len = max_len + 4
    print(f"max_len: {max_len}")

    assert 0 <= scfg.source_layer < cfg.n_layers, "source_layer out of range"
    assert 0 <= scfg.steer_layer < cfg.n_layers, "steer_layer out of range"
    # Resolve target_layer default.
    target_layer = scfg.target_layer if scfg.target_layer >= 0 else cfg.n_layers - 1
    assert 0 <= target_layer < cfg.n_layers, "target_layer out of range"

    model = TinyGPTWithResiduals(cfg, max_len=max_len).to(device)
    rnn = SteerRNN(d_model=cfg.d_model, scfg=scfg).to(device)

    n_params = count_params(model)
    n_rnn_params = count_params(rnn)
    print(f"Transformer params: {n_params:,}")
    print(f"RNN/steer params:   {n_rnn_params:,}")
    print(f"attach={scfg.attach}  w_recon={scfg.w_recon}  w_loss_pred={scfg.w_loss_pred}")

    xform_opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    steer_opt = torch.optim.AdamW(
        rnn.parameters(), lr=scfg.steer_lr, weight_decay=scfg.steer_weight_decay,
        betas=(0.9, 0.95),
    )

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.json", "w") as f:
        json.dump({"baseline": asdict(cfg), "steer": asdict(scfg),
                   "target_layer_resolved": target_layer}, f, indent=2)

    aux_on = (scfg.w_recon > 0.0) or (scfg.w_loss_pred > 0.0)

    log = {
        "step": [],
        "loss": [],                   # training CE (steered)
        "recon_loss": [],             # aux recon MSE (if on, else NaN)
        "loss_pred_mse": [],          # aux loss-prediction MSE (if on, else NaN)
        "delta": [],
        "delta_shuf": [],
        "loss_unsteered_eval": [],
        "loss_steered_eval": [],
        "per_task_acc": {name: [] for name in TASK_NAMES},
        "per_task_loss": {name: [] for name in TASK_NAMES},
        "per_task_delta": {name: [] for name in TASK_NAMES},
        "n_params": n_params,
        "n_rnn_params": n_rnn_params,
        "attach": scfg.attach,
        "w_recon": scfg.w_recon,
        "w_loss_pred": scfg.w_loss_pred,
    }

    rng = random.Random(cfg.seed + 1)
    eval_rng = random.Random(cfg.seed + 99991)
    model.train()
    rnn.train()
    t_start = time.time()
    running = {"loss": 0.0, "recon": 0.0, "loss_pred": 0.0}
    running_per_task_loss = {name: 0.0 for name in TASK_NAMES}
    running_per_task_count = {name: 0 for name in TASK_NAMES}

    for step in range(cfg.total_steps):
        input_ids, loss_mask, task_ids, _ = make_batch(rng, cfg)
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        task_ids = task_ids.to(device)

        ids_in = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:]

        # -----------------------------------------------------------------
        # SINGLE FORWARD PASS
        # -----------------------------------------------------------------
        logits, s, r_src, r_tgt, r2_hat, loss_hat = forward_with_steering(
            model, ids_in, rnn, scfg, target_layer=target_layer, want_aux=aux_on,
        )

        ce_loss, per_tok_ce = ce_masked(logits, targets, mask)

        # Aux losses. Targets detached so the aux objective shapes the
        # predictor, not the predicted. Averaged over answer tokens.
        denom = mask.sum().clamp(min=1.0)
        if aux_on:
            recon_per_tok = ((r2_hat - r_tgt.detach()) ** 2).mean(dim=-1)
            recon_loss = (recon_per_tok * mask).sum() / denom

            loss_pred_per_tok = (loss_hat - per_tok_ce.detach()) ** 2
            loss_pred_mse = (loss_pred_per_tok * mask).sum() / denom
        else:
            recon_loss = torch.zeros((), device=device)
            loss_pred_mse = torch.zeros((), device=device)

        total_loss = (
            ce_loss
            + scfg.w_recon * recon_loss
            + scfg.w_loss_pred * loss_pred_mse
        )

        # Per-task loss for logging (from the steered forward; this is
        # what the transformer is actually being trained on).
        with torch.no_grad():
            for t_idx, name in enumerate(TASK_NAMES):
                task_mask = (task_ids == t_idx).float().unsqueeze(-1) * mask
                tm_sum = task_mask.sum()
                if tm_sum > 0:
                    tl = (per_tok_ce * task_mask).sum() / tm_sum
                    running_per_task_loss[name] += tl.item()
                    running_per_task_count[name] += 1

        lr_now = get_lr(step, cfg)
        for g in xform_opt.param_groups:
            g["lr"] = lr_now

        # Single backward. Both optimisers step on their own parameter
        # groups. If attach=False, r_src was detached in the forward, so
        # RNN-routed gradients cannot reach the transformer; only the direct
        # CE path (block 1 -> (r_src + s) -> block 0) trains the transformer.
        xform_opt.zero_grad(set_to_none=True)
        steer_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg.grad_clip)
        xform_opt.step()
        steer_opt.step()

        running["loss"] += ce_loss.item()
        running["recon"] += recon_loss.item() if aux_on else float("nan")
        running["loss_pred"] += loss_pred_mse.item() if aux_on else float("nan")

        if (step + 1) % cfg.eval_every == 0 or step == 0:
            denom_log = cfg.eval_every if step > 0 else 1
            avg_loss = running["loss"] / denom_log
            avg_recon = (running["recon"] / denom_log) if aux_on else float("nan")
            avg_loss_pred = (running["loss_pred"] / denom_log) if aux_on else float("nan")
            running = {"loss": 0.0, "recon": 0.0, "loss_pred": 0.0}

            delta_info = eval_steering_delta(
                model, rnn, cfg, scfg, device, eval_rng,
                n_batches=scfg.delta_eval_batches,
            )

            # Downstream accuracy uses the plain (unsteered) transformer.
            # This is a probe of how the transformer does on its own, which
            # is informative even though at training time it's always seen
            # the steered residual.
            accs = evaluate(model, cfg, device)
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed

            log["step"].append(step + 1)
            log["loss"].append(avg_loss)
            log["recon_loss"].append(avg_recon)
            log["loss_pred_mse"].append(avg_loss_pred)
            log["delta"].append(delta_info["delta"])
            log["delta_shuf"].append(delta_info["delta_shuf"])
            log["loss_unsteered_eval"].append(delta_info["loss_unsteered"])
            log["loss_steered_eval"].append(delta_info["loss_steered"])
            for name in TASK_NAMES:
                log["per_task_acc"][name].append(accs[name])
                if running_per_task_count[name] > 0:
                    ptl = running_per_task_loss[name] / running_per_task_count[name]
                else:
                    ptl = float("nan")
                log["per_task_loss"][name].append(ptl)
                log["per_task_delta"][name].append(delta_info["per_task"][name]["delta"])
                running_per_task_loss[name] = 0.0
                running_per_task_count[name] = 0

            acc_str = " ".join(f"{name[:4]}:{accs[name]:.2f}" for name in TASK_NAMES)
            aux_str = ""
            if aux_on:
                aux_str = f" | recon {avg_recon:.4f} lossMSE {avg_loss_pred:.4f}"
            print(
                f"step {step+1:6d} | loss {avg_loss:.4f}{aux_str} "
                f"| D {delta_info['delta']:+.4f} (shuf {delta_info['delta_shuf']:+.4f}) "
                f"| lr {lr_now:.2e} | {rate:.1f}/s"
            )
            print(f"             {acc_str}")

            with open(log_dir / f"{log_name}.json", "w") as f:
                json.dump(log, f, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "rnn_state": rnn.state_dict(),
        "config": asdict(cfg),
        "steer_config": asdict(scfg),
        "final_log": log,
    }, log_dir / "final.pt")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("Final per-task accuracy and steering delta:")
    for name in TASK_NAMES:
        acc = log["per_task_acc"][name][-1]
        d = log["per_task_delta"][name][-1]
        print(f"  {name:10s}: acc={acc:.4f}  delta={d:+.4f}")

    return log


# =============================================================================
# CLI
# =============================================================================

# Condition presets for sweep mode. Each maps to SteerConfig overrides.
CONDITIONS = {
    "steer_only":        {"w_recon": 0.0, "w_loss_pred": 0.0, "attach": False},
    "steer_aux":         {"w_recon": 1.0, "w_loss_pred": 1.0, "attach": False},
    "steer_aux_attach":  {"w_recon": 1.0, "w_loss_pred": 1.0, "attach": True},
}


def parse_args():
    p = argparse.ArgumentParser(description="RNN steering of transformer computation.")
    # Model.
    p.add_argument("--n_layers", type=int, default=Config.n_layers)
    p.add_argument("--d_model", type=int, default=Config.d_model)
    p.add_argument("--n_heads", type=int, default=Config.n_heads)
    # Training.
    p.add_argument("--total_steps", type=int, default=Config.total_steps)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--eval_every", type=int, default=Config.eval_every)
    p.add_argument("--device", type=str, default=Config.device, choices=["cpu", "mps", "cuda"])
    p.add_argument("--log_dir", type=str, default="runs/rnn_steer")
    p.add_argument("--seed", type=int, default=Config.seed)
    # Steer.
    p.add_argument("--n_categoricals", type=int, default=SteerConfig.n_categoricals)
    p.add_argument("--n_classes", type=int, default=SteerConfig.n_classes)
    p.add_argument("--d_hidden", type=int, default=SteerConfig.d_hidden)
    p.add_argument("--steer_lr", type=float, default=SteerConfig.steer_lr)
    p.add_argument("--steer_scale", type=float, default=SteerConfig.steer_scale)
    p.add_argument("--source_layer", type=int, default=SteerConfig.source_layer)
    p.add_argument("--steer_layer", type=int, default=SteerConfig.steer_layer)
    p.add_argument("--target_layer", type=int, default=SteerConfig.target_layer,
                   help="Layer whose residual the recon head targets. -1 = last.")
    p.add_argument("--w_recon", type=float, default=SteerConfig.w_recon,
                   help="Weight for residual reconstruction aux loss. 0 disables.")
    p.add_argument("--w_loss_pred", type=float, default=SteerConfig.w_loss_pred,
                   help="Weight for per-token CE prediction aux loss. 0 disables.")
    p.add_argument("--attach", action="store_true",
                   help="Let RNN gradients flow into the transformer. "
                        "Default: RNN's input residual is detached.")
    p.add_argument("--delta_eval_batches", type=int, default=SteerConfig.delta_eval_batches)
    p.add_argument("--no_shuffle_eval", action="store_true",
                   help="Disable shuffled-context sanity check at eval.")
    # Sweep.
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds, e.g. '0,1,2'. Overrides --seed.")
    p.add_argument("--conditions", type=str, default=None,
                   help="Comma-separated conditions. Supported: "
                        f"{list(CONDITIONS.keys())}")
    p.add_argument("--log_root", type=str, default="runs",
                   help="Parent dir for multi-seed / multi-condition runs.")
    return p.parse_args()


def build_scfg(args, overrides=None):
    overrides = overrides or {}
    return SteerConfig(
        n_categoricals=args.n_categoricals,
        n_classes=args.n_classes,
        d_hidden=args.d_hidden,
        steer_lr=args.steer_lr,
        steer_scale=args.steer_scale,
        source_layer=args.source_layer,
        steer_layer=args.steer_layer,
        target_layer=args.target_layer,
        w_recon=overrides.get("w_recon", args.w_recon),
        w_loss_pred=overrides.get("w_loss_pred", args.w_loss_pred),
        attach=overrides.get("attach", args.attach),
        delta_eval_batches=args.delta_eval_batches,
        shuffle_h_eval=not args.no_shuffle_eval,
    )


def build_cfg(args, seed, log_dir):
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
            print(f"  rnn_steer_{c}_s{s}")
        print()

        t_sweep_start = time.time()
        for i, (cond, seed) in enumerate(runs):
            run_name = f"rnn_steer_{cond}_s{seed}"
            log_dir = str(log_root / run_name)
            print("=" * 80)
            print(f"RUN {i+1}/{len(runs)}: {run_name}")
            print("=" * 80)
            cfg = build_cfg(args, seed=seed, log_dir=log_dir)
            scfg = build_scfg(args, overrides=CONDITIONS[cond])
            train(cfg, scfg, log_name=f"{cond}_log")

        elapsed = time.time() - t_sweep_start
        print(f"\nSweep complete: {len(runs)} runs in {elapsed/60:.1f} min "
              f"(avg {elapsed/len(runs)/60:.1f} min/run)")
        return

    # Seed-only sweep (no conditions).
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        log_root = Path(args.log_root)
        log_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep: {len(seeds)} seeds under {log_root}")
        for seed in seeds:
            run_name = f"rnn_steer_s{seed}"
            log_dir = str(log_root / run_name)
            print("=" * 80)
            print(f"RUN seed={seed}: {run_name}")
            print("=" * 80)
            cfg = build_cfg(args, seed=seed, log_dir=log_dir)
            scfg = build_scfg(args)
            train(cfg, scfg, log_name="log")
        return

    cfg = build_cfg(args, seed=args.seed, log_dir=args.log_dir)
    scfg = build_scfg(args)
    train(cfg, scfg)


if __name__ == "__main__":
    main()
