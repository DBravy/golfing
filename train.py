"""
Training loop for HybridTransformerLM on shard-based token data.

Designed for a single T4 (16 GB). Uses fp16 AMP (T4 has no bf16).

Data is consumed from binary token shards produced by prepare_shards.py.
Sequences are fixed-length (--seq-len), every batch is exactly (B, seq_len),
no padding. Documents are concatenated in the stream with EOS between them;
EOS is registered as a phrase boundary so phrases don't span documents.

Usage:
    # First, write shards (one-time):
    python prepare_shards.py --dataset roneneldan/TinyStories --split train \\
        --tokenizer gpt2 --out-dir ./data/datasets/tinystories_gpt2
    python prepare_shards.py --dataset roneneldan/TinyStories --split validation \\
        --tokenizer gpt2 --out-dir ./data/datasets/tinystories_gpt2

    # Then train:
    python train.py --steps 5000 --batch-size 8 --seq-len 384 --lr 3e-4 \\
        --chunker online --log-file run_online.jsonl

    # Fixed-stride ablation:
    python train.py --steps 5000 --batch-size 8 --seq-len 384 --lr 3e-4 \\
        --chunker fixed --fixed-stride 4 --log-file run_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from model import ModelConfig, HybridTransformerLM
from online_phrase_builder import OnlinePhraseBuilder, FixedStridePhraseBuilder
from dataset import TokenLoader, load_validation_tokens


# Sentinel pad value: -1 cannot appear in real token streams (tokens are uint16
# in [0, vocab_size)), so the phrase builder treats every position as real.
PAD_SENTINEL = -1


# -------------------------------------------------------------------------
# LR schedule
# -------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, base_lr: float, warmup: int,
           min_lr: float = 1e-5) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# -------------------------------------------------------------------------
# Eval over the full validation token stream
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_tokens: torch.Tensor, seq_len: int, batch_size: int,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_seqs = (val_tokens.numel() - 1) // seq_len
    for batch_start in range(0, n_seqs, batch_size):
        batch_end = min(batch_start + batch_size, n_seqs)
        actual_b = batch_end - batch_start
        raw_start = batch_start * seq_len
        raw_end = batch_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end].to(device, dtype=torch.int64,
                                                 non_blocking=True)
        x = chunk[:-1].reshape(actual_b, seq_len)
        y = chunk[1:].reshape(actual_b, seq_len)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss = model(x, labels=y)
        n = actual_b * seq_len
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data paths.
    ap.add_argument("--data-dir", default="./data/datasets/tinystories_gpt2",
                    help="Directory containing the shard files.")
    ap.add_argument("--train-glob", default="*_train_*.bin",
                    help="Glob pattern (relative to --data-dir) for train shards.")
    ap.add_argument("--val-glob", default="*_validation_*.bin",
                    help="Glob pattern for validation shards.")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer name (used by OnlinePhraseBuilder).")

    # Sequence shape.
    ap.add_argument("--seq-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=1)

    # Training schedule.
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-overrides", type=str, default=None,
                    help="JSON dict of ModelConfig fields to override.")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--log-file", default=None)

    # Chunker.
    ap.add_argument("--chunker", choices=["online", "fixed"], default="online")
    ap.add_argument("--fixed-stride", type=int, default=4)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Tokenizer (used only by the phrase builder).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Loaders.
    data_dir = Path(args.data_dir)
    train_pattern = str(data_dir / args.train_glob)
    val_pattern = str(data_dir / args.val_glob)
    print(f"Train shards: {train_pattern}")
    print(f"Val shards:   {val_pattern}")

    train_loader = TokenLoader(train_pattern, device)
    val_tokens = load_validation_tokens(val_pattern, args.seq_len)
    print(f"Train shard files: {len(train_loader.stream.files)}")
    print(f"Validation tokens: {val_tokens.numel():,}")

    # Model config.
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    if args.config_overrides:
        overrides = json.loads(args.config_overrides)
        for k, v in overrides.items():
            setattr(model_cfg, k, v)
    print(f"\nModel config: {model_cfg}")

    # Phrase builder.
    if args.chunker == "online":
        extras = []
        if tokenizer.eos_token_id is not None:
            extras.append(tokenizer.eos_token_id)
        phrase_builder = OnlinePhraseBuilder(
            tokenizer=tokenizer,
            max_phrase_len=model_cfg.max_phrase_len,
            pad_token_id=PAD_SENTINEL,
            extra_boundary_token_ids=extras,
        )
        n_punct = int(phrase_builder.is_punct.sum().item())
        n_abbr = int(phrase_builder.is_abbreviation.sum().item())
        print(f"OnlinePhraseBuilder: {n_punct} boundary tokens "
              f"(incl. EOS={tokenizer.eos_token_id}), {n_abbr} abbreviations")
    else:
        phrase_builder = FixedStridePhraseBuilder(
            max_phrase_len=model_cfg.max_phrase_len,
            pad_token_id=PAD_SENTINEL,
            stride=args.fixed_stride,
        )
        print(f"FixedStridePhraseBuilder: stride={phrase_builder.stride}")

    model = HybridTransformerLM(model_cfg, phrase_builder).to(device)
    n_params = model.num_parameters(exclude_embeddings=True)
    n_total = model.num_parameters(exclude_embeddings=False)
    print(f"Non-embedding params: {n_params:,}")
    print(f"Total params:         {n_total:,}")

    # Optimizer with split decay.
    no_decay_substrings = {"bias", "B_pos", "sink_logits"}
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_nodecay = (any(k in name for k in no_decay_substrings)
                      or "norm" in name.lower())
        (nodecay_params if is_nodecay else decay_params).append(p)

    optim = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    log_fp = open(args.log_file, "w") if args.log_file else None
    def log(record: dict):
        line = json.dumps(record)
        print(line)
        if log_fp:
            log_fp.write(line + "\n")
            log_fp.flush()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    step = 0
    t_last = time.time()
    loss_accum = 0.0

    print(f"\nStarting training for {args.steps} steps "
          f"(effective batch = {args.batch_size * args.accum_steps} sequences "
          f"of {args.seq_len} tokens)")

    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        skip_step = False

        for _ in range(args.accum_steps):
            x, y = train_loader.next_batch(args.batch_size, args.seq_len)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                _, loss = model(x, labels=y)
                loss = loss / args.accum_steps

            if not torch.isfinite(loss):
                log({"step": step, "event": "non_finite_loss",
                     "loss": float(loss.item())})
                optim.zero_grad(set_to_none=True)
                skip_step = True
                break

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if skip_step:
            step += 1
            continue

        lr = get_lr(step, args.steps, args.lr, args.warmup)
        for pg in optim.param_groups:
            pg["lr"] = lr

        scaler.unscale_(optim)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        loss_accum += accum_loss
        step += 1

        if step % 20 == 0:
            now = time.time()
            avg_loss = loss_accum / 20
            tps = (args.batch_size * args.accum_steps * args.seq_len * 20) / (now - t_last)
            log({
                "step": step, "loss": round(avg_loss, 4),
                "lr": round(lr, 6), "grad_norm": round(float(gnorm), 3),
                "tokens_per_sec": int(tps),
            })
            loss_accum = 0.0
            t_last = now

        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_tokens, args.seq_len,
                                args.batch_size, device)
            log({"step": step, "event": "eval",
                 "val_loss": round(val_loss, 4),
                 "val_ppl": round(math.exp(val_loss), 2)})

        if step % args.ckpt_every == 0:
            ckpt_path = Path(args.ckpt_dir) / f"step_{step}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "config": vars(model_cfg),
                "chunker": args.chunker,
                "fixed_stride": args.fixed_stride,
            }, ckpt_path)
            log({"step": step, "event": "checkpoint", "path": str(ckpt_path)})

    val_loss = evaluate(model, val_tokens, args.seq_len, args.batch_size, device)
    log({"step": step, "event": "final_eval",
         "val_loss": round(val_loss, 4),
         "val_ppl": round(math.exp(val_loss), 2)})

    if log_fp:
        log_fp.close()


if __name__ == "__main__":
    main()
