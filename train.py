"""
Training loop for HybridTransformerLM on TinyStories with online phrase building.

Designed for a single T4 (16 GB). Uses fp16 AMP (T4 has no bf16).

Phrase boundaries are computed inside the model from input_ids. The dataset
just stores tokenized text. The --chunker flag selects between sentence-aware
(OnlinePhraseBuilder) and fixed-stride (FixedStridePhraseBuilder) ablation
baselines.

Usage:
    python train.py --steps 5000 --batch-size 8 --max-len 384 --lr 3e-4 \
        --chunker online --log-file run_online.jsonl

    python train.py --steps 5000 --batch-size 8 --max-len 384 --lr 3e-4 \
        --chunker fixed --fixed-stride 4 --log-file run_fixed.jsonl

The training script ignores the `phrase_spans` field in cached data files,
so existing prepare_data.py outputs work without modification.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from model import ModelConfig, HybridTransformerLM
from online_phrase_builder import OnlinePhraseBuilder, FixedStridePhraseBuilder


# -------------------------------------------------------------------------
# Dataset and collator
# -------------------------------------------------------------------------

class TokenIdsDataset(Dataset):
    """
    Loads cached token ids from prepare_data.py output. The phrase_spans
    field is ignored.
    """
    def __init__(self, path: str):
        blob = torch.load(path, weights_only=False)
        self.input_ids = blob["input_ids"]
        self.tokenizer_name = blob["tokenizer_name"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx]}


class TokenIdsCollator:
    """Pads input_ids to the batch max (clipped at max_len)."""
    def __init__(self, pad_token_id: int, max_len: int):
        self.pad_token_id = pad_token_id
        self.max_len = max_len

    def __call__(self, batch):
        T = min(self.max_len, max(len(b["input_ids"]) for b in batch))
        B = len(batch)
        input_ids = torch.full((B, T), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"][:T]
            L = ids.numel()
            input_ids[i, :L] = ids
            labels[i, :L] = ids
        return {"input_ids": input_ids, "labels": labels}


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
# Eval loop
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _, loss = model(batch["input_ids"], labels=batch["labels"])
        n = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--config-overrides", type=str, default=None,
                    help="JSON dict of ModelConfig fields to override.")
    ap.add_argument("--chunker", choices=["online", "fixed"], default="online",
                    help="Phrase builder type.")
    ap.add_argument("--fixed-stride", type=int, default=4,
                    help="Stride for the fixed builder (used when --chunker=fixed).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    train_path = Path(args.data_dir) / "train.pt"
    val_path = Path(args.data_dir) / "validation.pt"
    print(f"Loading {train_path}")
    train_ds = TokenIdsDataset(str(train_path))
    print(f"Loading {val_path}")
    val_ds = TokenIdsDataset(str(val_path))
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(train_ds.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    if args.config_overrides:
        overrides = json.loads(args.config_overrides)
        for k, v in overrides.items():
            setattr(model_cfg, k, v)
    print(f"\nModel config: {model_cfg}")

    # Build the phrase builder.
    if args.chunker == "online":
        phrase_builder = OnlinePhraseBuilder(
            tokenizer=tokenizer,
            max_phrase_len=model_cfg.max_phrase_len,
            pad_token_id=pad_id,
        )
        n_punct = int(phrase_builder.is_punct.sum().item())
        n_abbr = int(phrase_builder.is_abbreviation.sum().item())
        print(f"OnlinePhraseBuilder: {n_punct} punctuation tokens, "
              f"{n_abbr} abbreviation tokens")
    else:
        phrase_builder = FixedStridePhraseBuilder(
            max_phrase_len=model_cfg.max_phrase_len,
            pad_token_id=pad_id,
            stride=args.fixed_stride,
        )
        print(f"FixedStridePhraseBuilder: stride={phrase_builder.stride}")

    model = HybridTransformerLM(model_cfg, phrase_builder).to(device)
    n_params = model.num_parameters(exclude_embeddings=True)
    n_total = model.num_parameters(exclude_embeddings=False)
    print(f"Non-embedding params: {n_params:,}")
    print(f"Total params:         {n_total:,}")

    collator = TokenIdsCollator(pad_id, args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    no_decay_substrings = {"bias", "B_pos", "sink_logits"}
    decay_params = []
    nodecay_params = []
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
    train_iter = iter(train_loader)
    t_last = time.time()
    loss_accum = 0.0

    print(f"\nStarting training for {args.steps} steps "
          f"(effective batch = {args.batch_size * args.accum_steps})")

    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        skip_step = False

        for _ in range(args.accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.float16):
                _, loss = model(batch["input_ids"], labels=batch["labels"])
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
            tps = (args.batch_size * args.accum_steps * args.max_len * 20) / (now - t_last)
            log({
                "step": step, "loss": round(avg_loss, 4),
                "lr": round(lr, 6), "grad_norm": round(float(gnorm), 3),
                "tokens_per_sec": int(tps),
            })
            loss_accum = 0.0
            t_last = now

        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
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

    val_loss = evaluate(model, val_loader, device, max_batches=200)
    log({"step": step, "event": "final_eval",
         "val_loss": round(val_loss, 4),
         "val_ppl": round(math.exp(val_loss), 2)})

    if log_fp:
        log_fp.close()


if __name__ == "__main__":
    main()
