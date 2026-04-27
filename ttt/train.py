"""
Train and compare vanilla TTT-E2E against a standard transformer.

Both models share the same architecture (same n_layers, d_model, ff_mult)
and therefore the same total parameter count. The only difference is
whether the inner-loop TTT mechanism is active:
    --mode standard   : n_ttt_blocks=0, plain transformer
    --mode ttt        : n_ttt_blocks>0 with TTT inner loop
    --mode persistent : 
    --mode both       : run both back-to-back and plot the comparison

Example:
    python train.py                         # both modes, default config
    python train.py --max_steps 200         # shorter run
    python train.py --inner_lr 3.0          # sweep inner-loop lr
    python train.py --mode ttt --seq_len 512 --window_size 128
    python train.py --mode persistent --seq_len 256 --mini_batch_size 128

"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from ttt.ttt_model import (
    TTTTransformer,
    ttt_forward_and_loss,
    standard_forward_and_loss,
    persistent_ttt_forward_and_loss,
    persistent_eval_loss,
    snapshot_fast_params,
    get_non_fast_parameters,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tinystories(cache_dir="./data", max_chars=2_000_000):
    """
    Try HuggingFace datasets first; fall back to a synthetic repetitive text
    if datasets is unavailable or offline. Cached locally after first load.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / "tinystories_subset.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    try:
        from datasets import load_dataset
        print("Downloading TinyStories (validation split)...")
        ds = load_dataset("roneneldan/TinyStories", split="validation")
        text = "\n\n".join(ds["text"])[:max_chars]
        cache_file.write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"Could not download TinyStories ({e}). Using fallback sample text.")
        fallback = (
            "Once upon a time, there was a little girl named Lily. She lived in a "
            "small house with her mom and dad. One day, Lily found a small dog in "
            "the garden. The dog was very friendly and licked Lily's hand. Lily "
            "hugged the dog and took it home.\n\nHer mom and dad were happy to see "
            "the dog. They gave the dog food and water. The dog ate the food and "
            "drank the water. Then, the dog went to sleep on a soft blanket.\n\n"
            "The next day, Lily and the dog played in the garden. They ran around "
            "and had fun. Lily gave the dog a name. She called it Max. Max was "
            "happy with his new name. Lily and Max became best friends.\n\n"
        )
        text = fallback * 2000
        return text[:max_chars]


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return torch.tensor(
            [self.stoi[c] for c in text if c in self.stoi], dtype=torch.long
        )


class SeqDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start:start + self.seq_len]


def run_one(mode, config, tokenizer, train_tokens, eval_tokens, device):
    set_seed(config["seed"])

    n_ttt_blocks = config["n_ttt_blocks"] if mode in ("ttt", "persistent") else 0
    model = TTTTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ff_mult=config["ff_mult"],
        max_seq_len=config["seq_len"] + 16,
        window_size=config["window_size"],
        n_ttt_blocks=n_ttt_blocks,
    ).to(device)

    n_params = model.num_params()
    print(f"[{mode}] {n_params:,} parameters")

    # For persistent mode, fast params are updated by the inner loop only; exclude
    # them from the outer optimizer so AdamW's weight decay and momentum do not
    # fight the inner SGD.
    if mode == "persistent":
        opt_params = get_non_fast_parameters(model)
        print(f"[{mode}] optimizer sees {sum(p.numel() for p in opt_params):,} of "
              f"{n_params:,} params (fast MLPs excluded)")
    else:
        opt_params = list(model.parameters())

    optim = torch.optim.AdamW(
        opt_params,
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Persistent fast params state, if applicable. A separate inner_lr is used
    # because per-step magnitudes differ from bi-level TTT (these updates
    # accumulate rather than get thrown away).
    persistent_state = None
    if mode == "persistent":
        persistent_state = snapshot_fast_params(model.get_initial_fast_params())

    train_ds = SeqDataset(train_tokens, config["seq_len"])
    eval_ds = SeqDataset(eval_tokens, config["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=True)

    def eval_model(max_batches=16):
        nonlocal persistent_state
        model.eval()
        losses = []
        eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
        # For persistent mode: snapshot pre-eval state so eval adaptation does
        # not leak into training. Each eval sequence also starts from this
        # snapshot (per-sequence reset, matching TTT eval).
        pre_eval_snapshot = None
        if mode == "persistent":
            pre_eval_snapshot = snapshot_fast_params(persistent_state)
        for j, batch in enumerate(eval_loader):
            if j >= max_batches:
                break
            batch = batch.to(device)
            if mode == "ttt":
                loss = ttt_forward_and_loss(
                    model, batch,
                    config["mini_batch_size"], config["inner_lr"],
                    create_graph=False,
                )
                losses.append(float(loss.detach()))
            elif mode == "persistent":
                el = persistent_eval_loss(
                    model, batch,
                    config["mini_batch_size"],
                    config.get("persistent_inner_lr", config["inner_lr"]),
                    pre_eval_snapshot,
                )
                losses.append(el)
            else:
                with torch.no_grad():
                    loss = standard_forward_and_loss(model, batch)
                losses.append(float(loss.detach()))
        model.train()
        # Restore persistent state to its pre-eval snapshot (we never mutated
        # persistent_state during eval, since persistent_eval_loss snapshots
        # internally, but we re-snapshot to be explicit).
        if mode == "persistent":
            persistent_state = pre_eval_snapshot
        return sum(losses) / len(losses)

    model.train()
    t0 = time.time()
    log = []
    running = []
    step = 0
    iter_loader = iter(train_loader)

    # Baseline eval at step 0
    log.append({"step": 0, "eval_loss": eval_model(), "elapsed": 0.0})
    print(f"[{mode}] step     0 EVAL {log[0]['eval_loss']:.4f}")

    while step < config["max_steps"]:
        optim.zero_grad()
        accum = 0.0
        for _ in range(config["accum_steps"]):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)
            batch = batch.to(device)
            if mode == "ttt":
                loss = ttt_forward_and_loss(
                    model, batch,
                    config["mini_batch_size"], config["inner_lr"],
                    create_graph=True,
                )
            elif mode == "persistent":
                loss, persistent_state = persistent_ttt_forward_and_loss(
                    model, batch,
                    config["mini_batch_size"],
                    config.get("persistent_inner_lr", config["inner_lr"]),
                    persistent_state,
                    inner_weight_decay=config.get("persistent_inner_wd", 0.0),
                )
            else:
                loss = standard_forward_and_loss(model, batch)
            (loss / config["accum_steps"]).backward()
            accum += float(loss.detach()) / config["accum_steps"]
        torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
        optim.step()
        step += 1
        running.append(accum)

        if step % config["log_interval"] == 0:
            avg = sum(running) / len(running)
            elapsed = time.time() - t0
            extra = ""
            if mode == "persistent":
                norm = sum(p.norm().item() for fp in persistent_state for p in fp.values())
                extra = f" fast_norm {norm:.2f}"
            print(f"[{mode}] step {step:5d} train {avg:.4f} elapsed {elapsed:.1f}s{extra}")
            running = []

        if step % config["eval_interval"] == 0 or step == config["max_steps"]:
            el = eval_model()
            elapsed = time.time() - t0
            log.append({"step": step, "eval_loss": el, "elapsed": elapsed})
            print(f"[{mode}] step {step:5d} EVAL  {el:.4f} elapsed {elapsed:.1f}s")

    return {"params": n_params, "log": log}


def make_plot(results, modes, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(Plotting skipped: {e})")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"standard": "tab:orange", "ttt": "tab:blue", "persistent": "tab:green"}
    for mode in modes:
        log = results[mode]["log"]
        steps = [e["step"] for e in log]
        losses = [e["eval_loss"] for e in log]
        times = [e["elapsed"] for e in log]
        c = colors.get(mode, None)
        ax[0].plot(steps, losses, marker="o", label=mode, color=c)
        ax[1].plot(times, losses, marker="o", label=mode, color=c)
    ax[0].set_xlabel("outer step")
    ax[0].set_ylabel("eval NLL (per token)")
    ax[0].set_title("Loss vs step")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("wall-clock (s)")
    ax[1].set_ylabel("eval NLL (per token)")
    ax[1].set_title("Loss vs time")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "ttt", "persistent", "all"], default="all")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--mini_batch_size", type=int, default=32)
    ap.add_argument("--window_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--ff_mult", type=int, default=4)
    ap.add_argument("--n_ttt_blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--inner_lr", type=float, default=1.0,
                    help="Inner-loop SGD lr for TTT (bi-level) mode.")
    ap.add_argument("--persistent_inner_lr", type=float, default=0.1,
                    help="Inner-loop SGD lr for persistent mode. Smaller because "
                         "updates accumulate across sequences rather than being reset.")
    ap.add_argument("--persistent_inner_wd", type=float, default=0.0,
                    help="Optional weight decay for the persistent fast params' "
                         "inner SGD update. 0 = off.")
    ap.add_argument("--accum_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=25)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--max_chars", type=int, default=2_000_000)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    text = load_tinystories(max_chars=args.max_chars)
    print(f"Loaded {len(text):,} chars")

    tok = CharTokenizer(text)
    print(f"Vocab: {tok.vocab_size}")

    all_tokens = tok.encode(text)
    n_eval = min(len(all_tokens) // 20, 50_000)
    train_tokens = all_tokens[:-n_eval]
    eval_tokens = all_tokens[-n_eval:]
    print(f"Tokens: train {len(train_tokens):,}, eval {len(eval_tokens):,}")

    config = vars(args)
    results = {"config": config}

    if args.mode == "all":
        modes = ["standard", "ttt", "persistent"]
    else:
        modes = [args.mode]

    for mode in modes:
        print(f"\n=== Training: {mode} ===")
        results[mode] = run_one(mode, config, tok, train_tokens, eval_tokens, device)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {args.out}")

    plot_path = Path(args.out).with_suffix(".png")
    make_plot(results, modes, plot_path)

    print("\n=== Summary ===")
    for mode in modes:
        log = results[mode]["log"]
        final = log[-1]
        start = log[0]["eval_loss"]
        print(
            f"[{mode:11s}] eval NLL: {start:.4f} -> {final['eval_loss']:.4f}  "
            f"({final['step']} steps, {final['elapsed']:.1f}s)"
        )


if __name__ == "__main__":
    main()
