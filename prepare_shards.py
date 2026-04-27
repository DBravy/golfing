"""
Tokenize a Hugging Face dataset and write binary token shards.

Format (matches modded-nanogpt / nanogpt-speedrun):
  Header: 256 int32 little-endian (1024 bytes total)
    [0]: magic = 20240520
    [1]: version = 1
    [2]: num_tokens in this shard
    [3:]: zeros (reserved)
  Body: num_tokens uint16 little-endian token IDs.

The --tokenizer flag accepts:
  - A Hugging Face tokenizer name (e.g. "gpt2"), or
  - A path to a SentencePiece .model file.

Vocabulary must fit in uint16 (<= 65535 tokens).

Usage:
    # GPT-2 BPE
    python prepare_shards.py --dataset roneneldan/TinyStories --split train \\
        --tokenizer gpt2 --out-dir ./data/datasets/tinystories_gpt2

    # Custom SentencePiece BPE
    python prepare_shards.py --dataset roneneldan/TinyStories --split train \\
        --tokenizer ./data/tokenizers/tinystories_8192_bpe.model \\
        --out-dir ./data/datasets/tinystories_sp8192
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Callable, List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def write_shard(path: Path, tokens: np.ndarray) -> None:
    if tokens.dtype != np.uint16:
        if tokens.max() > 65535 or tokens.min() < 0:
            raise ValueError(f"Token ids out of uint16 range in {path}")
        tokens = tokens.astype(np.uint16)
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


def load_tokenizer(spec: str) -> Tuple[Callable[[List[str]], List[List[int]]],
                                       int, int]:
    """
    Returns (encode_batch, vocab_size, eos_id).

    encode_batch: function from a list of strings to a list of token-id lists.
    """
    if spec.endswith(".model"):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise SystemExit(
                "sentencepiece is not installed. Run: pip install sentencepiece"
            ) from e
        sp = spm.SentencePieceProcessor(model_file=spec)
        vocab_size = int(sp.vocab_size())
        eos_id = int(sp.eos_id())
        if eos_id < 0:
            raise ValueError(f"SentencePiece model {spec} has no EOS token.")
        # SP encode supports a list directly.
        def encode_batch(texts):
            return sp.encode(texts, out_type=int)
        return encode_batch, vocab_size, eos_id
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec)
        vocab_size = int(tok.vocab_size)
        eos_id = tok.eos_token_id
        if eos_id is None:
            raise ValueError(f"Tokenizer {spec} has no EOS token.")
        def encode_batch(texts):
            return tok(texts, add_special_tokens=False)["input_ids"]
        return encode_batch, vocab_size, int(eos_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF name or path to a .model SentencePiece file.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--tokens-per-shard", type=int, default=10_000_000)
    ap.add_argument("--max-stories", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or args.dataset.replace("/", "_")

    print(f"Loading tokenizer: {args.tokenizer}")
    encode_batch, vocab_size, eos_id = load_tokenizer(args.tokenizer)
    if vocab_size > 65535:
        raise ValueError(f"Vocab size {vocab_size} exceeds uint16 max")
    print(f"  vocab_size={vocab_size}  eos_id={eos_id}")

    print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_stories and len(ds) > args.max_stories:
        ds = ds.select(range(args.max_stories))
    print(f"  {len(ds):,} documents")

    print("Tokenizing...")
    def tokenize_batch(batch):
        return {"ids": encode_batch(batch["text"])}

    tokenized = ds.map(
        tokenize_batch, batched=True, batch_size=1024,
        remove_columns=ds.column_names, desc="tokenizing",
    )

    target = args.tokens_per_shard
    buf = np.empty(target + 65536, dtype=np.uint16)
    buf_pos = 0
    shard_idx = 0
    total_tokens = 0

    pbar = tqdm(tokenized, desc="writing")
    for record in pbar:
        ids = record["ids"]
        n = len(ids) + 1

        if buf_pos + n > buf.shape[0]:
            grown = np.empty(buf_pos + n + 65536, dtype=np.uint16)
            grown[:buf_pos] = buf[:buf_pos]
            buf = grown

        if ids:
            buf[buf_pos:buf_pos + len(ids)] = ids
            buf_pos += len(ids)
        buf[buf_pos] = eos_id
        buf_pos += 1
        total_tokens += n

        if buf_pos >= target:
            shard_path = out_dir / f"{prefix}_{args.split}_{shard_idx:06d}.bin"
            write_shard(shard_path, buf[:buf_pos])
            pbar.set_postfix({"shard": shard_idx, "tokens": f"{total_tokens:,}"})
            shard_idx += 1
            buf_pos = 0

    if buf_pos > 0:
        shard_path = out_dir / f"{prefix}_{args.split}_{shard_idx:06d}.bin"
        write_shard(shard_path, buf[:buf_pos])
        shard_idx += 1

    print(f"\nWrote {shard_idx} shard(s), {total_tokens:,} tokens total.")
    print(f"Pattern: {out_dir}/{prefix}_{args.split}_*.bin")


if __name__ == "__main__":
    main()
