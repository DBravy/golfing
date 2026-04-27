"""
Preprocess TinyStories for training: tokenize, run the sentence chunker,
and cache (input_ids, phrase_spans) to disk as a single .pt file per split.

Usage:
    python prepare_data.py --split train --max-stories 100000 --max-len 512
    python prepare_data.py --split validation --max-stories 1000 --max-len 512

Output files land in ./data/{split}.pt as a dict:
    {
        "input_ids":     list of 1-D LongTensors (variable length per story)
        "phrase_spans":  list of list of (start, end) tuples
        "tokenizer_name": str
        "max_phrase_len": int
    }

We don't pack examples or pre-pad here. Padding happens in the collator,
where we also clip phrase spans to fit any per-batch sequence length cap.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from sentence_chunker import sentence_chunker, validate_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--max-stories", type=int, default=100_000,
                    help="Cap the number of stories processed.")
    ap.add_argument("--max-len", type=int, default=512,
                    help="Truncate stories to this many tokens.")
    ap.add_argument("--max-phrase-len", type=int, default=16)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--chunker", choices=["sentence", "fixed"],
                    default="sentence",
                    help="Which chunker to use. 'fixed' is the V4-style baseline.")
    ap.add_argument("--fixed-stride", type=int, default=4,
                    help="Stride for the fixed chunker (only if --chunker=fixed).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading TinyStories split={args.split}")
    ds = load_dataset("roneneldan/TinyStories", split=args.split)
    if args.max_stories and len(ds) > args.max_stories:
        ds = ds.select(range(args.max_stories))

    if args.chunker == "sentence":
        chunker_fn = lambda txt: sentence_chunker(
            txt, tokenizer, max_phrase_len=args.max_phrase_len,
            add_special_tokens=False
        )
    else:
        from sentence_chunker import fixed_stride_chunker
        chunker_fn = lambda txt: fixed_stride_chunker(
            txt, tokenizer, max_phrase_len=args.max_phrase_len,
            stride=args.fixed_stride, add_special_tokens=False
        )

    all_input_ids = []
    all_spans = []

    n_skipped = 0
    n_long = 0
    truncated_phrases = 0

    for story in tqdm(ds, desc=f"chunking {args.split}"):
        text = story["text"]
        if not text or not text.strip():
            n_skipped += 1
            continue

        try:
            ids, spans = chunker_fn(text)
        except Exception as e:
            print(f"chunker error on story (skipping): {e}")
            n_skipped += 1
            continue

        # Truncate to max_len. Drop or truncate spans that fall outside.
        if len(ids) > args.max_len:
            n_long += 1
            ids = ids[:args.max_len]
            new_spans = []
            for s, e in spans:
                if s >= args.max_len:
                    break
                e_clip = min(e, args.max_len)
                if e_clip > s:
                    if e_clip < e:
                        truncated_phrases += 1
                    new_spans.append((s, e_clip))
            spans = new_spans

        # Sanity check.
        try:
            validate_spans(spans, len(ids), args.max_phrase_len)
        except AssertionError as e:
            print(f"validate_spans failed (skipping): {e}")
            n_skipped += 1
            continue

        all_input_ids.append(torch.tensor(ids, dtype=torch.long))
        all_spans.append(spans)

    print(f"\nProcessed: {len(all_input_ids):,} stories")
    print(f"Skipped:   {n_skipped:,}")
    print(f"Truncated long stories: {n_long:,}")
    print(f"Truncated phrases:      {truncated_phrases:,}")

    # Quick stats.
    lens = [len(x) for x in all_input_ids]
    n_phrases = [len(s) for s in all_spans]
    print(f"\nToken lengths:  min={min(lens)} med={sorted(lens)[len(lens)//2]} max={max(lens)}")
    print(f"Phrases/story:  min={min(n_phrases)} med={sorted(n_phrases)[len(n_phrases)//2]} max={max(n_phrases)}")

    out_path = out_dir / f"{args.split}.pt"
    torch.save({
        "input_ids": all_input_ids,
        "phrase_spans": all_spans,
        "tokenizer_name": args.tokenizer,
        "max_phrase_len": args.max_phrase_len,
        "chunker": args.chunker,
    }, out_path)
    print(f"\nWrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
