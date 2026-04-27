"""
TinyStories dataset with on-startup tokenization and optional disk caching.

Replaces the old prepare_data.py workflow. Run training directly:
    python train.py ...

The dataset module handles tokenization and caching automatically. The first
run on a given (split, tokenizer, max_len, max_stories) combination downloads
the corpus and tokenizes it; subsequent runs load instantly from cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """In-memory list of token id tensors."""
    def __init__(self, input_ids_list, tokenizer_name: str):
        self.input_ids = input_ids_list
        self.tokenizer_name = tokenizer_name

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


def _cache_filename(split: str, tokenizer_name: str, max_len: int,
                    max_stories: Optional[int]) -> str:
    safe_tok = tokenizer_name.replace("/", "_")
    cap = "all" if max_stories is None else str(max_stories)
    return f"tinystories_{split}_{safe_tok}_len{max_len}_cap{cap}.pt"


def build_tinystories_dataset(split: str,
                              tokenizer,
                              max_len: int,
                              cache_dir: Optional[str] = None,
                              max_stories: Optional[int] = None,
                              tokenizer_name: Optional[str] = None
                              ) -> TokenizedDataset:
    """
    Load TinyStories, tokenize, and return an in-memory dataset.

    Args:
        split:         "train" or "validation".
        tokenizer:     HF tokenizer instance.
        max_len:       truncation length in tokens.
        cache_dir:     if given, cache tokenized output to disk for fast reload.
                       Pass None or empty to disable caching.
        max_stories:   if given, cap the number of stories (debug runs).
        tokenizer_name: name used in the cache filename. If None, derived
                       from tokenizer.name_or_path.

    Returns:
        TokenizedDataset.
    """
    if tokenizer_name is None:
        tokenizer_name = getattr(tokenizer, "name_or_path", None) or "unknown"

    # Try cache first.
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / _cache_filename(
            split, tokenizer_name, max_len, max_stories
        )
        if cache_path.exists():
            print(f"Loading tokenized {split} from cache: {cache_path}")
            blob = torch.load(cache_path, weights_only=False)
            return TokenizedDataset(blob["input_ids"], blob["tokenizer_name"])

    # Cache miss: download and tokenize.
    from datasets import load_dataset
    print(f"Downloading TinyStories ({split}) from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split=split)
    if max_stories is not None and len(ds) > max_stories:
        ds = ds.select(range(max_stories))

    print(f"Tokenizing {len(ds):,} stories...")

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch["text"], truncation=True, max_length=max_len,
            add_special_tokens=False,
        )
        return {"input_ids": encoded["input_ids"]}

    tokenized = ds.map(
        tokenize_batch, batched=True, batch_size=1024,
        remove_columns=ds.column_names,
        desc=f"tokenizing {split}",
    )

    input_ids_list = [
        torch.tensor(ids, dtype=torch.long)
        for ids in tokenized["input_ids"]
        if ids  # skip empty stories
    ]
    n_skipped = len(tokenized) - len(input_ids_list)

    print(f"Tokenized {len(input_ids_list):,} stories ({n_skipped:,} skipped)")
    lens = [len(x) for x in input_ids_list]
    if lens:
        med = sorted(lens)[len(lens) // 2]
        print(f"  token lengths: min={min(lens)} median={med} max={max(lens)}")

    # Save cache if requested.
    if cache_path is not None:
        torch.save({
            "input_ids": input_ids_list,
            "tokenizer_name": tokenizer_name,
        }, cache_path)
        size_mb = cache_path.stat().st_size / 1e6
        print(f"Cached to: {cache_path} ({size_mb:.1f} MB)")

    return TokenizedDataset(input_ids_list, tokenizer_name)
