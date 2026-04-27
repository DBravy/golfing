"""
Single-machine token stream loader for binary shard files.

Replaces the HF DataLoader with sequential streaming over pre-tokenized shards
written by prepare_shards.py. Each batch is a contiguous slice of token IDs,
sized to fit batch_size sequences of seq_len, plus one extra token to support
the (x, y) shifted-label split.

Document boundaries (EOS tokens) are passed through as data; treat them as
phrase boundaries via OnlinePhraseBuilder's extra_boundary_token_ids.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def load_data_shard(file: Path) -> torch.Tensor:
    """Load a single shard file. Returns a uint16 tensor of token IDs."""
    header = np.fromfile(file, dtype="<i4", count=256)
    if (header.size != 256
            or int(header[0]) != SHARD_MAGIC
            or int(header[1]) != SHARD_VERSION):
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected = 256 * 4 + num_tokens * 2
    if file.stat().st_size != expected:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np)


class TokenStream:
    """Reads shards sequentially in filename order, wraps around forever."""
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No shard files matched: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        chunks = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class TokenLoader:
    """Yields (x, y) batches from a TokenStream. y is x shifted left by one."""
    def __init__(self, pattern: str, device: torch.device):
        self.stream = TokenStream(pattern)
        self.device = device

    def next_batch(self, batch_size: int, seq_len: int
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = batch_size * seq_len + 1
        chunk = self.stream.take(n).to(dtype=torch.int64)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))


def load_validation_tokens(pattern: str, seq_len: int) -> torch.Tensor:
    """Concatenate all val shards into one CPU tensor, sized to fit seq_len."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val shards matched: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation set too short for seq_len={seq_len}")
    return tokens[:usable + 1]
