"""
Online phrase boundary computation from input token IDs.

The boundary rule:
  1. Token t is a phrase end if its decoded form ends in '.', '!', or '?'
     AND the previous token is not a known abbreviation.
  2. If max_phrase_len tokens have passed since the last boundary, force one.
  3. Force a boundary at the last non-pad position of each sequence.

Outputs match what UnifiedHybridAttention expects:
    phrase_mask:      (B, P, Lmax) bool
    phrase_token_idx: (B, P, Lmax) long
    phrase_end_pos:   (B, P) long; -1 for padding phrases

This file also provides FixedStridePhraseBuilder for the fixed-stride ablation
baseline. Both builders are nn.Modules with the same forward signature, so
they're swappable in the model.
"""

from __future__ import annotations

from typing import Set, Tuple

import torch
import torch.nn as nn


DEFAULT_ABBREVIATIONS: Set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt",
    "vs", "etc", "ie", "eg", "no", "vol", "fig", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "e.u",
    "a.m", "p.m",
}


def build_token_lookups(tokenizer,
                        abbreviations: Set[str] = DEFAULT_ABBREVIATIONS
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (is_punct, is_abbreviation) boolean lookup tensors over the vocab."""
    vocab_size = tokenizer.vocab_size
    is_punct = torch.zeros(vocab_size, dtype=torch.bool)
    is_abbreviation = torch.zeros(vocab_size, dtype=torch.bool)

    sentence_end_chars = {".", "!", "?"}
    strip_trailing = ' \t\n\r"\')]'

    for tok_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        if not decoded:
            continue

        stripped_right = decoded.rstrip(strip_trailing)
        if stripped_right and stripped_right[-1] in sentence_end_chars:
            is_punct[tok_id] = True

        cleaned = decoded.strip().strip(strip_trailing).lower()
        cleaned_no_dot = cleaned.rstrip(".")
        if cleaned in abbreviations or cleaned_no_dot in abbreviations:
            is_abbreviation[tok_id] = True

    return is_punct, is_abbreviation


def compute_is_end(input_ids: torch.Tensor,
                   is_punct: torch.Tensor,
                   is_abbreviation: torch.Tensor,
                   max_phrase_len: int,
                   pad_token_id: int) -> torch.Tensor:
    """Apply the boundary rule. Returns is_end: (B, T) bool."""
    B, T = input_ids.shape
    device = input_ids.device

    punct_at_t = is_punct[input_ids]

    prev_ids = torch.cat([
        torch.zeros(B, 1, dtype=input_ids.dtype, device=device),
        input_ids[:, :-1],
    ], dim=1)
    prev_is_abbr = is_abbreviation[prev_ids]
    prev_is_abbr[:, 0] = False

    is_end = punct_at_t & (~prev_is_abbr)

    is_real = (input_ids != pad_token_id)
    pos_arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    last_real = torch.where(is_real, pos_arange,
                            torch.full_like(pos_arange, -1)).max(dim=1).values

    pad_mask = ~is_real
    is_end = is_end & (~pad_mask)

    last_boundary = torch.full((B,), -1, dtype=torch.long, device=device)
    for t in range(T):
        at_last = (last_real == t)
        forced = ((t - last_boundary) >= max_phrase_len)

        fire = is_end[:, t] | at_last | forced
        fire = fire & (t <= last_real)
        is_end[:, t] = fire

        last_boundary = torch.where(fire,
                                    torch.full_like(last_boundary, t),
                                    last_boundary)

    return is_end


def pack_phrases_from_is_end(is_end: torch.Tensor,
                             max_phrase_len: int,
                             pad_token_id: int,
                             input_ids: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert is_end[B, T] boolean to packed phrase tensors."""
    B, T = is_end.shape
    device = is_end.device

    is_end_long = is_end.long()
    phrase_id = torch.zeros_like(is_end_long)
    phrase_id[:, 1:] = torch.cumsum(is_end_long[:, :-1], dim=1)

    P = int(phrase_id.max().item()) + 1
    pos_arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    first_pos = torch.full((B, P), T, dtype=torch.long, device=device)
    first_pos.scatter_reduce_(
        dim=1, index=phrase_id, src=pos_arange, reduce="amin", include_self=True,
    )

    slot = pos_arange - first_pos.gather(1, phrase_id)

    is_real = (input_ids != pad_token_id)
    in_range = (slot < max_phrase_len) & (slot >= 0) & is_real

    phrase_mask = torch.zeros(B, P, max_phrase_len, dtype=torch.bool, device=device)
    phrase_token_idx = torch.zeros(B, P, max_phrase_len, dtype=torch.long, device=device)

    valid_b, valid_t = torch.where(in_range)
    valid_p = phrase_id[valid_b, valid_t]
    valid_slot = slot[valid_b, valid_t]

    phrase_mask[valid_b, valid_p, valid_slot] = True
    phrase_token_idx[valid_b, valid_p, valid_slot] = valid_t

    phrase_end_pos = torch.full((B, P), -1, dtype=torch.long, device=device)
    end_src = torch.where(in_range, pos_arange, torch.full_like(pos_arange, -1))
    phrase_end_pos.scatter_reduce_(
        dim=1, index=phrase_id, src=end_src, reduce="amax", include_self=True,
    )

    return phrase_mask, phrase_token_idx, phrase_end_pos


# -------------------------------------------------------------------------
# Builders (nn.Modules with a uniform forward(input_ids) interface)
# -------------------------------------------------------------------------

class OnlinePhraseBuilder(nn.Module):
    """
    Punctuation + abbreviation veto + forced-flush phrase builder.
    Sentence-aware. The primary chunker.
    """
    def __init__(self,
                 tokenizer,
                 max_phrase_len: int,
                 pad_token_id: int,
                 abbreviations: Set[str] = DEFAULT_ABBREVIATIONS):
        super().__init__()
        self.max_phrase_len = max_phrase_len
        self.pad_token_id = pad_token_id

        is_punct, is_abbr = build_token_lookups(tokenizer, abbreviations)
        self.register_buffer("is_punct", is_punct, persistent=False)
        self.register_buffer("is_abbreviation", is_abbr, persistent=False)

    def forward(self, input_ids: torch.Tensor):
        is_end = compute_is_end(
            input_ids, self.is_punct, self.is_abbreviation,
            self.max_phrase_len, self.pad_token_id,
        )
        return pack_phrases_from_is_end(
            is_end, self.max_phrase_len, self.pad_token_id, input_ids,
        )


class FixedStridePhraseBuilder(nn.Module):
    """
    Fires a boundary every `stride` tokens. The V4-style baseline.
    No tokenizer needed.
    """
    def __init__(self,
                 max_phrase_len: int,
                 pad_token_id: int,
                 stride: int = 4):
        super().__init__()
        self.max_phrase_len = max_phrase_len
        self.pad_token_id = pad_token_id
        self.stride = min(stride, max_phrase_len)

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        device = input_ids.device

        is_real = (input_ids != self.pad_token_id)

        positions = torch.arange(T, device=device)
        is_stride_end = ((positions + 1) % self.stride == 0)
        is_stride_end = is_stride_end.unsqueeze(0).expand(B, T)
        is_end = is_stride_end & is_real

        # Mirror the online builder: ensure the trailing partial phrase has
        # an end at the last real position.
        pos_arange = positions.unsqueeze(0).expand(B, T)
        last_real = torch.where(is_real, pos_arange,
                                torch.full_like(pos_arange, -1)).max(dim=1).values
        last_real_mask = (pos_arange == last_real.unsqueeze(1)) & is_real
        is_end = is_end | last_real_mask

        return pack_phrases_from_is_end(
            is_end, self.max_phrase_len, self.pad_token_id, input_ids,
        )


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    class MockTokenizer:
        def __init__(self):
            self.id_to_str = {
                0: "[PAD]", 1: ".", 2: "!", 3: "?",
                4: "Dr", 5: "Mr", 6: "Smith", 7: "hello",
                8: "world", 9: "yes", 10: "and", 11: "the",
                12: "cat", 13: "sat",
            }
            self.vocab_size = len(self.id_to_str)

        def decode(self, ids):
            return " ".join(self.id_to_str.get(i, "?") for i in ids)

    tok = MockTokenizer()
    is_punct, is_abbr = build_token_lookups(tok)
    assert set(torch.where(is_punct)[0].tolist()) == {1, 2, 3}
    assert set(torch.where(is_abbr)[0].tolist()) == {4, 5}
    print("Lookup tables OK")

    # Existing rule tests (kept compact).
    seq = torch.tensor([[7, 8, 1, 9, 1]])
    is_end = compute_is_end(seq, is_punct, is_abbr, 8, 0)
    assert is_end[0].tolist() == [False, False, True, False, True]

    seq = torch.tensor([[4, 1, 6, 1, 9, 1]])
    is_end = compute_is_end(seq, is_punct, is_abbr, 8, 0)
    assert is_end[0].tolist() == [False, False, False, True, False, True]
    print("Punctuation + abbreviation veto OK")

    seq = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 7, 8, 9]])
    is_end = compute_is_end(seq, is_punct, is_abbr, 4, 0)
    assert is_end[0, 3].item() and is_end[0, 7].item() and is_end[0, 9].item()
    print("Forced flush OK")

    # OnlinePhraseBuilder end-to-end.
    builder = OnlinePhraseBuilder(tok, max_phrase_len=8, pad_token_id=0)
    seq = torch.tensor([[7, 8, 1, 9, 1, 0, 0]])
    pm, pti, pep = builder(seq)
    valid = (pep[0] >= 0).sum().item()
    assert valid == 2, f"OnlineBuilder: expected 2 valid phrases, got {valid}"
    print(f"OnlinePhraseBuilder OK: end_pos={[x for x in pep[0].tolist() if x >= 0]}")

    # FixedStridePhraseBuilder end-to-end.
    fixed = FixedStridePhraseBuilder(max_phrase_len=8, pad_token_id=0, stride=4)
    seq = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 0, 0]])
    pm, pti, pep = fixed(seq)
    valid_ends = sorted(set(x for x in pep[0].tolist() if x >= 0))
    assert valid_ends == [3, 7, 9], f"got {valid_ends}"
    print(f"FixedStridePhraseBuilder OK: end_pos={valid_ends}")

    print("\nAll tests passed.")
