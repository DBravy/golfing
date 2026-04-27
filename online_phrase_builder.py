"""
Online phrase boundary computation from input token IDs.

The boundary rule:
  1. Token t is a phrase end if its decoded form ends in '.', '!', or '?'
     AND the previous token is not a known abbreviation.
  2. If max_phrase_len tokens have passed since the last boundary, force one.
  3. Force a boundary at the last non-pad position of each sequence.
  4. Tokens listed in extra_boundary_token_ids (e.g. EOS) also trigger
     boundaries.

build_token_lookups dispatches between Hugging Face tokenizers and
SentencePiece processors automatically.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

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

_SENTENCE_END_CHARS = {".", "!", "?"}
_STRIP_TRAILING = ' \t\n\r"\')]'


def _is_sentencepiece(tok) -> bool:
    """Detect a SentencePiece processor by duck typing."""
    return hasattr(tok, "id_to_piece") and hasattr(tok, "is_byte")


def get_vocab_size(tok) -> int:
    if _is_sentencepiece(tok):
        return int(tok.vocab_size())
    return int(tok.vocab_size)


def get_eos_id(tok) -> Optional[int]:
    if _is_sentencepiece(tok):
        eid = tok.eos_id()
        return int(eid) if eid is not None and eid >= 0 else None
    return getattr(tok, "eos_token_id", None)


def _classify_decoded(decoded: str, abbreviations: Set[str]
                      ) -> Tuple[bool, bool]:
    """Given a decoded piece, return (is_punct, is_abbreviation)."""
    if not decoded:
        return False, False
    stripped_right = decoded.rstrip(_STRIP_TRAILING)
    is_punct = bool(stripped_right) and stripped_right[-1] in _SENTENCE_END_CHARS

    cleaned = decoded.strip().strip(_STRIP_TRAILING).lower()
    cleaned_no_dot = cleaned.rstrip(".")
    is_abbr = (cleaned in abbreviations) or (cleaned_no_dot in abbreviations)
    return is_punct, is_abbr


def _build_lookups_hf(tokenizer, abbreviations: Set[str]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_size = int(tokenizer.vocab_size)
    is_punct = torch.zeros(vocab_size, dtype=torch.bool)
    is_abbreviation = torch.zeros(vocab_size, dtype=torch.bool)

    for tok_id in range(vocab_size):
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            continue
        p, a = _classify_decoded(decoded, abbreviations)
        if p:
            is_punct[tok_id] = True
        if a:
            is_abbreviation[tok_id] = True
    return is_punct, is_abbreviation


def _build_lookups_sp(sp, abbreviations: Set[str]
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab_size = int(sp.vocab_size())
    is_punct = torch.zeros(vocab_size, dtype=torch.bool)
    is_abbreviation = torch.zeros(vocab_size, dtype=torch.bool)

    for tok_id in range(vocab_size):
        # Skip special tokens: control (BOS/EOS/PAD), unknown, unused, byte fallback.
        if (sp.is_control(tok_id) or sp.is_unknown(tok_id)
                or sp.is_unused(tok_id) or sp.is_byte(tok_id)):
            continue
        piece = sp.id_to_piece(tok_id)
        # Strip the SP whitespace marker for content checks. Pieces consisting
        # only of '▁' have empty content and are skipped.
        content = piece.lstrip("▁")
        if not content:
            continue
        p, a = _classify_decoded(content, abbreviations)
        if p:
            is_punct[tok_id] = True
        if a:
            is_abbreviation[tok_id] = True
    return is_punct, is_abbreviation


def build_token_lookups(tokenizer,
                        abbreviations: Set[str] = DEFAULT_ABBREVIATIONS
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (is_punct, is_abbreviation) for any supported tokenizer."""
    if _is_sentencepiece(tokenizer):
        return _build_lookups_sp(tokenizer, abbreviations)
    return _build_lookups_hf(tokenizer, abbreviations)


# -------------------------------------------------------------------------
# is_end and packing (unchanged from previous version)
# -------------------------------------------------------------------------

def compute_is_end(input_ids: torch.Tensor,
                   is_punct: torch.Tensor,
                   is_abbreviation: torch.Tensor,
                   max_phrase_len: int,
                   pad_token_id: int) -> torch.Tensor:
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
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return phrase_mask, phrase_token_idx, phrase_end_pos, phrase_id


# -------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------

class OnlinePhraseBuilder(nn.Module):
    """
    Punctuation + abbreviation veto + forced-flush phrase builder.
    Accepts either a Hugging Face tokenizer or a SentencePiece processor.
    """
    def __init__(self,
                 tokenizer,
                 max_phrase_len: int,
                 pad_token_id: int,
                 abbreviations: Set[str] = DEFAULT_ABBREVIATIONS,
                 extra_boundary_token_ids: Optional[Iterable[int]] = None):
        super().__init__()
        self.max_phrase_len = max_phrase_len
        self.pad_token_id = pad_token_id

        is_punct, is_abbr = build_token_lookups(tokenizer, abbreviations)

        if extra_boundary_token_ids:
            for tid in extra_boundary_token_ids:
                if tid is None:
                    continue
                if 0 <= int(tid) < is_punct.numel():
                    is_punct[int(tid)] = True
                    is_abbr[int(tid)] = False

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
    """Fires a boundary every `stride` tokens. The V4-style baseline."""
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

        pos_arange = positions.unsqueeze(0).expand(B, T)
        last_real = torch.where(is_real, pos_arange,
                                torch.full_like(pos_arange, -1)).max(dim=1).values
        last_real_mask = (pos_arange == last_real.unsqueeze(1)) & is_real
        is_end = is_end | last_real_mask

        return pack_phrases_from_is_end(
            is_end, self.max_phrase_len, self.pad_token_id, input_ids,
        )


# -------------------------------------------------------------------------
# Smoke test (covers HF mock and SP mock paths)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # HF-style mock.
    class MockHFTokenizer:
        def __init__(self):
            self.id_to_str = {
                0: "[PAD]", 1: ".", 2: "!", 3: "?",
                4: "Dr", 5: "Mr", 6: "Smith", 7: "hello",
                8: "world", 9: "yes", 14: "<EOS>",
            }
            self.vocab_size = len(self.id_to_str) + 5  # padded to test bounds

        def decode(self, ids):
            return " ".join(self.id_to_str.get(i, "") for i in ids)

    # SentencePiece-style mock.
    class MockSP:
        # 0=<unk>, 1=<bos>, 2=<eos>, 3=<pad>, 4=▁hello, 5=▁world, 6=., 7=▁Dr, 8=!,
        # 9=<0xE2> (byte fallback), 10=▁Smith.
        def __init__(self):
            self.pieces = ["<unk>", "<bos>", "<eos>", "<pad>",
                           "▁hello", "▁world", ".", "▁Dr", "!",
                           "<0xE2>", "▁Smith."]
            self._control = {1, 2, 3}
            self._byte = {9}

        def vocab_size(self): return len(self.pieces)
        def is_control(self, i): return i in self._control
        def is_unknown(self, i): return i == 0
        def is_unused(self, i): return False
        def is_byte(self, i): return i in self._byte
        def id_to_piece(self, i): return self.pieces[i]
        def eos_id(self): return 2

    # HF lookups.
    hf = MockHFTokenizer()
    is_punct_hf, is_abbr_hf = build_token_lookups(hf)
    assert set(torch.where(is_punct_hf)[0].tolist()) == {1, 2, 3}
    assert set(torch.where(is_abbr_hf)[0].tolist()) == {4, 5}
    print(f"HF dispatch OK: punct={sorted(torch.where(is_punct_hf)[0].tolist())}")

    # SP lookups: should mark "." (id 6), "!" (id 8), and "▁Smith." (id 10) as punct.
    # Should mark "▁Dr" (id 7) as abbreviation.
    # Should NOT mark special tokens (0,1,2,3) or byte fallback (9).
    sp = MockSP()
    is_punct_sp, is_abbr_sp = build_token_lookups(sp)
    punct_ids = sorted(torch.where(is_punct_sp)[0].tolist())
    abbr_ids = sorted(torch.where(is_abbr_sp)[0].tolist())
    print(f"SP dispatch OK: punct={punct_ids} abbr={abbr_ids}")
    assert punct_ids == [6, 8, 10], f"got {punct_ids}"
    assert abbr_ids == [7], f"got {abbr_ids}"

    # End-to-end SP: build a sequence and verify boundaries fire correctly.
    builder = OnlinePhraseBuilder(
        sp, max_phrase_len=8, pad_token_id=-1,
        extra_boundary_token_ids=[2],  # EOS
    )
    # Sequence: "▁hello ▁world . <eos> ▁Dr . ▁Smith. !"
    # ids:     [   4,      5,    6,  2,    7,    6,   10,      8]
    # Expected boundaries:
    #   pos 2 (.)
    #   pos 3 (EOS)
    #   pos 5 (.) -- but veto'd because prev is "▁Dr" (abbreviation)
    #   pos 6 (▁Smith. -- ends with .)
    #   pos 7 (!)
    # plus forced last_real at pos 7 (already a boundary).
    seq = torch.tensor([[4, 5, 6, 2, 7, 6, 10, 8]])
    pm, pti, pep, pid = builder(seq)
    valid_ends = sorted(set(x for x in pep[0].tolist() if x >= 0))
    print(f"SP end-to-end: end_pos={valid_ends}")
    assert valid_ends == [2, 3, 6, 7], f"got {valid_ends}"

    print("\nAll tests passed.")
