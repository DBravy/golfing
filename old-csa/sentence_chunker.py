"""
Sentence-boundary chunker for dynamic CSA.

Produces (token_ids, phrase_spans) where phrase_spans is a list of
(start, end_exclusive) pairs in token-index coordinates. Each span is bounded
by max_phrase_len; sentences longer than that get split into equal-sized
sub-chunks.

Usage:

    from transformers import AutoTokenizer
    from sentence_chunker import sentence_chunker, FixedStrideChunker

    tok = AutoTokenizer.from_pretrained("gpt2")
    text = "The cat sat. It was a sunny day. Birds were singing in the trees."
    token_ids, spans = sentence_chunker(text, tok, max_phrase_len=16)

    # spans is something like [(0, 4), (4, 10), (10, 17)]

The output format matches what `pack_phrases` in dynamic_csa_unified.py expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Protocol


# -------------------------------------------------------------------------
# Sentence splitter
# -------------------------------------------------------------------------

# A small set of common abbreviations whose trailing period should NOT be
# treated as a sentence end. Extend as needed for your domain.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt",
    "vs", "etc", "ie", "eg", "no", "vol", "fig", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    "u.s", "u.k", "u.n", "e.u",
}

# Sentence-ending punctuation followed by whitespace and a capital letter
# is a strong signal of a sentence break. We use this pattern to find
# candidate split points, then filter out abbreviations.
_CANDIDATE_BREAK = re.compile(r'([.!?]+)(\s+)(?=[A-Z"\'\(])')


def split_sentences(text: str) -> List[Tuple[int, int]]:
    """
    Returns a list of (char_start, char_end) sentence spans in `text`.

    Tries to handle common abbreviations. Not perfect; for production-grade
    splitting, swap in nltk.sent_tokenize or spacy's sentencizer.
    """
    if not text or not text.strip():
        return []

    spans: List[Tuple[int, int]] = []
    cursor = 0

    for match in _CANDIDATE_BREAK.finditer(text):
        punct_end = match.start() + len(match.group(1))      # end of "."/"!"/"?"
        # Look at the word ending right before the punctuation.
        # If it's a known abbreviation, skip this candidate.
        word_match = re.search(r'(\S+?)$', text[cursor:match.start()])
        prev_word = ""
        if word_match:
            prev_word = word_match.group(1).lower()
            # Strip leading punctuation/quotes from the word for comparison.
            prev_word = re.sub(r'^[^a-z0-9]+', '', prev_word)
        if prev_word in _ABBREVIATIONS:
            continue

        # Single uppercase letter followed by period (initials like "J. Smith").
        if re.match(r'^[a-z]$', prev_word):
            continue

        sentence_end = punct_end
        spans.append((cursor, sentence_end))
        cursor = match.end()  # skip past the whitespace after punctuation

    # Final sentence (or the whole string if no breaks found).
    tail_start = cursor
    tail_end = len(text)
    if tail_start < tail_end and text[tail_start:tail_end].strip():
        spans.append((tail_start, tail_end))

    return spans


# -------------------------------------------------------------------------
# Char span -> token span mapping
# -------------------------------------------------------------------------

def char_span_to_token_span(char_start: int, char_end: int,
                            offsets: List[Tuple[int, int]]
                            ) -> Tuple[int, int]:
    """
    Map a [char_start, char_end) span to a [tok_start, tok_end) span using
    the tokenizer's offset_mapping output.

    Tokens that are entirely inside [char_start, char_end) are included.
    Tokens that straddle a boundary are included if at least one of their
    characters is within the span.

    Special tokens (with offset (0, 0) but non-zero token index) are skipped.
    """
    tok_start = None
    tok_end = None
    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            # Special token; its position is not text-derived.
            continue
        # Token covers chars [a, b).
        if b <= char_start:
            continue
        if a >= char_end:
            break
        if tok_start is None:
            tok_start = i
        tok_end = i + 1
    if tok_start is None:
        return (0, 0)  # no overlap; caller should treat this as empty
    return (tok_start, tok_end)


# -------------------------------------------------------------------------
# Splitting long phrases into equal sub-chunks
# -------------------------------------------------------------------------

def split_long_phrase(start: int, end: int, max_phrase_len: int
                      ) -> List[Tuple[int, int]]:
    """
    If end - start > max_phrase_len, split into equal-sized sub-spans.
    The last sub-span absorbs the remainder.
    """
    length = end - start
    if length <= max_phrase_len:
        return [(start, end)]

    # Compute the number of sub-chunks needed.
    n_chunks = (length + max_phrase_len - 1) // max_phrase_len
    base = length // n_chunks
    remainder = length % n_chunks

    sub_spans = []
    cursor = start
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        sub_spans.append((cursor, cursor + size))
        cursor += size
    return sub_spans


# -------------------------------------------------------------------------
# Main chunker entry point
# -------------------------------------------------------------------------

def sentence_chunker(text: str,
                     tokenizer,
                     max_phrase_len: int = 16,
                     add_special_tokens: bool = False
                     ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Tokenize `text` and produce phrase spans aligned to sentence boundaries.

    Args:
        text: input string.
        tokenizer: a Hugging Face fast tokenizer (must support
            return_offsets_mapping=True).
        max_phrase_len: cap on tokens per phrase. Sentences longer than this
            are split into roughly equal sub-chunks.
        add_special_tokens: whether the tokenizer should add BOS/EOS/etc.
            If True, those tokens become singleton phrases.

    Returns:
        token_ids:    list of token ids.
        phrase_spans: list of (start, end_exclusive) in token-index space,
                      sorted and non-overlapping. Covers all non-special tokens.
                      Special tokens (if any) are emitted as singleton phrases.
    """
    enc = tokenizer(text,
                    return_offsets_mapping=True,
                    add_special_tokens=add_special_tokens)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    if not token_ids:
        return [], []

    # 1. Find sentence char-spans.
    sent_char_spans = split_sentences(text)
    if not sent_char_spans:
        # No sentences detected; fall back to one phrase covering everything.
        sent_char_spans = [(0, len(text))]

    # 2. Map each sentence to a token span.
    sent_token_spans: List[Tuple[int, int]] = []
    for cs, ce in sent_char_spans:
        ts, te = char_span_to_token_span(cs, ce, offsets)
        if te > ts:
            sent_token_spans.append((ts, te))

    # 3. Insert singleton phrases for any special tokens.
    special_token_phrases: List[Tuple[int, int]] = []
    for i, (a, b) in enumerate(offsets):
        if a == 0 and b == 0:
            special_token_phrases.append((i, i + 1))

    # 4. Insert singleton phrases for any non-special tokens left uncovered.
    #    This catches tokens between sentences (rare; usually whitespace gets
    #    folded into adjacent tokens, but be safe).
    covered = [False] * len(token_ids)
    for s, e in sent_token_spans:
        for i in range(s, e):
            covered[i] = True
    for s, e in special_token_phrases:
        for i in range(s, e):
            covered[i] = True

    gap_phrases: List[Tuple[int, int]] = []
    i = 0
    while i < len(token_ids):
        if not covered[i]:
            j = i
            while j < len(token_ids) and not covered[j]:
                j += 1
            gap_phrases.append((i, j))
            i = j
        else:
            i += 1

    # 5. Combine all spans, sort, and split anything longer than max_phrase_len.
    all_spans = sent_token_spans + special_token_phrases + gap_phrases
    all_spans.sort()

    final_spans: List[Tuple[int, int]] = []
    for s, e in all_spans:
        final_spans.extend(split_long_phrase(s, e, max_phrase_len))

    return token_ids, final_spans


# -------------------------------------------------------------------------
# Baselines: fixed-stride and per-sentence-as-singleton (for ablation)
# -------------------------------------------------------------------------

class Chunker(Protocol):
    def __call__(self, text: str, tokenizer, max_phrase_len: int
                 ) -> Tuple[List[int], List[Tuple[int, int]]]:
        ...


def fixed_stride_chunker(text: str,
                         tokenizer,
                         max_phrase_len: int = 16,
                         stride: int = 4,
                         add_special_tokens: bool = False
                         ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Splits the token stream into fixed `stride`-sized chunks. The V4 baseline.

    `max_phrase_len` is mostly a sanity guard here; if `stride > max_phrase_len`
    the stride is silently reduced.
    """
    enc = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = enc["input_ids"]
    T = len(token_ids)

    chunk_size = min(stride, max_phrase_len)
    if chunk_size < 1:
        chunk_size = 1

    spans = [(s, min(s + chunk_size, T)) for s in range(0, T, chunk_size)]
    return token_ids, spans


# -------------------------------------------------------------------------
# Validation helper
# -------------------------------------------------------------------------

def validate_spans(spans: List[Tuple[int, int]],
                   seq_len: int,
                   max_phrase_len: int) -> None:
    """
    Asserts the invariants the attention module assumes:
      - spans are sorted by start
      - spans are non-overlapping
      - all span ends fit in [1, seq_len]
      - all span lengths are in [1, max_phrase_len]
    """
    prev_end = 0
    for i, (s, e) in enumerate(spans):
        assert 0 <= s < e <= seq_len, \
            f"span {i} out of bounds: ({s}, {e}) for seq_len={seq_len}"
        assert e - s <= max_phrase_len, \
            f"span {i} too long: {e - s} > {max_phrase_len}"
        assert s >= prev_end, \
            f"span {i} overlaps previous: starts at {s}, prev ended at {prev_end}"
        prev_end = e


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the sentence splitter and span logic without requiring a real
    # tokenizer (we mock one with a simple whitespace tokenizer).

    class MockTokenizer:
        """A trivial whitespace tokenizer for offline testing."""
        def __call__(self, text, return_offsets_mapping=False,
                     add_special_tokens=False):
            tokens = []
            offsets = []
            i = 0
            while i < len(text):
                if text[i].isspace():
                    i += 1
                    continue
                start = i
                while i < len(text) and not text[i].isspace():
                    i += 1
                tokens.append(text[start:i])
                offsets.append((start, i))
            ids = list(range(len(tokens)))
            out = {"input_ids": ids}
            if return_offsets_mapping:
                out["offset_mapping"] = offsets
            return out

    tok = MockTokenizer()

    # Test 1: simple multi-sentence text.
    text1 = "The cat sat. It was a sunny day. Birds were singing in the trees."
    ids1, spans1 = sentence_chunker(text1, tok, max_phrase_len=16)
    print("Test 1: simple sentences")
    print(f"  tokens: {len(ids1)}")
    print(f"  spans:  {spans1}")
    validate_spans(spans1, len(ids1), max_phrase_len=16)
    print("  -> validated")

    # Test 2: abbreviation handling.
    text2 = "Dr. Smith arrived. He was tired."
    ids2, spans2 = sentence_chunker(text2, tok, max_phrase_len=16)
    print("\nTest 2: abbreviation handling")
    print(f"  tokens: {len(ids2)}")
    print(f"  spans:  {spans2}")
    # Should be 2 sentences, not 3.
    n_phrase_groups = len(spans2)
    assert n_phrase_groups == 2, \
        f"expected 2 phrases (abbreviation handled), got {n_phrase_groups}"
    print("  -> abbreviation correctly handled")

    # Test 3: long sentence that gets split.
    text3 = "This is a very long sentence with many words that should " \
            "definitely exceed the phrase length limit and force a split."
    ids3, spans3 = sentence_chunker(text3, tok, max_phrase_len=8)
    print("\nTest 3: long sentence split")
    print(f"  tokens: {len(ids3)}")
    print(f"  spans:  {spans3}")
    validate_spans(spans3, len(ids3), max_phrase_len=8)
    assert len(spans3) > 1, "long sentence should have been split"
    print(f"  -> split into {len(spans3)} sub-phrases")

    # Test 4: empty / edge cases.
    print("\nTest 4: edge cases")
    for edge_text in ["", "   ", "Hello.", "Yes."]:
        ids, spans = sentence_chunker(edge_text, tok, max_phrase_len=16)
        validate_spans(spans, len(ids), max_phrase_len=16)
        print(f"  '{edge_text}' -> {len(ids)} tokens, spans={spans}")

    # Test 5: fixed-stride baseline.
    text5 = "one two three four five six seven eight nine ten"
    ids5, spans5 = fixed_stride_chunker(text5, tok, max_phrase_len=16, stride=3)
    print("\nTest 5: fixed-stride baseline (stride=3)")
    print(f"  tokens: {len(ids5)}")
    print(f"  spans:  {spans5}")
    validate_spans(spans5, len(ids5), max_phrase_len=16)
    print("  -> validated")

    # Test 6: full coverage (every token is in exactly one span).
    text6 = "First sentence. Second sentence here. Third one."
    ids6, spans6 = sentence_chunker(text6, tok, max_phrase_len=16)
    covered = [False] * len(ids6)
    for s, e in spans6:
        for i in range(s, e):
            assert not covered[i], f"token {i} in multiple spans"
            covered[i] = True
    assert all(covered), f"uncovered tokens: {[i for i, c in enumerate(covered) if not c]}"
    print("\nTest 6: full coverage")
    print(f"  all {len(ids6)} tokens covered exactly once")

    print("\nAll tests passed.")
