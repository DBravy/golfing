"""
Dynamic Compressed Sparse Attention with Phrase-Level Chunking.

A simplified, research-grade implementation that combines:
  * Sliding-window multi-head attention (MHA) over the most recent n_win tokens
  * Compressed Sparse Attention (CSA) over phrase-compressed older tokens

Departures from DeepSeek-V4 CSA, made for clarity:
  - Sliding window is a separate MHA branch, not folded into CSA's MQA core.
  - Single-stream compression (no overlapping C^a / C^b dual stream).
  - Single output projection per branch (no grouped output projection).
  - Full RoPE on Q and compressed K (not partial last-64-dim RoPE).
  - mHC, FP4/FP8, and most stability tricks dropped. Attention sink is included
    as an option since it's cheap; toggle with `use_attn_sink`.

What is kept that matters for the paper's effect:
  - RMSNorm on Q and compressed K before core attention.
  - Lightning indexer: low-rank query, multi-head, ReLU per-head dot product,
    head-weighted sum with W_w, top-k selection.
  - Softmax-gated pooling with learnable per-position bias inside each phrase.
  - MQA core attention over selected compressed entries.
  - Strict causal masking on phrase visibility (a phrase is visible only to
    queries at or beyond its end token).

Key shapes used throughout:
  B   = batch
  T   = sequence length (in tokens)
  D   = model hidden size
  H   = number of query heads (core MQA queries)
  Hi  = number of indexer heads
  c   = head dim for core attention KV (and Q after up-projection)
  ci  = head dim for indexer
  P   = max number of phrases per sequence in the batch
  Lmax= max tokens per phrase (max_phrase_len)
  k   = number of compressed entries selected by the indexer
  W   = sliding window size (n_win)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

@dataclass
class HybridAttentionConfig:
    d_model: int = 512
    # Core (CSA) attention
    n_query_heads: int = 8           # H
    head_dim: int = 64               # c
    query_compress_dim: int = 256    # d_c (low-rank latent for queries)
    top_k: int = 32                  # k compressed entries selected per query
    # Lightning indexer
    n_indexer_heads: int = 4         # Hi
    indexer_head_dim: int = 64       # ci
    # Phrase compression
    max_phrase_len: int = 16         # Lmax; longer phrases get truncated
    # Sliding window
    n_win: int = 128                 # W
    sw_n_heads: int = 8              # heads for the sliding-window MHA branch
    # Misc
    use_attn_sink: bool = True
    rope_base: float = 10_000.0
    dropout: float = 0.0


# -------------------------------------------------------------------------
# RMSNorm and RoPE helpers
# -------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, base: float,
                     device: torch.device, dtype: torch.dtype):
    """Returns cos, sin tensors of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)             # (T, half)
    emb = torch.cat([freqs, freqs], dim=-1)      # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               positions: torch.Tensor) -> torch.Tensor:
    """
    x:         (..., T, head_dim)
    cos, sin:  (T_max, head_dim) full cache
    positions: (T,) integer positions to look up
    """
    cos_p = cos[positions]
    sin_p = sin[positions]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos_p + rotated * sin_p


# -------------------------------------------------------------------------
# Phrase compressor: softmax-gated pooling with learnable position bias
# -------------------------------------------------------------------------

class PhraseCompressor(nn.Module):
    """
    Compresses variable-length phrases to a single KV entry each.

    Inputs:
        h:                (B, T, D) hidden states
        phrase_mask:      (B, P, Lmax) bool, True where a phrase token exists
        phrase_token_idx: (B, P, Lmax) long, index into T for each (phrase, slot);
                          masked positions hold any valid index (e.g. 0); use mask.

    Output:
        c_comp:           (B, P, c) compressed KV entries (one per phrase)
    """
    def __init__(self, d_model: int, head_dim: int, max_phrase_len: int):
        super().__init__()
        self.W_kv = nn.Linear(d_model, head_dim, bias=False)
        self.W_z = nn.Linear(d_model, head_dim, bias=False)
        # Learnable position bias indexed by intra-phrase slot position.
        self.B_pos = nn.Parameter(torch.zeros(max_phrase_len, head_dim))
        nn.init.normal_(self.B_pos, std=0.02)
        self.max_phrase_len = max_phrase_len

    def forward(self, h: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor) -> torch.Tensor:
        B, P, Lmax = phrase_mask.shape
        D = h.size(-1)
        c = self.W_kv.out_features

        # Gather token-level features per (phrase, slot).
        # phrase_token_idx: (B, P, Lmax) -> flat: (B, P*Lmax)
        idx_flat = phrase_token_idx.reshape(B, P * Lmax)
        idx_exp = idx_flat.unsqueeze(-1).expand(-1, -1, D)        # (B, P*Lmax, D)
        h_phrase = h.gather(1, idx_exp).view(B, P, Lmax, D)       # (B, P, Lmax, D)

        c_tok = self.W_kv(h_phrase)                               # (B, P, Lmax, c)
        z_tok = self.W_z(h_phrase)                                # (B, P, Lmax, c)

        # Add positional bias along the intra-phrase dimension.
        z_tok = z_tok + self.B_pos.view(1, 1, Lmax, c)

        # Softmax over slot dimension (within each phrase), masking pads.
        z_masked = z_tok.masked_fill(~phrase_mask.unsqueeze(-1), float("-inf"))
        gates = F.softmax(z_masked, dim=2)                        # (B, P, Lmax, c)

        # Empty phrases would yield NaN; replace those rows with zeros.
        any_token = phrase_mask.any(dim=2, keepdim=True).unsqueeze(-1)
        gates = torch.where(any_token, gates, torch.zeros_like(gates))

        c_comp = (gates * c_tok).sum(dim=2)                       # (B, P, c)
        return c_comp


# -------------------------------------------------------------------------
# Lightning indexer: low-rank queries, multi-head ReLU dot product, head-weighted
# -------------------------------------------------------------------------

class LightningIndexer(nn.Module):
    """
    Scores each (query token, compressed phrase) pair.

    Inputs:
        h:        (B, T, D) hidden states for queries (used for W_w only)
        q_latent: (B, T, d_c) low-rank query latent shared with core MQA queries
        k_comp:   (B, P, ci) compressed indexer keys

    Output:
        scores:   (B, T, P) real-valued scores (higher = more relevant)
    """
    def __init__(self, d_model: int, query_compress_dim: int,
                 n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Up-project from shared latent to multi-head indexer queries.
        self.W_iuq = nn.Linear(query_compress_dim, n_heads * head_dim, bias=False)
        # Per-head weight produced from h directly.
        self.W_w = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, h: torch.Tensor, q_latent: torch.Tensor,
                k_comp: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        q_i = self.W_iuq(q_latent).view(B, T, self.n_heads, self.head_dim)
        # Per-head dot product: (B, T, H, ci) x (B, P, ci) -> (B, T, H, P)
        scores_per_head = torch.einsum("bthd,bpd->bthp", q_i, k_comp)
        scores_per_head = F.relu(scores_per_head)
        # Per-head weights w(h) and weighted sum across heads.
        w_h = self.W_w(h)                                          # (B, T, H)
        scores = (scores_per_head * w_h.unsqueeze(-1)).sum(dim=2)  # (B, T, P)
        return scores


# -------------------------------------------------------------------------
# CSA branch: indexer + top-k + MQA core attention over selected compressed KVs
# -------------------------------------------------------------------------

class CompressedSparseAttention(nn.Module):
    def __init__(self, cfg: HybridAttentionConfig):
        super().__init__()
        self.cfg = cfg
        D, H, c = cfg.d_model, cfg.n_query_heads, cfg.head_dim

        # Shared low-rank query latent (used by indexer and core MQA).
        self.W_dq = nn.Linear(D, cfg.query_compress_dim, bias=False)

        # Up-projection to core MQA queries.
        self.W_uq = nn.Linear(cfg.query_compress_dim, H * c, bias=False)

        # Compressors: one for the core KV, one for the indexer key path.
        self.kv_compressor = PhraseCompressor(D, c, cfg.max_phrase_len)
        self.idx_k_compressor = PhraseCompressor(D, cfg.indexer_head_dim,
                                                 cfg.max_phrase_len)

        # Indexer.
        self.indexer = LightningIndexer(D, cfg.query_compress_dim,
                                        cfg.n_indexer_heads,
                                        cfg.indexer_head_dim)

        # RMSNorms on queries and on compressed K.
        self.q_norm = RMSNorm(c)
        self.k_norm = RMSNorm(c)

        # Output projection back to D.
        self.W_o = nn.Linear(H * c, D, bias=False)

        # Optional attention sink (one learnable logit per head).
        if cfg.use_attn_sink:
            self.sink_logits = nn.Parameter(torch.zeros(H))
        else:
            self.register_parameter("sink_logits", None)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor,
                phrase_end_pos: torch.Tensor,
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        """
        h:                (B, T, D)
        phrase_mask:      (B, P, Lmax) bool
        phrase_token_idx: (B, P, Lmax) long
        phrase_end_pos:   (B, P) long, last token index of each phrase (inclusive);
                          for padding/empty phrases use -1 so they're never visible.
        Returns:          (B, T, D) attention output for this branch.
        """
        cfg = self.cfg
        B, T, _ = h.shape
        H, c = cfg.n_query_heads, cfg.head_dim

        # 1. Compressed entries and indexer keys.
        c_comp = self.kv_compressor(h, phrase_mask, phrase_token_idx)   # (B, P, c)
        k_idx = self.idx_k_compressor(h, phrase_mask, phrase_token_idx) # (B, P, ci)

        # 2. Shared low-rank query latent.
        q_latent = self.W_dq(h)                                          # (B, T, d_c)

        # 3. Indexer scores -> top-k compressed indices per query.
        scores = self.indexer(h, q_latent, k_idx)                        # (B, T, P)

        # Causal phrase visibility: a phrase is visible at query t iff
        # phrase_end_pos < t. Tokens within the phrase itself are handled by
        # the sliding-window branch.
        positions = torch.arange(T, device=h.device)
        vis = phrase_end_pos.unsqueeze(1) < positions.view(1, T, 1)      # (B, T, P)
        scores = scores.masked_fill(~vis, float("-inf"))

        P = phrase_mask.size(1)
        k_select = min(cfg.top_k, P)
        top_scores, top_idx = scores.topk(k_select, dim=-1)              # (B, T, k)
        # Identify queries with no visible phrase at all (early tokens).
        has_any = torch.isfinite(top_scores).any(dim=-1, keepdim=True)   # (B, T, 1)

        # 4. Gather selected compressed entries.
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, c)         # (B, T, k, c)
        c_comp_exp = c_comp.unsqueeze(1).expand(-1, T, -1, -1)           # (B, T, P, c)
        k_sel = c_comp_exp.gather(2, gather_idx)                         # (B, T, k, c)

        # 5. Core MQA attention.
        q = self.W_uq(q_latent).view(B, T, H, c)                         # (B, T, H, c)
        q = self.q_norm(q)
        k_sel = self.k_norm(k_sel)

        # RoPE on queries (their absolute positions).
        q = apply_rope(q.transpose(1, 2),                                # (B, H, T, c)
                       rope_cos, rope_sin, positions).transpose(1, 2)

        # RoPE on compressed keys: use the end position of the source phrase.
        sel_end_pos = phrase_end_pos.gather(1, top_idx.reshape(B, -1)) \
                                    .view(B, T, k_select)
        sel_end_pos = sel_end_pos.clamp(min=0)  # safe lookup; masked out below.

        # Apply RoPE per slot. Flatten the k dim for cache lookup.
        k_flat = k_sel.reshape(-1, c)                                    # (B*T*k, c)
        pos_flat = sel_end_pos.reshape(-1)                               # (B*T*k,)
        cos_p = rope_cos[pos_flat]                                       # (B*T*k, c)
        sin_p = rope_sin[pos_flat]
        half = c // 2
        x1, x2 = k_flat[:, :half], k_flat[:, half:]
        k_rot = torch.cat([-x2, x1], dim=-1)
        k_flat = k_flat * cos_p + k_rot * sin_p
        k_sel = k_flat.view(B, T, k_select, c)

        # MQA: H query heads attend to a single shared K (=V) per slot.
        logits = torch.einsum("bthc,btkc->bthk", q, k_sel) / math.sqrt(c)

        # Mask invalid top-k slots (those that came from -inf scores).
        slot_valid = torch.isfinite(top_scores)                          # (B, T, k)
        logits = logits.masked_fill(~slot_valid.unsqueeze(2), float("-inf"))

        # Optional attention sink: one learnable per-head sink logit added to denom.
        if self.sink_logits is not None:
            sink = self.sink_logits.view(1, 1, H, 1).expand(B, T, H, 1)
            logits_with_sink = torch.cat([logits, sink], dim=-1)
            attn = F.softmax(logits_with_sink, dim=-1)[..., :-1]
        else:
            attn = F.softmax(logits, dim=-1)

        # If a query had no visible phrase, attn rows can be NaN; zero them.
        attn = torch.where(has_any.unsqueeze(-1), attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        # Apply attention to V (= compressed K, MQA-style).
        out = torch.einsum("bthk,btkc->bthc", attn, k_sel)               # (B, T, H, c)
        out = out.reshape(B, T, H * c)
        return self.W_o(out)


# -------------------------------------------------------------------------
# Sliding-window MHA branch (separate, plain multi-head attention)
# -------------------------------------------------------------------------

class SlidingWindowMHA(nn.Module):
    def __init__(self, cfg: HybridAttentionConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        n_heads = cfg.sw_n_heads
        # Per-head dim matches CSA head_dim by default; you can decouple if desired.
        self.head_dim = cfg.head_dim
        self.n_heads = n_heads

        self.W_q = nn.Linear(D, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(D, n_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(D, n_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, D, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor,
                rope_cos: torch.Tensor,
                rope_sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        H, c = self.n_heads, self.head_dim
        W = self.cfg.n_win

        q = self.W_q(h).view(B, T, H, c)
        k = self.W_k(h).view(B, T, H, c)
        v = self.W_v(h).view(B, T, H, c)
        q = self.q_norm(q)
        k = self.k_norm(k)

        positions = torch.arange(T, device=h.device)
        q = apply_rope(q.transpose(1, 2), rope_cos, rope_sin, positions).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2), rope_cos, rope_sin, positions).transpose(1, 2)

        # Sliding-window causal mask: query t attends to keys in (t-W, t].
        i = positions.view(T, 1)
        j = positions.view(1, T)
        sw_mask = (j <= i) & (j > i - W)                                 # (T, T)

        logits = torch.einsum("bthc,bshc->bhts", q, k) / math.sqrt(c)
        logits = logits.masked_fill(~sw_mask.view(1, 1, T, T), float("-inf"))
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhts,bshc->bthc", attn, v)
        out = out.reshape(B, T, H * c)
        return self.W_o(out)


# -------------------------------------------------------------------------
# Combined block: sliding window + dynamic CSA, summed
# -------------------------------------------------------------------------

class HybridAttentionBlock(nn.Module):
    """
    Output = SlidingWindowMHA(norm(x)) + CSA(norm(x), phrase_info)

    Pre-norm; this returns just the attention output. Wrap in a Transformer
    block with residual + FFN as usual.
    """
    def __init__(self, cfg: HybridAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.d_model)
        self.sw = SlidingWindowMHA(cfg)
        self.csa = CompressedSparseAttention(cfg)

    def forward(self, x: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor,
                phrase_end_pos: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        T = h.size(1)
        cos, sin = build_rope_cache(T, self.cfg.head_dim,
                                    self.cfg.rope_base, h.device, h.dtype)
        out_sw = self.sw(h, cos, sin)
        out_csa = self.csa(h, phrase_mask, phrase_token_idx,
                           phrase_end_pos, cos, sin)
        return out_sw + out_csa


# -------------------------------------------------------------------------
# Phrase boundary -> tensor packing helper
# -------------------------------------------------------------------------

def pack_phrases(phrase_spans_per_seq: List[List[Tuple[int, int]]],
                 seq_len: int,
                 max_phrase_len: int,
                 device: torch.device):
    """
    Convert per-sequence phrase spans into the padded tensors expected by
    HybridAttentionBlock.

    Args:
        phrase_spans_per_seq: list of length B; each element is a list of
            (start, end_exclusive) token-index tuples for that sequence.
            Spans must be sorted by start, non-overlapping. Gaps are allowed
            (those tokens just won't appear in any compressed entry).
        seq_len: T
        max_phrase_len: cap on tokens per phrase. Phrases longer than this are
            truncated to their first max_phrase_len tokens.
        device: target device.

    Returns:
        phrase_mask:      (B, P, Lmax) bool
        phrase_token_idx: (B, P, Lmax) long
        phrase_end_pos:   (B, P) long; -1 for padding phrases
    """
    B = len(phrase_spans_per_seq)
    P = max(1, max(len(s) for s in phrase_spans_per_seq))
    Lmax = max_phrase_len

    mask = torch.zeros(B, P, Lmax, dtype=torch.bool, device=device)
    tok_idx = torch.zeros(B, P, Lmax, dtype=torch.long, device=device)
    end_pos = torch.full((B, P), -1, dtype=torch.long, device=device)

    for b, spans in enumerate(phrase_spans_per_seq):
        for p, (s, e) in enumerate(spans):
            e_trunc = min(e, s + Lmax, seq_len)
            length = e_trunc - s
            if length <= 0:
                continue
            mask[b, p, :length] = True
            tok_idx[b, p, :length] = torch.arange(s, e_trunc, device=device)
            end_pos[b, p] = e_trunc - 1
    return mask, tok_idx, end_pos


# -------------------------------------------------------------------------
# Smoke test: run a forward pass and a basic causality check.
# -------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = HybridAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16, sw_n_heads=4,
    )
    block = HybridAttentionBlock(cfg)

    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model)

    # Stand-in chunker: split each sequence into fixed 3-token spans.
    spans_per_seq = []
    for _ in range(B):
        spans = [(s, min(s + 3, T)) for s in range(0, T, 3)]
        spans_per_seq.append(spans)

    mask, tok_idx, end_pos = pack_phrases(spans_per_seq, T,
                                          cfg.max_phrase_len, x.device)
    print("phrase tensor shapes:",
          tuple(mask.shape), tuple(tok_idx.shape), tuple(end_pos.shape))

    out = block(x, mask, tok_idx, end_pos)
    print("output shape:", tuple(out.shape))
    print("output mean/std:", round(out.mean().item(), 4),
                              round(out.std().item(), 4))

    # Causality check: perturb the LAST token; earlier outputs should not change.
    x2 = x.clone()
    x2[:, T - 1, :] += 1.0
    out2 = block(x2, mask, tok_idx, end_pos)
    diff = (out - out2).abs().max(dim=-1).values            # (B, T)
    print("max diff at last token (should be > 0):", diff[:, -1].max().item())
    print("max diff over first half (should be 0):",
          diff[:, : T // 2].max().item())
