"""
Unified V4-faithful Multi-Head Hybrid Attention.

A single attention call over the union of:
  * top-k retrieved compressed phrase entries (CSA path), bounded to n_csa lookback
  * the most recent n_win raw token K=V entries (sliding-window path)

One softmax denominator across both sources, exactly as V4 does. Differences
from the two-branch version in dynamic_csa.py:

  - Shared low-rank Q latent feeds both the indexer and the core MHA.
  - CSA compressor is multi-head: each phrase gets H summary vectors.
  - Sliding-window K=V is multi-head: produced from h via one projection.
  - K and V come from the same projection on each side; RoPE is applied only
    to the K-side use, V is the unrotated copy. Identical parameter count to
    full K=V sharing, cleaner semantics, no -i output compensation needed.
  - Single softmax over [csa_selected || sliding_window] keys.
  - CSA is bounded: a phrase is visible to query t iff
        t - n_csa <= phrase_end < t

Parameter count for D=512, H=8, c=64, d_c=256, Hi=2, ci=64, Lmax=16:
  ~1.4 M parameters per attention block.

Compared to vanilla MHA at the same D and H*c (~1.0 M), this is ~1.4x.
The extra ~0.4 M buys you the indexer, the per-head phrase compressor,
and the bounded retrieval path.
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
class UnifiedAttentionConfig:
    d_model: int = 512                # D
    # Core attention (multi-head, shared Q across both sources)
    n_query_heads: int = 8            # H
    head_dim: int = 64                # c
    query_compress_dim: int = 256     # d_c (low-rank latent)
    # Lightning indexer (single-source scoring; shared across H core heads)
    n_indexer_heads: int = 2          # Hi
    indexer_head_dim: int = 64        # ci
    # CSA path
    top_k: int = 32                   # k compressed entries selected per query
    n_csa: int = 1024                 # bounded lookback for CSA (in tokens)
    max_phrase_len: int = 16          # Lmax; longer phrases truncated
    # Sliding window
    n_win: int = 128                  # W (only used when sw_mode == "fixed")
    sw_mode: str = "fixed"            # "fixed" = banded causal, "phrase" = same-phrase causal
    # HCA (Heavily Compressed Attention) path: MQA over fixed-stride blocks.
    # Disabled when use_hca is False; enable to add a third concurrent path
    # alongside CSA and SW in the unified softmax.
    use_hca: bool = False
    hca_stride: int = 64              # m' (>> max_phrase_len). Tokens per HCA block.
    n_hca: int = -1                   # Bounded HCA lookback in tokens. -1 = unbounded.
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
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x and cos/sin must be broadcastable on the head_dim axis."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def apply_rope_uniform(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       positions: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE where every slot at the second-to-last axis uses the same
    position index from `positions`.
    x:         (..., T, head_dim)
    positions: (T,)
    """
    return _rope_rotate(x, cos[positions], sin[positions])


def apply_rope_per_slot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                        positions: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE where each slot has its own position.
    x:         (B, T, K, H, c)
    positions: (B, T, K)  long
    """
    cos_p = cos[positions].unsqueeze(-2)   # (B, T, K, 1, c)
    sin_p = sin[positions].unsqueeze(-2)
    return _rope_rotate(x, cos_p, sin_p)


def apply_rope_single_head_per_slot(x: torch.Tensor, cos: torch.Tensor,
                                    sin: torch.Tensor,
                                    positions: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE where each entry along dim 1 has its own position. No head dim.
    x:         (B, P, c)
    positions: (B, P)  long
    """
    cos_p = cos[positions]   # (B, P, c)
    sin_p = sin[positions]
    return _rope_rotate(x, cos_p, sin_p)


# -------------------------------------------------------------------------
# Multi-head phrase compressor: softmax-gated pooling, per head
# -------------------------------------------------------------------------

class MultiHeadPhraseCompressor(nn.Module):
    """
    Compresses variable-length phrases to one vector per (phrase, head).

    Output shape: (B, P, H, c)
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 max_phrase_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_phrase_len = max_phrase_len
        self.W_kv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.W_z = nn.Linear(d_model, n_heads * head_dim, bias=False)
        # Per-head position bias indexed by intra-phrase slot.
        self.B_pos = nn.Parameter(torch.zeros(max_phrase_len, n_heads, head_dim))
        nn.init.normal_(self.B_pos, std=0.02)

    def forward(self, h: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor) -> torch.Tensor:
        B, P, Lmax = phrase_mask.shape
        D = h.size(-1)
        H, c = self.n_heads, self.head_dim

        idx_flat = phrase_token_idx.reshape(B, P * Lmax)
        idx_exp = idx_flat.unsqueeze(-1).expand(-1, -1, D)
        h_phrase = h.gather(1, idx_exp).view(B, P, Lmax, D)

        c_tok = self.W_kv(h_phrase).view(B, P, Lmax, H, c)
        z_tok = self.W_z(h_phrase).view(B, P, Lmax, H, c)

        z_tok = z_tok + self.B_pos.view(1, 1, Lmax, H, c)

        mask_exp = phrase_mask.unsqueeze(-1).unsqueeze(-1)  # (B, P, Lmax, 1, 1)
        z_masked = z_tok.masked_fill(~mask_exp, float("-inf"))
        gates = F.softmax(z_masked, dim=2)                   # over Lmax

        any_token = phrase_mask.any(dim=2, keepdim=True) \
                               .unsqueeze(-1).unsqueeze(-1)
        gates = torch.where(any_token, gates, torch.zeros_like(gates))

        c_comp = (gates * c_tok).sum(dim=2)                  # (B, P, H, c)
        return c_comp


class SingleHeadPhraseCompressor(nn.Module):
    """
    Same structure but produces (B, P, ci) for the indexer key path.
    The indexer is intentionally narrow and shared across core heads.
    """
    def __init__(self, d_model: int, head_dim: int, max_phrase_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.W_kv = nn.Linear(d_model, head_dim, bias=False)
        self.W_z = nn.Linear(d_model, head_dim, bias=False)
        self.B_pos = nn.Parameter(torch.zeros(max_phrase_len, head_dim))
        nn.init.normal_(self.B_pos, std=0.02)

    def forward(self, h: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor) -> torch.Tensor:
        B, P, Lmax = phrase_mask.shape
        D = h.size(-1)

        idx_flat = phrase_token_idx.reshape(B, P * Lmax)
        idx_exp = idx_flat.unsqueeze(-1).expand(-1, -1, D)
        h_phrase = h.gather(1, idx_exp).view(B, P, Lmax, D)

        c_tok = self.W_kv(h_phrase)
        z_tok = self.W_z(h_phrase) + self.B_pos.view(1, 1, Lmax, -1)

        z_masked = z_tok.masked_fill(~phrase_mask.unsqueeze(-1), float("-inf"))
        gates = F.softmax(z_masked, dim=2)

        any_token = phrase_mask.any(dim=2, keepdim=True).unsqueeze(-1)
        gates = torch.where(any_token, gates, torch.zeros_like(gates))

        return (gates * c_tok).sum(dim=2)                    # (B, P, ci)


# -------------------------------------------------------------------------
# HCA (Heavily Compressed Attention) compressor
#
# Single-head, fixed-stride, no-overlap softmax-pooling compressor. Every
# `stride` consecutive tokens are folded into one KV vector. Tail tokens
# that don't form a complete block are dropped here; they are still seen
# by CSA and the sliding window, and a future persistent cache can absorb
# completed blocks at sequence end.
#
# Output shape: (B, P', c) where P' = T // stride.
# -------------------------------------------------------------------------

class FixedStrideHCACompressor(nn.Module):
    def __init__(self, d_model: int, head_dim: int, stride: int):
        super().__init__()
        self.head_dim = head_dim
        self.stride = stride
        self.W_kv = nn.Linear(d_model, head_dim, bias=False)
        self.W_z = nn.Linear(d_model, head_dim, bias=False)
        # Per-slot position bias indexed by intra-block offset.
        self.B_pos = nn.Parameter(torch.zeros(stride, head_dim))
        nn.init.normal_(self.B_pos, std=0.02)

    def forward(self, h: torch.Tensor):
        """
        h: (B, T, D)
        Returns:
          c_hca:   (B, P', c)         compressed single-head KV entries
          end_pos: (B, P')            last-token position of each block
        where P' = T // stride. May be 0 when T < stride.
        """
        B, T, D = h.shape
        m = self.stride
        P = T // m

        if P == 0:
            empty_kv = h.new_zeros(B, 0, self.head_dim)
            empty_pos = torch.zeros(B, 0, dtype=torch.long, device=h.device)
            return empty_kv, empty_pos

        # Truncate to complete blocks and reshape into (B, P, m, D).
        # `reshape` (rather than `view`) handles the non-contiguous slice that
        # results when T > P * m (i.e., when there are tail tokens to drop).
        h_full = h[:, :P * m, :].reshape(B, P, m, D)

        c_tok = self.W_kv(h_full)                           # (B, P, m, c)
        z_tok = self.W_z(h_full)                            # (B, P, m, c)
        z_tok = z_tok + self.B_pos.view(1, 1, m, -1)

        # Softmax over the m intra-block positions. Every block is fully real
        # (we already truncated to complete blocks), so no masking needed.
        gates = F.softmax(z_tok, dim=2)                     # (B, P, m, c)
        c_comp = (gates * c_tok).sum(dim=2)                 # (B, P, c)

        # End position of block i is (i + 1) * m - 1.
        end_pos = torch.arange(m - 1, P * m, m, device=h.device, dtype=torch.long)
        end_pos = end_pos.unsqueeze(0).expand(B, -1)        # (B, P)

        return c_comp, end_pos


# -------------------------------------------------------------------------
# Lightning indexer (unchanged in spirit; outputs one score per query/phrase)
# -------------------------------------------------------------------------

class LightningIndexer(nn.Module):
    def __init__(self, d_model: int, query_compress_dim: int,
                 n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.W_iuq = nn.Linear(query_compress_dim, n_heads * head_dim, bias=False)
        self.W_w = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, h: torch.Tensor, q_latent: torch.Tensor,
                k_comp: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        q_i = self.W_iuq(q_latent).view(B, T, self.n_heads, self.head_dim)
        scores_per_head = torch.einsum("bthd,bpd->bthp", q_i, k_comp)
        scores_per_head = F.relu(scores_per_head)
        w_h = self.W_w(h)                                        # (B, T, Hi)
        return (scores_per_head * w_h.unsqueeze(-1)).sum(dim=2)  # (B, T, P)


# -------------------------------------------------------------------------
# Unified V4-faithful MHA hybrid attention
# -------------------------------------------------------------------------

class UnifiedHybridAttention(nn.Module):
    """
    One MHA call over the union of CSA-selected compressed entries and the
    sliding window. Single softmax across both sources.
    """
    def __init__(self, cfg: UnifiedAttentionConfig):
        super().__init__()
        self.cfg = cfg
        D, H, c = cfg.d_model, cfg.n_query_heads, cfg.head_dim

        # Shared low-rank Q latent (consumed by indexer and core).
        self.W_dq = nn.Linear(D, cfg.query_compress_dim, bias=False)
        self.W_uq = nn.Linear(cfg.query_compress_dim, H * c, bias=False)

        # CSA path: per-head phrase compressor for the core, single-head for the indexer.
        self.csa_compressor = MultiHeadPhraseCompressor(D, H, c, cfg.max_phrase_len)
        self.idx_compressor = SingleHeadPhraseCompressor(D, cfg.indexer_head_dim,
                                                         cfg.max_phrase_len)
        self.indexer = LightningIndexer(D, cfg.query_compress_dim,
                                        cfg.n_indexer_heads,
                                        cfg.indexer_head_dim)

        # Sliding-window source: one projection from h, used as both K and V
        # (with RoPE applied only on the K-side).
        self.W_swkv = nn.Linear(D, H * c, bias=False)

        # HCA path: single-head fixed-stride compressor (MQA: 1 KV head shared
        # across H query heads). Optional.
        if cfg.use_hca:
            self.hca_compressor = FixedStrideHCACompressor(
                d_model=D, head_dim=c, stride=cfg.hca_stride,
            )
        else:
            self.hca_compressor = None

        # RMSNorms on Q and on K (one shared K norm covers both sources).
        self.q_norm = RMSNorm(c)
        self.k_norm = RMSNorm(c)

        self.W_o = nn.Linear(H * c, D, bias=False)

        if cfg.use_attn_sink:
            self.sink_logits = nn.Parameter(torch.zeros(H))
        else:
            self.register_parameter("sink_logits", None)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor,
                phrase_end_pos: torch.Tensor,
                phrase_id: torch.Tensor) -> torch.Tensor:
        """
        h:                (B, T, D) pre-normed input
        phrase_mask:      (B, P, Lmax) bool
        phrase_token_idx: (B, P, Lmax) long
        phrase_end_pos:   (B, P) long; -1 for padding phrases
        phrase_id:        (B, T) long; phrase index each token belongs to.
                          Used only when cfg.sw_mode == "phrase".

        Returns:          (B, T, D)
        """
        cfg = self.cfg
        B, T, _ = h.shape
        H, c = cfg.n_query_heads, cfg.head_dim

        # 1. RoPE cache covering all positions used.
        cos, sin = build_rope_cache(T, c, cfg.rope_base, h.device, h.dtype)
        positions = torch.arange(T, device=h.device)

        # 2. Shared Q latent and core queries.
        q_latent = self.W_dq(h)                                  # (B, T, d_c)
        q = self.W_uq(q_latent).view(B, T, H, c)                 # (B, T, H, c)
        q = self.q_norm(q)
        q = apply_rope_uniform(q.transpose(1, 2),                # rotate over T
                               cos, sin, positions).transpose(1, 2)

        # 3. CSA path: compressed entries (per-head) and indexer keys.
        c_comp = self.csa_compressor(h, phrase_mask, phrase_token_idx)  # (B, P, H, c)
        k_idx = self.idx_compressor(h, phrase_mask, phrase_token_idx)   # (B, P, ci)

        # 4. Indexer scores with bounded causal visibility.
        scores = self.indexer(h, q_latent, k_idx)                # (B, T, P)
        end_pos_b = phrase_end_pos.unsqueeze(1)                  # (B, 1, P)
        pos_b = positions.view(1, T, 1)
        vis = (end_pos_b < pos_b) & (end_pos_b >= pos_b - cfg.n_csa)
        scores = scores.masked_fill(~vis, float("-inf"))

        # 5. Top-k phrase selection.
        P = phrase_mask.size(1)
        k_select = min(cfg.top_k, P)
        top_scores, top_idx = scores.topk(k_select, dim=-1)      # (B, T, k)
        slot_valid = torch.isfinite(top_scores)                  # (B, T, k)

        # 6. Gather selected per-head compressed entries.
        # c_comp:   (B, P, H, c)   ->   expand to (B, T, P, H, c)
        c_comp_exp = c_comp.unsqueeze(1).expand(-1, T, -1, -1, -1)
        gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1) \
                            .expand(-1, -1, -1, H, c)
        csa_kv = c_comp_exp.gather(2, gather_idx)                # (B, T, k, H, c)
        csa_kv = self.k_norm(csa_kv)

        # 7. Sliding window K=V from h (multi-head). RoPE on the K-side use.
        sw_kv = self.W_swkv(h).view(B, T, H, c)                  # (B, T, H, c)
        sw_kv = self.k_norm(sw_kv)
        sw_k = apply_rope_uniform(sw_kv.transpose(1, 2),
                                  cos, sin, positions).transpose(1, 2)
        sw_v = sw_kv                                             # V unrotated

        # 8. RoPE on CSA keys: each slot uses its source phrase's end position.
        sel_end_pos = phrase_end_pos.gather(1, top_idx.reshape(B, -1)) \
                                    .view(B, T, k_select).clamp(min=0)
        csa_k = apply_rope_per_slot(csa_kv, cos, sin, sel_end_pos)
        csa_v = csa_kv                                           # V unrotated

        # 8b. HCA path: heavily compressed, MQA, dense.
        # Single-head KV of shape (B, P', c). Each query head dot-products this
        # via einsum broadcast (one KV head, H query heads = MQA).
        if self.hca_compressor is not None:
            hca_kv, hca_end_pos = self.hca_compressor(h)         # (B, P', c), (B, P')
            hca_kv = self.k_norm(hca_kv)
            hca_k = apply_rope_single_head_per_slot(
                hca_kv, cos, sin, hca_end_pos.clamp(min=0),
            )                                                    # (B, P', c)
            hca_v = hca_kv                                       # V unrotated
            P_hca = hca_kv.size(1)

            # Causal + bounded-lookback visibility: (B, T, P').
            hca_end_pos_b = hca_end_pos.unsqueeze(1)             # (B, 1, P')
            hca_pos_b = positions.view(1, T, 1)
            hca_vis = (hca_end_pos_b < hca_pos_b)
            if cfg.n_hca >= 0:
                hca_vis = hca_vis & (hca_end_pos_b >= hca_pos_b - cfg.n_hca)
        else:
            hca_kv = h.new_zeros(B, 0, c)
            hca_k = hca_kv
            hca_v = hca_kv
            hca_end_pos = torch.zeros(B, 0, dtype=torch.long, device=h.device)
            hca_vis = torch.zeros(B, T, 0, dtype=torch.bool, device=h.device)
            P_hca = 0

        # 9. Logits over all sources.
        # CSA logits: (B, T, H, k)
        csa_logits = torch.einsum("bthc,btkhc->bthk", q, csa_k) / math.sqrt(c)
        csa_logits = csa_logits.masked_fill(~slot_valid.unsqueeze(2),
                                            float("-inf"))

        # SW logits: (B, H, T, T) -> (B, T, H, T) for concat compatibility.
        sw_logits = torch.einsum("bthc,bshc->bhts", q, sw_k) / math.sqrt(c)

        # SW mask construction. Output shape: (B, T, T).
        i = positions.view(1, T, 1)
        j = positions.view(1, 1, T)
        causal = (j <= i)
        if cfg.sw_mode == "phrase":
            same_phrase = (phrase_id.unsqueeze(2) == phrase_id.unsqueeze(1))
            sw_mask = same_phrase & causal                       # (B, T, T)
        else:  # "fixed"
            band = (j > i - cfg.n_win)
            sw_mask = (causal & band).expand(B, -1, -1)          # (B, T, T)

        sw_logits = sw_logits.masked_fill(~sw_mask.unsqueeze(1),
                                          float("-inf"))
        sw_logits = sw_logits.permute(0, 2, 1, 3)                # (B, T, H, T)

        # HCA logits: (B, T, H, P'). MQA via broadcast over the H axis.
        # einsum: q is (B, T, H, c), hca_k is (B, P', c).
        hca_logits = torch.einsum("bthc,bpc->bthp", q, hca_k) / math.sqrt(c)
        hca_logits = hca_logits.masked_fill(~hca_vis.unsqueeze(2),
                                            float("-inf"))

        # 10. Concatenate slot axis and softmax once across all sources.
        combined = torch.cat([csa_logits, sw_logits, hca_logits],
                             dim=-1)                             # (B, T, H, k+T+P')

        if self.sink_logits is not None:
            sink = self.sink_logits.view(1, 1, H, 1).expand(B, T, H, 1)
            combined = torch.cat([combined, sink], dim=-1)
            attn = F.softmax(combined, dim=-1)[..., :-1]
        else:
            attn = F.softmax(combined, dim=-1)

        # If a query has no valid keys at all, zero its row to avoid NaNs.
        # any_valid is (B, T, 1, 1) after broadcasting across heads/keys.
        sw_any = sw_mask.any(dim=-1, keepdim=True)               # (B, T, 1)
        csa_any = slot_valid.any(dim=-1, keepdim=True)           # (B, T, 1)
        hca_any = hca_vis.any(dim=-1, keepdim=True)              # (B, T, 1)
        any_valid = (csa_any | sw_any | hca_any).unsqueeze(2)    # (B, T, 1, 1)
        attn = torch.where(any_valid, attn, torch.zeros_like(attn))
        attn = self.dropout(attn)

        # 11. Apply attention back to each source's V and sum.
        csa_attn = attn[..., :k_select]                              # (B, T, H, k)
        sw_attn = attn[..., k_select:k_select + T]                   # (B, T, H, T)
        hca_attn = attn[..., k_select + T:k_select + T + P_hca]      # (B, T, H, P')

        csa_out = torch.einsum("bthk,btkhc->bthc", csa_attn, csa_v)
        sw_out = torch.einsum("bths,bshc->bthc", sw_attn, sw_v)
        # MQA output: (B, T, H, P') @ (B, P', c) -> (B, T, H, c) via broadcast.
        hca_out = torch.einsum("bthp,bpc->bthc", hca_attn, hca_v)
        out = csa_out + sw_out + hca_out                         # (B, T, H, c)

        out = out.reshape(B, T, H * c)
        return self.W_o(out)


# -------------------------------------------------------------------------
# Block wrapper with pre-norm
# -------------------------------------------------------------------------

class UnifiedHybridAttentionBlock(nn.Module):
    def __init__(self, cfg: UnifiedAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.d_model)
        self.attn = UnifiedHybridAttention(cfg)

    def forward(self, x: torch.Tensor,
                phrase_mask: torch.Tensor,
                phrase_token_idx: torch.Tensor,
                phrase_end_pos: torch.Tensor,
                phrase_id: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x), phrase_mask, phrase_token_idx,
                         phrase_end_pos, phrase_id)


# -------------------------------------------------------------------------
# Phrase boundary -> tensor packing helper (same as before)
# -------------------------------------------------------------------------

def pack_phrases(phrase_spans_per_seq: List[List[Tuple[int, int]]],
                 seq_len: int,
                 max_phrase_len: int,
                 device: torch.device):
    B = len(phrase_spans_per_seq)
    P = max(1, max(len(s) for s in phrase_spans_per_seq))
    Lmax = max_phrase_len

    mask = torch.zeros(B, P, Lmax, dtype=torch.bool, device=device)
    tok_idx = torch.zeros(B, P, Lmax, dtype=torch.long, device=device)
    end_pos = torch.full((B, P), -1, dtype=torch.long, device=device)
    phrase_id = torch.zeros(B, seq_len, dtype=torch.long, device=device)

    for b, spans in enumerate(phrase_spans_per_seq):
        for p, (s, e) in enumerate(spans):
            e_trunc = min(e, s + Lmax, seq_len)
            length = e_trunc - s
            if length <= 0:
                continue
            mask[b, p, :length] = True
            tok_idx[b, p, :length] = torch.arange(s, e_trunc, device=device)
            end_pos[b, p] = e_trunc - 1
            phrase_id[b, s:e_trunc] = p
    return mask, tok_idx, end_pos, phrase_id


# -------------------------------------------------------------------------
# Parameter accounting
# -------------------------------------------------------------------------

def count_attention_params(cfg: UnifiedAttentionConfig) -> dict:
    """
    Returns a breakdown of parameter counts per sub-module.
    Useful for ablating H, c, d_c, Hi, ci against your competition budget.
    """
    D = cfg.d_model
    H, c = cfg.n_query_heads, cfg.head_dim
    d_c = cfg.query_compress_dim
    Hi, ci = cfg.n_indexer_heads, cfg.indexer_head_dim
    Lmax = cfg.max_phrase_len

    parts = {
        "W_dq (Q down)":          D * d_c,
        "W_uq (Q up)":            d_c * H * c,
        "W_iuq (idx Q up)":       d_c * Hi * ci,
        "W_w (idx head weight)":  D * Hi,
        "csa W_kv":               D * H * c,
        "csa W_z":                D * H * c,
        "csa B_pos":              Lmax * H * c,
        "idx W_kv":               D * ci,
        "idx W_z":                D * ci,
        "idx B_pos":              Lmax * ci,
        "W_swkv (SW K=V)":        D * H * c,
        "hca W_kv":               (D * c) if cfg.use_hca else 0,
        "hca W_z":                (D * c) if cfg.use_hca else 0,
        "hca B_pos":              (cfg.hca_stride * c) if cfg.use_hca else 0,
        "W_o":                    H * c * D,
        "q_norm + k_norm":        2 * c,
        "sink_logits":            H if cfg.use_attn_sink else 0,
        "pre-norm (block-level)": D,
    }
    parts["TOTAL"] = sum(v for k, v in parts.items() if k != "TOTAL")
    return parts


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = UnifiedAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8, n_csa=64,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16,
    )
    block = UnifiedHybridAttentionBlock(cfg)

    B, T = 2, 64
    x = torch.randn(B, T, cfg.d_model)

    spans_per_seq = [[(s, min(s + 3, T)) for s in range(0, T, 3)]
                     for _ in range(B)]
    mask, tok_idx, end_pos, pid = pack_phrases(spans_per_seq, T,
                                               cfg.max_phrase_len, x.device)
    out = block(x, mask, tok_idx, end_pos, pid)

    print("output shape:", tuple(out.shape))
    print("output mean/std:", round(out.mean().item(), 4),
                              round(out.std().item(), 4))

    # Causality check.
    x2 = x.clone()
    x2[:, T - 1, :] += 1.0
    out2 = block(x2, mask, tok_idx, end_pos, pid)
    diff = (out - out2).abs().max(dim=-1).values
    print("max diff at last token (should be > 0):",
          round(diff[:, -1].max().item(), 4))
    print("max diff over first half (should be 0):",
          round(diff[:, : T // 2].max().item(), 4))

    # Phrase-bounded SW sanity check.
    cfg_phrase = UnifiedAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8, n_csa=64,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16, sw_mode="phrase",
    )
    block_phrase = UnifiedHybridAttentionBlock(cfg_phrase)
    out_phrase = block_phrase(x, mask, tok_idx, end_pos, pid)
    print("phrase-mode output shape:", tuple(out_phrase.shape))

    # Parameter counts at the prototype config and at the recommended target.
    print("\nParameter breakdown at smoke-test config (D=128, H=4, c=32):")
    for k, v in count_attention_params(cfg).items():
        print(f"  {k:<28s} {v:>10,d}")

    target = UnifiedAttentionConfig(
        d_model=512, n_query_heads=8, head_dim=64,
        query_compress_dim=256, top_k=64, n_csa=1024,
        n_indexer_heads=2, indexer_head_dim=64,
        max_phrase_len=16, n_win=128,
    )
    print("\nParameter breakdown at target config (D=512, H=8, c=64):")
    for k, v in count_attention_params(target).items():
        print(f"  {k:<28s} {v:>10,d}")

    # ---------------------------------------------------------------------
    # HCA smoke test
    # ---------------------------------------------------------------------
    print("\n=== HCA smoke test ===")
    torch.manual_seed(0)
    cfg_hca = UnifiedAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8, n_csa=64,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16,
        use_hca=True, hca_stride=16, n_hca=-1,
    )
    block_hca = UnifiedHybridAttentionBlock(cfg_hca)

    B, T = 2, 64
    x = torch.randn(B, T, cfg_hca.d_model)
    spans = [[(s, min(s + 3, T)) for s in range(0, T, 3)] for _ in range(B)]
    mask, tok_idx, end_pos, pid = pack_phrases(spans, T,
                                               cfg_hca.max_phrase_len, x.device)
    out_hca = block_hca(x, mask, tok_idx, end_pos, pid)
    print("HCA output shape:", tuple(out_hca.shape))

    # Compressor sanity: T=64, stride=16 -> P'=4 blocks ending at 15,31,47,63.
    hca_kv, hca_end_pos = block_hca.attn.hca_compressor(x)
    print("HCA compressor output:", tuple(hca_kv.shape),
          "end_pos sample:", hca_end_pos[0].tolist())
    assert hca_kv.shape == (B, 4, cfg_hca.head_dim)
    assert hca_end_pos[0].tolist() == [15, 31, 47, 63]

    # Causality with HCA: perturb last token, check earlier outputs unchanged.
    x2 = x.clone()
    x2[:, T - 1, :] += 1.0
    out_hca2 = block_hca(x2, mask, tok_idx, end_pos, pid)
    diff = (out_hca - out_hca2).abs().max(dim=-1).values
    print("HCA max diff at last token (>0):",
          round(diff[:, -1].max().item(), 4))
    # Last block (positions 48..63) and any query at t >= 16 will see the
    # last completed block ending at 15, but blocks ending at 31/47/63 are
    # only visible to queries strictly later. Position 47 should not see the
    # block ending at 47. So position 47 is unaffected by the last-token
    # perturbation at t=63.
    print("HCA max diff at t=47 (should be 0):",
          round(diff[:, 47].max().item(), 4))
    print("HCA max diff over first half (should be 0):",
          round(diff[:, : T // 2].max().item(), 4))

    # Edge case: T < stride. Should produce 0 HCA blocks and not crash.
    cfg_short = UnifiedAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8, n_csa=64,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16,
        use_hca=True, hca_stride=64,
    )
    block_short = UnifiedHybridAttentionBlock(cfg_short)
    T_short = 32
    x_short = torch.randn(B, T_short, 128)
    spans_short = [[(s, min(s + 3, T_short)) for s in range(0, T_short, 3)]
                   for _ in range(B)]
    m2, ti2, ep2, pid2 = pack_phrases(spans_short, T_short,
                                      cfg_short.max_phrase_len, x_short.device)
    out_short = block_short(x_short, m2, ti2, ep2, pid2)
    print("Short-seq HCA output shape (P'=0):", tuple(out_short.shape))
    hca_kv_s, _ = block_short.attn.hca_compressor(x_short)
    assert hca_kv_s.shape == (B, 0, cfg_short.head_dim)

    # Bounded n_hca: queries beyond the lookback window should not see old blocks.
    cfg_bounded = UnifiedAttentionConfig(
        d_model=128, n_query_heads=4, head_dim=32,
        query_compress_dim=64, top_k=8, n_csa=64,
        n_indexer_heads=2, indexer_head_dim=32,
        max_phrase_len=8, n_win=16,
        use_hca=True, hca_stride=16, n_hca=20,
    )
    block_bnd = UnifiedHybridAttentionBlock(cfg_bounded)
    out_bnd = block_bnd(x, mask, tok_idx, end_pos, pid)
    print("Bounded-HCA output shape:", tuple(out_bnd.shape))

    # Parameter counts including HCA path.
    print("\nParameter breakdown with HCA enabled (D=128, H=4, c=32, m'=16):")
    for k, v in count_attention_params(cfg_hca).items():
        print(f"  {k:<28s} {v:>10,d}")

    # Backward pass test.
    out_hca.sum().backward()
    print("\nHCA backward pass OK")
