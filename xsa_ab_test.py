"""
XSA A/B test on a stripped-down version of the competition leader's
architecture. Runs baseline vs XSA back-to-back with identical seeds and
identical batch streams, prints training curves, and reports whether XSA
produces a detectable signal at the chosen scale.

What's kept from the leader (because it matters for the XSA effect):
  - 11-layer encoder-decoder split with skip connections + skip_gates
  - resid_mix (per-block blend of running residual and post-embedding x0)
  - Grouped query attention (8 heads, 4 KV heads)
  - Partial-dim RoPE (16 of 64 head dims rotated)
  - q_gain parameter, ln_scale, logit_softcap
  - Muon optimizer for matrix params, Adam for embed/scalar
  - Parallel residual starting at a configurable layer
  - XSA implementation matching the leader's _xsa_efficient

What's dropped (irrelevant to A/B):
  - Distributed training, GPTQ quantization, serialization
  - TTT, ETLB, sliding window eval
  - Layer looping (not needed for the comparison)
  - EMA weights (would dampen the very signal we're looking for at short runs)
  - flash_attn_3 -> replaced with torch SDPA

Data: WikiText-103 via HuggingFace parquet, tokenized with tiktoken GPT-2.
Cached to disk after first run. ~100M tokens, plenty for a short A/B.

Usage:
  # Full default A/B: 2 seeds x 2 modes, 1000 iters each
  python xsa_ab_test.py

  # Single-seed smoke test
  python xsa_ab_test.py --seeds 42 --iters 500

  # Longer run if you have time
  python xsa_ab_test.py --iters 3000
"""

import argparse
import copy
import json
import math
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = "wikitext-103"
_PQ_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/refs%2Fconvert%2Fparquet/wikitext-103-v1"


def low_precision_dtype():
    """Pick the best low-precision dtype for this GPU.
    Ampere+ (SM 8.0+) has native bf16. Turing (T4, SM 7.5) does not and
    falls back to fp32 with bf16, which is slow and VRAM-hungry. Use fp16
    on pre-Ampere cards instead."""
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


LOW_P_DTYPE = None  # populated in main() once CUDA is initialized


# ==================== Config ====================

@dataclass
class Config:
    # Model (matches leader defaults where possible)
    vocab_size: int = 50257       # gpt-2 tiktoken
    seq_len: int = 512            # reduced from 2048 for consumer GPU
    num_layers: int = 11
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: float = 4.0
    rope_base: float = 1e4
    rope_dims: int = 16           # partial-dim RoPE like leader
    qk_gain_init: float = 5.0
    logit_softcap: float = 30.0
    ln_scale: bool = True
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    parallel_residual_start: int = 7
    skip_gates_enabled: bool = True

    # Attention interventions (all composable with XSA)
    xsa_enabled: bool = False              # A/B knob: XSA
    sparsemax_k_enabled: bool = False      # strong: simplex projection on K
    sparsemax_k_mask_enabled: bool = False # soft: use sparsemax as mask only
    dg_k_enabled: bool = False             # DG-parallel: expand-sparsify-compress

    # DG-parallel hyperparams (only used when dg_k_enabled=True)
    dg_expansion_mult: int = 4   # expand head_dim -> expansion_mult * head_dim
    dg_sparsity_frac: float = 0.05  # fraction of expanded units active (top-k)

    # Training
    iters: int = 1000
    train_batch_tokens: int = 32768   # 32k tokens per step (T4-friendly)
    grad_accum_steps: int = 4
    warmup_steps: int = 20
    warmdown_frac: float = 0.5
    min_lr: float = 0.0
    grad_clip_norm: float = 0.3

    # Optimizers
    matrix_lr: float = 0.022
    scalar_lr: float = 0.02
    tied_embed_lr: float = 0.03
    muon_momentum: float = 0.99
    muon_momentum_warmup_start: float = 0.92
    muon_momentum_warmup_steps: int = 500
    muon_backend_steps: int = 5
    muon_wd: float = 0.095
    adam_wd: float = 0.02
    embed_wd: float = 0.085
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8

    # Eval
    val_every: int = 250
    val_batches: int = 16

    # Diagnostics
    log_attn_bias: bool = True    # log <y_i, v_i> per layer during training


# ==================== Data ====================

def get_data(seq_len: int):
    """
    Download WikiText-103 parquet if missing, tokenize with tiktoken gpt2,
    cache as .bin. Returns (train_tokens, val_tokens) as torch long tensors.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "tokens_gpt2.npz")
    if os.path.exists(cache):
        print(f"Loading cached tokens from {cache}")
        data = np.load(cache)
        return (torch.from_numpy(data["train"].astype(np.int64)),
                torch.from_numpy(data["val"].astype(np.int64)))

    import tiktoken
    tok = tiktoken.get_encoding("gpt2")

    import pandas as pd
    all_tokens = {}
    for split, hf_split in [("train", "train"), ("val", "validation")]:
        print(f"Downloading WikiText-103 {split} split...")
        # WikiText-103 train is multi-shard; val is single.
        if split == "train":
            shards = []
            for i in range(2):   # 2 shards for wikitext-103 train
                url = f"{_PQ_BASE}/{hf_split}/000{i}.parquet"
                pq = os.path.join(DATA_DIR, f"{split}_{i}.parquet")
                try:
                    with urllib.request.urlopen(url) as resp:
                        with open(pq, "wb") as fh:
                            shutil.copyfileobj(resp, fh)
                    shards.append(pq)
                except Exception as e:
                    if i == 0:
                        raise
                    print(f"  shard {i} not available ({e}), stopping")
                    break
            df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
            for p in shards:
                os.remove(p)
        else:
            url = f"{_PQ_BASE}/{hf_split}/0000.parquet"
            pq = os.path.join(DATA_DIR, f"{split}.parquet")
            with urllib.request.urlopen(url) as resp:
                with open(pq, "wb") as fh:
                    shutil.copyfileobj(resp, fh)
            df = pd.read_parquet(pq)
            os.remove(pq)

        print(f"  tokenizing {split} ({len(df)} rows)...")
        texts = df["text"].tolist()
        del df  # free the dataframe memory before encoding

        # Chunked tokenization to avoid OOM on low-RAM boxes. tiktoken builds
        # intermediate buffers proportional to input size; chunking caps peak
        # memory at ~chunk_bytes * constant rather than the full 515MB.
        CHUNK = 2000  # rows per tiktoken call
        all_ids = []
        for i in range(0, len(texts), CHUNK):
            chunk_text = "\n\n".join(texts[i:i + CHUNK])
            all_ids.extend(tok.encode_ordinary(chunk_text))
            if (i // CHUNK) % 50 == 0:
                print(f"    ...{i}/{len(texts)} rows, {len(all_ids):,} tokens")
        del texts
        all_tokens[split] = np.array(all_ids, dtype=np.int32)
        del all_ids
        print(f"  {split}: {all_tokens[split].size:,} tokens")

    np.savez(cache, train=all_tokens["train"], val=all_tokens["val"])
    return (torch.from_numpy(all_tokens["train"].astype(np.int64)),
            torch.from_numpy(all_tokens["val"].astype(np.int64)))


class BatchSampler:
    """Deterministic sampler: same seed produces the same batch stream.
    Crucial for a fair A/B: both modes must see identical batches."""
    def __init__(self, data, seq_len, batch_size, seed):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self):
        ix = torch.randint(
            len(self.data) - self.seq_len - 1,
            (self.batch_size,), generator=self.gen,
        )
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y


# ==================== Model ====================

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    """Partial-dim RoPE: only the first `rope_dims` of each head get rotated."""
    def __init__(self, head_dim, base=1e4, rope_dims=0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_dims = rope_dims if rope_dims > 0 else head_dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = None

    def forward(self, seq_len, device, dtype):
        if self._cache is None or self._cache[0].size(1) != seq_len \
                or self._cache[0].device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            cos = freqs.cos()[None, :, None, :]
            sin = freqs.sin()[None, :, None, :]
            self._cache = (cos, sin)
        c, s = self._cache
        return c.to(dtype), s.to(dtype)


def apply_rotary(x, cos, sin, rope_dims):
    """Rotate the first `rope_dims` dims of each head; pass the rest through."""
    if 0 < rope_dims < x.size(-1):
        x_r, x_p = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_r[..., :half], x_r[..., half:]
        x_r = torch.cat((x1*cos + x2*sin, -x1*sin + x2*cos), dim=-1)
        return torch.cat((x_r, x_p), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1*cos + x2*sin, -x1*sin + x2*cos), dim=-1)


def sparsemax(z, dim=-1):
    """
    Sparsemax (Martins & Astudillo 2016): projection onto the probability
    simplex that produces truly-zero entries along `dim`. Differentiable via
    autograd through the sort/cumsum/clamp; the active-support selection has
    zero gradient, which is correct per the sparsemax Jacobian.
    """
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    d = orig_shape[-1]
    z_flat = z.reshape(-1, d)
    z_sorted, _ = torch.sort(z_flat, dim=-1, descending=True)
    cumsum = torch.cumsum(z_sorted, dim=-1)
    range_vec = torch.arange(
        1, d + 1, dtype=z.dtype, device=z.device
    ).unsqueeze(0)
    support = (1 + range_vec * z_sorted) > cumsum
    k_support = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(-1, k_support - 1) - 1) / k_support.to(z.dtype)
    out_flat = torch.clamp(z_flat - tau, min=0)
    return out_flat.reshape(orig_shape).transpose(dim, -1)


def sparsemax_mask(k, dim=-1):
    """
    Soft variant of sparsemax-on-K: use sparsemax to pick the support set
    (which positions are nonzero), but keep the original values where the
    mask is 1. Preserves sign and magnitude; only sparsifies.

    Implementation: sparsemax on |k| tells us which positions sparsemax would
    keep. We gate the original k by that support. No gradient flows through
    the masking decision (boolean threshold), but gradients flow cleanly
    through the retained positions.
    """
    with torch.no_grad():
        support = (sparsemax(k.abs(), dim=dim) > 0).to(k.dtype)
    return k * support


def topk_sparse_ste(x, k, dim=-1):
    """
    Top-k sparsification with straight-through estimator.
    Forward: keep only the k largest entries along `dim`, zero the rest.
    Backward: pass gradients through as if identity (STE), biased but
    empirically fine for this kind of selection. Differentiable alternative
    to iterative competitive dynamics.
    """
    # Use the non-differentiable path for the forward, then plumb gradients
    # through the identity via x - x.detach() + masked_x.detach() trick.
    with torch.no_grad():
        vals, idx = torch.topk(x, k, dim=dim)
        mask = torch.zeros_like(x)
        mask.scatter_(dim, idx, 1.0)
    # masked_x has gradient 1 w.r.t. the retained positions, 0 elsewhere.
    # This is the standard STE for discrete selection.
    return x * mask


class CausalSelfAttention(nn.Module):
    """
    GQA causal attention with:
      - RMSNorm on Q and K before attention
      - Partial-dim RoPE
      - Per-head q_gain (learned scalar per query head)
      - Optional XSA: remove projection of attention output onto value vector
      - Optional sparsemax-on-K: sparsify K along non-RoPE feature dims (so
        positional info in the RoPE-rotated dims is preserved). Applied
        after RoPE, before Q @ K^T.
    Uses torch SDPA instead of flash_attn_3 for portability.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.model_dim % cfg.num_heads == 0
        assert cfg.num_heads % cfg.num_kv_heads == 0
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.model_dim // cfg.num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.rope_dims = cfg.rope_dims
        self.use_xsa = cfg.xsa_enabled
        self.use_spk = cfg.sparsemax_k_enabled
        self.use_spk_mask = cfg.sparsemax_k_mask_enabled
        self.use_dg_k = cfg.dg_k_enabled

        # At most one K-sparsification mode active at a time.
        k_modes_active = sum([self.use_spk, self.use_spk_mask, self.use_dg_k])
        assert k_modes_active <= 1, \
            f"At most one K-sparsification mode can be active; got {k_modes_active}"

        kv_dim = cfg.num_kv_heads * self.head_dim
        self.c_q = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.c_k = nn.Linear(cfg.model_dim, kv_dim, bias=False)
        self.c_v = nn.Linear(cfg.model_dim, kv_dim, bias=False)
        self.proj = nn.Linear(cfg.model_dim, cfg.model_dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((cfg.num_heads,), cfg.qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=cfg.rope_base, rope_dims=cfg.rope_dims)

        # DG-parallel projection matrices. Only instantiated if enabled so
        # the parameter count and optimizer state are identical across other
        # modes. Operates only on the non-RoPE portion of K.
        if self.use_dg_k:
            non_rope_dim = self.head_dim - self.rope_dims
            assert non_rope_dim > 0, \
                "DG-K requires some non-RoPE head dims"
            self.dg_expanded_dim = cfg.dg_expansion_mult * non_rope_dim
            self.dg_k_active = max(
                1, int(round(cfg.dg_sparsity_frac * self.dg_expanded_dim))
            )
            # Per-kv-head expand/compress projections. We use a single shared
            # projection across kv heads to keep parameter count modest; the
            # alternative (per-head) would multiply params by num_kv_heads.
            self.dg_expand = nn.Linear(non_rope_dim, self.dg_expanded_dim, bias=False)
            self.dg_compress = nn.Linear(self.dg_expanded_dim, non_rope_dim, bias=False)
            # Orthogonal init for the two new matrices (fits with the model's
            # default init policy for >=64-dim matrices).
            nn.init.orthogonal_(self.dg_expand.weight, gain=1.0)
            nn.init.orthogonal_(self.dg_compress.weight, gain=1.0)

        # Diagnostics: cosine sim of y with v, measured at two points.
        # _bias_pre:  before any XSA subtraction. Tracks the raw pathology
        #             (attention's output dragged toward self-value) and
        #             also reveals how the model reorganizes when XSA is
        #             active (pre-XSA bias can grow as model "outsources"
        #             the self-echo removal to the subtraction step).
        # _bias_post: after XSA subtraction if enabled, else same as pre.
        #             This is what actually reaches the output projection.
        self._bias_pre = None
        self._bias_post = None

    def _xsa(self, y, v):
        """Remove projection of y onto v (per head, per token).
        For GQA: y has H heads, v has Hkv heads, each v shared across group=H/Hkv
        query heads. Reshape y to (B, T, Hkv, group, D), project each group
        element onto the shared v, subtract. Matches the leader's _xsa_efficient.
        """
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)  # (B, T, Hkv, 1, D)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    @staticmethod
    def _compute_bias(y, v, num_kv_heads, group, head_dim):
        """Mean cos(y_i, v_i) across batch, time, and (query) heads."""
        B, T = y.shape[:2]
        y_g = y.reshape(B, T, num_kv_heads, group, head_dim)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        yn = F.normalize(y_g, dim=-1)
        return (yn * vn).sum(dim=-1).mean()

    def forward(self, x, log_bias=False):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary(q, cos, sin, self.rope_dims)
        k = apply_rotary(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]

        # K-sparsification interventions. At most one active at a time
        # (enforced in __init__). All operate on the non-RoPE portion of K
        # (if any), preserving positional info in the rotated dims. Applied
        # after RoPE, before Q @ K^T.
        has_split = 0 < self.rope_dims < self.head_dim
        if self.use_spk or self.use_spk_mask or self.use_dg_k:
            if has_split:
                k_rope = k[..., :self.rope_dims]
                k_rest = k[..., self.rope_dims:]
            else:
                k_rope = None
                k_rest = k

            if self.use_spk:
                # Strong: simplex projection. Drops sign, L1-normalizes.
                k_rest = sparsemax(k_rest, dim=-1)
            elif self.use_spk_mask:
                # Soft: use sparsemax as support selector, keep original vals.
                k_rest = sparsemax_mask(k_rest, dim=-1)
            elif self.use_dg_k:
                # DG-parallel: expand -> top-k sparsify -> compress.
                # Operates in a higher-dimensional space where orthogonality
                # among sparse patterns is cheap, then projects back.
                expanded = self.dg_expand(k_rest)
                sparse = topk_sparse_ste(expanded, self.dg_k_active, dim=-1)
                k_rest = self.dg_compress(sparse)

            if has_split:
                k = torch.cat([k_rope, k_rest], dim=-1)
            else:
                k = k_rest

        # SDPA wants (B, H, T, D). It supports GQA natively when Hq > Hkv
        # is divisible; we just repeat K and V to match.
        group = self.num_heads // self.num_kv_heads
        k_ex = k.repeat_interleave(group, dim=2)
        v_ex = v.repeat_interleave(group, dim=2)

        q_bhtd = q.transpose(1, 2)
        k_bhtd = k_ex.transpose(1, 2)
        v_bhtd = v_ex.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_bhtd, k_bhtd, v_bhtd, is_causal=True)
        y = y.transpose(1, 2)  # (B, T, H, D)

        # Pre-intervention bias measurement.
        if log_bias:
            with torch.no_grad():
                self._bias_pre = float(self._compute_bias(
                    y, v, self.num_kv_heads, group, self.head_dim
                ).item())

        if self.use_xsa:
            y = self._xsa(y, v)

        # Post-intervention bias measurement. If XSA is off, equals pre.
        if log_bias:
            with torch.no_grad():
                self._bias_post = float(self._compute_bias(
                    y, v, self.num_kv_heads, group, self.head_dim
                ).item())

        y = y.reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(cfg.mlp_mult * cfg.model_dim)
        self.fc = nn.Linear(cfg.model_dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.model_dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        # Squared leaky ReLU, matching the leader.
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = nn.Parameter(torch.ones(cfg.model_dim))
        self.mlp_scale = nn.Parameter(torch.ones(cfg.model_dim))
        # resid_mix: per-feature blend of running x and post-embedding x0.
        # Init = [1s, 0s] -> starts as a vanilla residual, learns to pull in x0.
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(cfg.model_dim), torch.zeros(cfg.model_dim)))
        )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if cfg.ln_scale else 1.0
        self.parallel = layer_idx >= cfg.parallel_residual_start

    def forward(self, x, x0, log_bias=False):
        mix = self.resid_mix.to(x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(
            self.attn_norm(x_in) * self.ln_scale_factor, log_bias=log_bias
        )
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            x_out = (x_in
                     + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
                     + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out)
        else:
            x_out = x_in + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
            mlp_out = self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
            x_out = x_out + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
        return x_out


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.blocks = nn.ModuleList(
            [Block(cfg, i) for i in range(cfg.num_layers)]
        )
        self.final_norm = RMSNorm()
        self.lm_head = (None if cfg.tie_embeddings
                        else nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False))

        # Encoder-decoder split with skip connections.
        self.num_enc = cfg.num_layers // 2
        self.num_dec = cfg.num_layers - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, cfg.model_dim))
        self.skip_gates = (nn.Parameter(torch.zeros(self.num_skip, cfg.model_dim))
                           if cfg.skip_gates_enabled else None)

        self._init_weights()

    def _init_weights(self):
        if self.cfg.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0,
                            std=self.cfg.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2
                      and module.weight.shape[0] >= 64
                      and module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids, log_bias=False):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0, log_bias=log_bias)
            skips.append(x)
        dec_range = range(self.num_enc, self.num_enc + self.num_dec)
        for skip_idx, i in enumerate(dec_range):
            if skip_idx < self.num_skip and skips:
                scaled_skip = (
                    self.skip_weights[skip_idx].to(x.dtype)[None, None, :]
                    * skips.pop()
                )
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(x.dtype))
                    x = torch.lerp(scaled_skip, x, g[None, None, :])
                else:
                    x = x + scaled_skip
            x = self.blocks[i](x, x0, log_bias=log_bias)
        x = self.final_norm(x)
        if self.cfg.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap)

    def forward(self, input_ids, target_ids, log_bias=False):
        logits = self.forward_logits(input_ids, log_bias=log_bias)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1), reduction="mean"
        )

    def per_layer_biases(self):
        """Return list of (pre, post) bias tuples per layer, or (None, None)."""
        return [(blk.attn._bias_pre, blk.attn._bias_post) for blk in self.blocks]


# ==================== Muon optimizer ====================

def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize G via 5-step Newton-Schulz, as in the leader."""
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.to(LOW_P_DTYPE if LOW_P_DTYPE is not None else torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon: Newton-Schulz orthogonalized momentum update for 2D matrices."""
    def __init__(self, params, lr, momentum, backend_steps=5,
                 weight_decay=0.0, nesterov=True, row_normalize=True):
        super().__init__(params, dict(
            lr=lr, momentum=momentum, backend_steps=backend_steps,
            weight_decay=weight_decay, nesterov=nesterov,
            row_normalize=row_normalize,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            row_norm = group["row_normalize"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                if row_norm:
                    row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                    g = g / row_norms.to(g.dtype)
                g = _zeropower_via_newtonschulz5(g, steps=backend_steps)
                g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g.to(p.dtype), alpha=-lr)
        return loss


CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain",
                    "skip_weight", "skip_gates")


def build_optimizers(cfg: Config, model: GPT):
    """Split params: matrix (Muon), scalar (AdamW), embeddings (AdamW)."""
    block_params = list(model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    if model.skip_gates is not None and model.skip_gates.numel() > 0:
        scalar_params.append(model.skip_gates)

    token_lr = cfg.tied_embed_lr if cfg.tie_embeddings else cfg.matrix_lr
    opt_tok = torch.optim.AdamW(
        [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
        weight_decay=cfg.embed_wd, fused=torch.cuda.is_available(),
    )
    opt_muon = Muon(
        matrix_params, lr=cfg.matrix_lr, momentum=cfg.muon_momentum,
        backend_steps=cfg.muon_backend_steps, weight_decay=cfg.muon_wd,
    )
    for g in opt_muon.param_groups:
        g["base_lr"] = cfg.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": cfg.scalar_lr, "base_lr": cfg.scalar_lr}],
        betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps,
        weight_decay=cfg.adam_wd, fused=torch.cuda.is_available(),
    )
    return [opt_tok, opt_muon, opt_scalar]


# ==================== Training ====================

def lr_mul(frac: float, warmdown_frac: float, min_lr: float) -> float:
    if warmdown_frac <= 0:
        return 1.0
    if frac >= 1.0 - warmdown_frac:
        return max((1.0 - frac) / warmdown_frac, min_lr)
    return 1.0


@torch.no_grad()
def evaluate(model, val_data, cfg: Config, device, n_batches: int):
    model.eval()
    batch_size = cfg.train_batch_tokens // cfg.seq_len // cfg.grad_accum_steps
    batch_size = max(batch_size, 1)
    sampler = BatchSampler(val_data, cfg.seq_len, batch_size, seed=12345)
    total_loss, total_tokens = 0.0, 0
    for _ in range(n_batches):
        x, y = sampler.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=LOW_P_DTYPE, enabled=(device.type == "cuda")):
            logits = model.forward_logits(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y.reshape(-1), reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg = total_loss / total_tokens
    return {"loss": avg, "ppl": math.exp(avg)}


MODE_CONFIG = {
    "baseline":     {"xsa": False, "spk": False, "spk_mask": False, "dg_k": False},
    "xsa":          {"xsa": True,  "spk": False, "spk_mask": False, "dg_k": False},
    "spk":          {"xsa": False, "spk": True,  "spk_mask": False, "dg_k": False},
    "xsa_spk":      {"xsa": True,  "spk": True,  "spk_mask": False, "dg_k": False},
    "spk_mask":     {"xsa": False, "spk": False, "spk_mask": True,  "dg_k": False},
    "xsa_spk_mask": {"xsa": True,  "spk": False, "spk_mask": True,  "dg_k": False},
    "dg_k":         {"xsa": False, "spk": False, "spk_mask": False, "dg_k": True},
    "xsa_dg_k":     {"xsa": True,  "spk": False, "spk_mask": False, "dg_k": True},
}
ALL_MODES = list(MODE_CONFIG.keys())


def train_one(cfg: Config, mode: str, seed: int, train_data, val_data, device):
    """Train a single condition with a fixed seed. Mode keys in MODE_CONFIG."""
    assert mode in MODE_CONFIG, f"unknown mode {mode}; must be one of {ALL_MODES}"
    cfg = copy.deepcopy(cfg)
    flags = MODE_CONFIG[mode]
    cfg.xsa_enabled = flags["xsa"]
    cfg.sparsemax_k_enabled = flags["spk"]
    cfg.sparsemax_k_mask_enabled = flags["spk_mask"]
    cfg.dg_k_enabled = flags["dg_k"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GPT(cfg).to(device)
    if device.type == "cuda":
        if LOW_P_DTYPE == torch.bfloat16:
            # Ampere+ path: cast weights to bf16. bf16 has fp32's numerical
            # range, so no GradScaler needed; gradients are bf16 and step
            # directly. Matches the leader's base_model.bfloat16() pattern.
            model = model.to(LOW_P_DTYPE)
            # Control tensors and scalars stay fp32 for training stability.
            for name, p in model.named_parameters():
                if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) \
                        and p.dtype != torch.float32:
                    p.data = p.data.float()
        else:
            # Turing / fp16 path: keep the model in fp32 and use the standard
            # AMP pattern. autocast downcasts activations to fp16 during
            # forward, backward produces fp32 gradients (which GradScaler can
            # unscale), optimizer updates fp32 master weights. Pays ~230MB
            # for fp32 params vs fp16 but necessary for scaler compatibility.
            pass

    n_params = sum(p.numel() for p in model.parameters())
    active_flags = ", ".join(k for k, v in flags.items() if v) or "none"
    print(f"  [{mode}] params: {n_params:,} | active: {active_flags}")

    optimizers = build_optimizers(cfg, model)

    # Batch sampler: seeded identically across modes so both see the same data.
    batch_size = cfg.train_batch_tokens // cfg.seq_len // cfg.grad_accum_steps
    batch_size = max(batch_size, 1)
    sampler = BatchSampler(train_data, cfg.seq_len, batch_size, seed=seed + 777)

    history = []
    model.train()
    t0 = time.time()
    ema_loss = None

    # GradScaler only needed for fp16 (underflow risk). bf16 has fp32 range.
    use_scaler = (device.type == "cuda" and LOW_P_DTYPE == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    for step in range(1, cfg.iters + 1):
        # Zero grads
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Micro-batch grad accumulation
        total_loss = 0.0
        log_bias_this_step = (cfg.log_attn_bias
                              and (step % cfg.val_every == 0
                                   or step == 1
                                   or step == cfg.iters))
        for mb in range(cfg.grad_accum_steps):
            x, y = sampler.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=LOW_P_DTYPE, enabled=(device.type == "cuda")
            ):
                loss = model(x, y, log_bias=(log_bias_this_step and mb == 0))
            scaled = loss / cfg.grad_accum_steps
            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += float(loss.detach())
        total_loss /= cfg.grad_accum_steps
        ema_loss = total_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * total_loss

        # LR schedule + Muon momentum warmup
        frac = step / cfg.iters
        scale = lr_mul(frac, cfg.warmdown_frac, cfg.min_lr)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", g["lr"]) * scale
        mm_frac = (min(step / cfg.muon_momentum_warmup_steps, 1.0)
                   if cfg.muon_momentum_warmup_steps > 0 else 1.0)
        muon_mom = ((1 - mm_frac) * cfg.muon_momentum_warmup_start
                    + mm_frac * cfg.muon_momentum)
        for opt in optimizers:
            if isinstance(opt, Muon):
                for g in opt.param_groups:
                    g["momentum"] = muon_mom

        # Unscale before clipping/stepping (fp16 path only).
        if use_scaler:
            for opt in optimizers:
                scaler.unscale_(opt)

        # Grad clip
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        # Step. GradScaler's .step() skips update if grads contain inf/nan and
        # adjusts the scale; we call it on Adam optimizers. Muon is a custom
        # optimizer and doesn't integrate with GradScaler, so we step it
        # directly (grads have already been unscaled above).
        if use_scaler:
            for opt in optimizers:
                if isinstance(opt, Muon):
                    opt.step()
                else:
                    scaler.step(opt)
            scaler.update()
        else:
            for opt in optimizers:
                opt.step()

        # Log + eval
        do_log = (step % cfg.val_every == 0 or step == 1 or step == cfg.iters)
        if do_log:
            metrics = evaluate(model, val_data, cfg, device, cfg.val_batches)
            mean_bias_pre = None
            mean_bias_post = None
            if log_bias_this_step:
                pairs = model.per_layer_biases()
                pres = [p for (p, _) in pairs if p is not None]
                posts = [q for (_, q) in pairs if q is not None]
                if pres:
                    mean_bias_pre = float(np.mean(pres))
                if posts:
                    mean_bias_post = float(np.mean(posts))
            elapsed = time.time() - t0
            rec = {
                "step": step,
                "train_ema": ema_loss,
                "val_loss": metrics["loss"],
                "val_ppl": metrics["ppl"],
                "bias_pre": mean_bias_pre,
                "bias_post": mean_bias_post,
                "elapsed_s": elapsed,
            }
            history.append(rec)
            pre_s = f"{mean_bias_pre:.3f}" if mean_bias_pre is not None else "--"
            post_s = f"{mean_bias_post:.3f}" if mean_bias_post is not None else "--"
            print(f"  [{mode:>18}] step {step:5d}/{cfg.iters} "
                  f"| train {ema_loss:.4f} | val {metrics['loss']:.4f} "
                  f"| ppl {metrics['ppl']:7.2f} "
                  f"| bias pre/post {pre_s}/{post_s} "
                  f"| {elapsed:5.1f}s")
    return history


# ==================== Main ====================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--modes", nargs="+", default=ALL_MODES,
                   choices=ALL_MODES,
                   help="Which conditions to run. Default: all 4.")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_tokens", type=int, default=32768)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--out", default="xsa_ab_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    global LOW_P_DTYPE
    LOW_P_DTYPE = low_precision_dtype()
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute capability: {cap[0]}.{cap[1]}")
        print(f"  Low-precision dtype: {LOW_P_DTYPE} "
              f"({'bf16 native' if LOW_P_DTYPE == torch.bfloat16 else 'fp16 (Turing/older)'})")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    cfg = Config(
        iters=args.iters,
        seq_len=args.seq_len,
        train_batch_tokens=args.batch_tokens,
        grad_accum_steps=args.grad_accum,
        val_every=args.val_every,
    )
    print(f"Config: iters={cfg.iters} seq_len={cfg.seq_len} "
          f"batch_tok={cfg.train_batch_tokens} grad_accum={cfg.grad_accum_steps}")
    print(f"  layers={cfg.num_layers} dim={cfg.model_dim} "
          f"heads={cfg.num_heads} kv={cfg.num_kv_heads}")
    print(f"  seeds={args.seeds} modes={args.modes}")

    print("\n--- Loading data ---")
    train_data, val_data = get_data(cfg.seq_len)
    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    all_results = {}
    for seed in args.seeds:
        for mode in args.modes:
            key = f"seed{seed}_{mode}"
            print(f"\n{'='*70}")
            print(f"Run: {key}")
            print(f"{'='*70}")
            hist = train_one(cfg, mode, seed, train_data, val_data, device)
            all_results[key] = hist
            # Free GPU memory between runs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = (f"{'seed':>6} {'mode':>10} {'train':>10} {'val':>10} "
              f"{'ppl':>10} {'bias_pre':>10} {'bias_post':>10}")
    print(header)
    print("-" * len(header))
    finals = {}
    for key, hist in all_results.items():
        # Parse "seed{N}_{mode}" where mode can contain underscores.
        rest = key.removeprefix("seed")
        seed_str, mode = rest.split("_", 1)
        seed = int(seed_str)
        final = hist[-1]
        pre = final.get("bias_pre")
        post = final.get("bias_post")
        pre_s = f"{pre:.3f}" if pre is not None else "--"
        post_s = f"{post:.3f}" if post is not None else "--"
        print(f"{seed:>6} {mode:>10} {final['train_ema']:>10.4f} "
              f"{final['val_loss']:>10.4f} {final['val_ppl']:>10.2f} "
              f"{pre_s:>10} {post_s:>10}")
        finals[(seed, mode)] = final

    # Paired deltas vs baseline for each mode.
    def _paired_stats(ref_mode, target_mode):
        """Return list of (seed, d_val, d_bias_post) tuples."""
        rows = []
        for seed in args.seeds:
            if (seed, ref_mode) not in finals or (seed, target_mode) not in finals:
                continue
            b = finals[(seed, ref_mode)]
            t = finals[(seed, target_mode)]
            dv = t["val_loss"] - b["val_loss"]
            db = None
            if (b.get("bias_post") is not None
                    and t.get("bias_post") is not None):
                db = t["bias_post"] - b["bias_post"]
            rows.append((seed, dv, db))
        return rows

    def _report_pair(ref_mode, target_mode):
        rows = _paired_stats(ref_mode, target_mode)
        if not rows:
            return None
        print(f"\nPaired deltas ({target_mode} - {ref_mode}):")
        print(f"{'seed':>6} {'d_val':>10} {'d_bias_post':>14}")
        dvs = []
        for seed, dv, db in rows:
            db_s = f"{db:+.3f}" if db is not None else "--"
            print(f"{seed:>6} {dv:>+10.4f} {db_s:>14}")
            dvs.append(dv)
        m = float(np.mean(dvs))
        if len(dvs) > 1:
            s = float(np.std(dvs, ddof=1))
            ratio = abs(m) / s if s > 0 else float("inf")
            print(f"  mean d_val: {m:+.4f}  |  std: {s:.4f}  |  |mean|/std: {ratio:.2f}")
        else:
            print(f"  mean d_val: {m:+.4f}  (single seed; no variance estimate)")
        return m

    # Primary comparisons: each intervention vs baseline, and the combo vs XSA
    # (because XSA is the actual reference the combo has to beat).
    d_xsa = d_spk = d_both_vs_base = d_both_vs_xsa = None
    if "baseline" in args.modes and "xsa" in args.modes:
        d_xsa = _report_pair("baseline", "xsa")
    if "baseline" in args.modes and "spk" in args.modes:
        d_spk = _report_pair("baseline", "spk")
    if "baseline" in args.modes and "xsa_spk" in args.modes:
        d_both_vs_base = _report_pair("baseline", "xsa_spk")
    if "xsa" in args.modes and "xsa_spk" in args.modes:
        d_both_vs_xsa = _report_pair("xsa", "xsa_spk")

    # Super-additivity analysis. The question: does SPK add value *on top of*
    # XSA? Two ways to look at it:
    #   1) Simple additivity: is d(xsa_spk vs base) better than d(xsa) + d(spk)?
    #   2) Direct contribution: is d(xsa_spk vs xsa) negative (SPK helps over
    #      the XSA baseline)? This is the practically relevant question if
    #      you're deciding whether to add SPK to an architecture that already
    #      has XSA.
    if d_xsa is not None and d_spk is not None and d_both_vs_base is not None:
        predicted_additive = d_xsa + d_spk
        excess = d_both_vs_base - predicted_additive  # negative = super-additive
        print("\nAdditivity test (val_loss improvements, negative is better):")
        print(f"  xsa vs baseline:      d_val = {d_xsa:+.4f}")
        print(f"  spk vs baseline:      d_val = {d_spk:+.4f}")
        print(f"  sum (predicted):              {predicted_additive:+.4f}")
        print(f"  xsa_spk vs baseline:          {d_both_vs_base:+.4f}")
        print(f"  excess over additive:         {excess:+.4f}  "
              f"({'super-additive' if excess < 0 else 'sub-additive'})")
    if d_both_vs_xsa is not None:
        print(f"\nDirect SPK contribution (xsa_spk vs xsa): d_val = {d_both_vs_xsa:+.4f}")
        print("  Negative means SPK adds value on top of XSA alone.")
        print("  This is the practically relevant number for the competition.")

    # Save
    with open(args.out, "w") as fh:
        json.dump({
            "config": asdict(cfg),
            "seeds": args.seeds,
            "modes": args.modes,
            "runs": all_results,
        }, fh, indent=2)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()
