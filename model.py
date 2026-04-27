"""
Small Transformer LM using UnifiedHybridAttention.

Phrase boundaries are computed inside the model via a phrase_builder module
passed at construction time. forward(input_ids, labels) takes already-shifted
labels (caller responsibility), matching the (x, y) pattern from the shard
loader. Pass labels=None for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_csa_unified import (
    UnifiedAttentionConfig,
    UnifiedHybridAttention,
    RMSNorm,
    count_attention_params,
)


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 4
    ffn_mult: int = 4
    tie_embeddings: bool = True

    n_query_heads: int = 4
    head_dim: int = 64
    query_compress_dim: int = 128
    n_indexer_heads: int = 2
    indexer_head_dim: int = 32
    top_k: int = 16
    n_csa: int = 256
    max_phrase_len: int = 16
    n_win: int = 64
    sw_mode: str = "fixed"            # "fixed" or "phrase"
    use_attn_sink: bool = True

    dropout: float = 0.0

    def to_attn_config(self) -> UnifiedAttentionConfig:
        return UnifiedAttentionConfig(
            d_model=self.d_model,
            n_query_heads=self.n_query_heads,
            head_dim=self.head_dim,
            query_compress_dim=self.query_compress_dim,
            n_indexer_heads=self.n_indexer_heads,
            indexer_head_dim=self.indexer_head_dim,
            top_k=self.top_k,
            n_csa=self.n_csa,
            max_phrase_len=self.max_phrase_len,
            n_win=self.n_win,
            sw_mode=self.sw_mode,
            use_attn_sink=self.use_attn_sink,
            dropout=self.dropout,
        )


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * ffn_mult
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = UnifiedHybridAttention(cfg.to_attn_config())
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.ffn_mult, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, phrase_mask, phrase_token_idx, phrase_end_pos, phrase_id):
        a = self.attn(self.norm1(x), phrase_mask, phrase_token_idx,
                      phrase_end_pos, phrase_id)
        x = x + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class HybridTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, phrase_builder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.phrase_builder = phrase_builder
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        input_ids: (B, T)
        labels:    (B, T) optional, already shifted by the caller (labels[t]
                   is the target prediction for input_ids[t]). If provided,
                   returns (logits, loss); otherwise (logits, None).
        """
        phrase_mask, phrase_token_idx, phrase_end_pos, phrase_id = self.phrase_builder(input_ids)

        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, phrase_mask, phrase_token_idx, phrase_end_pos, phrase_id)
        x = self.final_norm(x)

        if self.lm_head is None:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, exclude_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.embed.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n


# -------------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    from online_phrase_builder import FixedStridePhraseBuilder

    cfg = ModelConfig(
        vocab_size=1024, d_model=128, n_layers=2,
        n_query_heads=4, head_dim=32, query_compress_dim=64,
        n_indexer_heads=2, indexer_head_dim=32,
        top_k=8, n_csa=64, max_phrase_len=8, n_win=16,
    )
    builder = FixedStridePhraseBuilder(
        max_phrase_len=cfg.max_phrase_len, pad_token_id=-1, stride=4,
    )
    model = HybridTransformerLM(cfg, builder)

    print(f"non-embedding params: {model.num_parameters(True):,}")
    print(f"total params:         {model.num_parameters(False):,}")
    print(f"\nattention params per layer:")
    for k, v in count_attention_params(cfg.to_attn_config()).items():
        if k == "TOTAL":
            print(f"  {k:<28s} {v:>10,d}")

    B, T = 2, 64
    chunk = torch.randint(0, cfg.vocab_size, (B, T + 1))
    x = chunk[:, :-1]
    y = chunk[:, 1:]
    logits, loss = model(x, labels=y)
    print(f"\nlogits shape: {tuple(logits.shape)}")
    print(f"loss:         {loss.item():.4f}")
    loss.backward()
    print("backward pass ok")
