"""
Faithful implementation of vanilla TTT-E2E for small-scale experiments.

Reference:
    Tandon et al., "End-to-End Test-Time Training for Long Context" (2025)
    https://arxiv.org/abs/2512.23675

Key design choices (matching the paper where feasible):
- Sliding-window attention with a configurable window.
- The last n_ttt_blocks are TTT blocks. Each TTT block has two MLPs:
  a "fast" MLP that is updated during the TTT inner loop, and a "slow"
  MLP that stays static. Both MLPs use half the hidden dim of a regular
  MLP so that a TTT model and a standard model have the same total
  parameter count.
- The two MLPs in a TTT block are combined additively in parallel:
    x = x + attn(norm1(x))
    h = norm2(x)
    x = x + mlp_fast(h) + mlp_slow(h)
  The paper is not explicit about series vs. parallel; parallel keeps
  the block structure closest to a regular block.
- Only the fast MLPs are updated in the inner loop. Attention, norms,
  embeddings, and the slow MLP are frozen during TTT (per the paper).
- Inner loop: mini-batch SGD on next-token prediction loss (Eq. 5).
- Outer loop: gradient of gradients via torch.autograd.grad(create_graph=True),
  optimizing the post-TTT loss (Eq. 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if self.window_size is not None and self.window_size < T:
            window = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(-(self.window_size - 1))
            mask = causal & window
        else:
            mask = causal

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class RegularBlock(nn.Module):
    is_ttt = False

    def __init__(self, d_model, ff_hidden, n_heads, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_hidden)

    def forward(self, x, fast_params=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TTTBlock(nn.Module):
    """
    Fast MLP (TTT'd) and slow MLP (static), combined in parallel.
    Each MLP has half the hidden dim of a regular MLP, so a TTT block
    has the same MLP parameter count as a regular block.
    """
    is_ttt = True

    def __init__(self, d_model, ff_hidden, n_heads, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, window_size)
        self.norm2 = nn.LayerNorm(d_model)
        half = ff_hidden // 2
        self.mlp_fast = MLP(d_model, half)
        self.mlp_slow = MLP(d_model, half)

    def forward(self, x, fast_params=None):
        x = x + self.attn(self.norm1(x))
        h = self.norm2(x)
        if fast_params is None:
            h_fast = self.mlp_fast(h)
        else:
            h_fast = functional_call(self.mlp_fast, fast_params, h)
        h_slow = self.mlp_slow(h)
        x = x + h_fast + h_slow
        return x


class TTTTransformer(nn.Module):
    """
    If n_ttt_blocks == 0, this is a plain transformer with sliding-window
    attention. If n_ttt_blocks > 0, the last n_ttt_blocks are TTT blocks.
    The total parameter count is identical in either case (for the same
    n_layers, d_model, ff_mult), because TTT blocks use two half-sized MLPs.
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=8,
        n_heads=4,
        ff_mult=4,
        max_seq_len=512,
        window_size=256,
        n_ttt_blocks=2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_ttt_blocks = n_ttt_blocks

        ff_hidden = d_model * ff_mult

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        blocks = []
        first_ttt = n_layers - n_ttt_blocks
        for i in range(n_layers):
            cls = TTTBlock if i >= first_ttt else RegularBlock
            blocks.append(cls(d_model, ff_hidden, n_heads, window_size))
        self.blocks = nn.ModuleList(blocks)

        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie head to embedding
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2-style initialization
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_initial_fast_params(self):
        """List of {name: Parameter} dicts, one per TTT block."""
        return [
            {name: p for name, p in block.mlp_fast.named_parameters()}
            for block in self.blocks if block.is_ttt
        ]

    def forward(self, x, fast_params_list=None):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(pos)

        fp_idx = 0
        for block in self.blocks:
            if block.is_ttt and fast_params_list is not None:
                h = block(h, fast_params=fast_params_list[fp_idx])
                fp_idx += 1
            else:
                h = block(h)

        h = self.norm_final(h)
        return self.head(h)

    def num_params(self):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


@torch.enable_grad()
def ttt_forward_and_loss(model, tokens, mini_batch_size, inner_lr, create_graph=True):
    """
    Paper Eqs. 5 and 6:
        W_i = W_{i-1} - eta/b * sum_t grad(ell_t(W_{i-1}))     (inner)
        L(W_0; X) = (1/T) * sum_i sum_t ell_t(W_{i-1})         (outer)

    Expects tokens of shape [B, T]. Returns the mean per-token NLL.

    When create_graph=True (training), the returned loss is differentiable
    w.r.t. model parameters through the inner-loop updates. Use
    create_graph=False during evaluation to save memory and time.

    Note on batching: for B>1 the fast params are shared across the batch
    (simpler than vmap). For strict per-sequence TTT use B=1.
    """
    B, T = tokens.shape
    V = model.vocab_size

    n_mini = (T - 1) // mini_batch_size
    assert n_mini > 0, f"Need at least one mini-batch (T-1={T-1}, b={mini_batch_size})"

    fast_params_list = model.get_initial_fast_params()

    total_loss = 0.0
    total_tokens = 0

    for i in range(n_mini):
        pred_start = i * mini_batch_size + 1
        pred_end = (i + 1) * mini_batch_size + 1
        context_end = pred_end

        logits = model(tokens[:, :context_end], fast_params_list=fast_params_list)
        batch_logits = logits[:, pred_start - 1:pred_end - 1]
        batch_targets = tokens[:, pred_start:pred_end]

        sum_loss = F.cross_entropy(
            batch_logits.reshape(-1, V),
            batch_targets.reshape(-1),
            reduction="sum",
        )
        total_loss = total_loss + sum_loss
        total_tokens += B * mini_batch_size

        # Skip the final inner update (it would not affect the outer loss)
        if i < n_mini - 1:
            all_fast = [p for fp in fast_params_list for p in fp.values()]
            grads = torch.autograd.grad(sum_loss, all_fast, create_graph=create_graph)
            scale = inner_lr / (B * mini_batch_size)
            idx = 0
            new_list = []
            for fp in fast_params_list:
                new_fp = {}
                for name, p in fp.items():
                    new_fp[name] = p - scale * grads[idx]
                    idx += 1
                new_list.append(new_fp)
            fast_params_list = new_list

    return total_loss / total_tokens


def standard_forward_and_loss(model, tokens):
    """Standard next-token cross-entropy over the full sequence."""
    V = model.vocab_size
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
