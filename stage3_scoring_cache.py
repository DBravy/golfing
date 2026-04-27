"""
Stage 2: Score-based KV cache eviction.

Extends Stage 1 with a per-token importance score derived from attention
weights received during each forward pass. Three eviction modes run in
parallel so we can compare:

  1. fifo           (Stage 1 baseline: drop oldest)
  2. attn           (drop lowest accumulated attention, with decay)
  3. attn+recency   (as above, plus a recency bonus so new keys aren't
                     evicted before they get a chance to be useful)

Transformers 5.x removed `output_attentions` from the top-level forward,
but GPT2Attention.forward still returns (attn_output, attn_weights)
when attn_implementation='eager'. We capture the weights with forward
hooks; everything downstream is version-independent.

Run:
    pip install 'torch' 'transformers>=5.0'
    python stage2_scoring_cache.py
"""

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.cache_utils import DynamicCache
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


# ---------------------------------------------------------------------------
# Attention collector
# ---------------------------------------------------------------------------

class AttentionCollector:
    """Captures attn_weights from each GPT2Attention layer via forward hooks.

    Hook output is the tuple (attn_output, attn_weights) returned by
    GPT2Attention.forward. We detach and stash the weights. The top-level
    model no longer surfaces them, so hooks are the only way in tfm 5.x.
    """

    def __init__(self, model):
        self.attentions = []
        self.hooks = []
        for module in model.modules():
            if isinstance(module, GPT2Attention):
                self.hooks.append(module.register_forward_hook(self._hook))
        if not self.hooks:
            raise RuntimeError("No GPT2Attention modules found.")

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.attentions.append(output[1].detach())

    def reset(self):
        self.attentions = []

    def get_and_reset(self):
        a = self.attentions
        self.attentions = []
        return a

    def remove(self):
        for h in self.hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

class FIFOKVCache:
    """Baseline with optional attention sinks.

    When n_sinks > 0, the first n_sinks cache slots are always preserved
    (classic attention sinks per Xiao et al. 2023). The remaining
    (budget - n_sinks) slots are filled with the most recent tokens.
    """

    def __init__(self, budget, n_sinks=4):
        assert 0 <= n_sinks < budget
        self.budget = budget
        self.n_sinks = n_sinks
        self.cache = DynamicCache()
        self.token_ids = None  # [L] LongTensor; original token id for each cache slot

    @property
    def length(self):
        if not self.cache.layers:
            return 0
        return self.cache.layers[0].get_seq_length()

    def update(self, new_cache, new_token_ids, attentions=None):
        self.cache = new_cache
        # Track which token each cache slot came from
        new_ids = new_token_ids.detach().view(-1)
        if self.token_ids is None:
            self.token_ids = new_ids.clone()
        else:
            self.token_ids = torch.cat([self.token_ids, new_ids])
        # Evict if over budget: keep first n_sinks + last (budget - n_sinks)
        if self.length > self.budget:
            L = self.length
            K = self.n_sinks
            tail = self.budget - K
            device = self.cache.layers[0].keys.device
            if K == 0:
                for layer in self.cache.layers:
                    layer.keys = layer.keys[..., -tail:, :].contiguous()
                    layer.values = layer.values[..., -tail:, :].contiguous()
                self.token_ids = self.token_ids[-tail:]
            else:
                keep = torch.cat([
                    torch.arange(0, K, device=device),
                    torch.arange(L - tail, L, device=device),
                ])
                for layer in self.cache.layers:
                    layer.keys = layer.keys.index_select(-2, keep).contiguous()
                    layer.values = layer.values.index_select(-2, keep).contiguous()
                self.token_ids = self.token_ids[keep]


class ScoringKVCache:
    """Score-based eviction using accumulated attention (with optional recency).

    Per cached token we track:
      - attn_score: exponentially-decayed sum of attention received across
                    all queries, heads, and layers. Normalized per forward
                    pass so values are comparable across chunk sizes.
      - birth_step: the forward-pass count when the token was added.

    Eviction scores:
      mode='attention':          attn_score
      mode='attention_recency':  attn_score + alpha * exp(-age / tau)

    where age = current_step - birth_step. Recency is a small additive
    bonus that fades over ~tau forward passes.

    Note on the order of scoring vs. eviction: when a chunk is processed,
    the model first computes attention against the full (cached + new)
    keys and returns weights of shape [B, H, Q, K_total]. We roll that
    into the score for *every* key seen this step (including the new
    ones). Only then do we evict. This means fresh keys do get some
    self-attention credit before they can be removed.
    """

    def __init__(self, budget, mode="attention_recency",
                 decay=0.99, recency_alpha=1.0, recency_tau=None,
                 pin_last=32, n_sinks=4):
        assert mode in {"attention", "attention_recency"}
        assert 0 <= pin_last < budget, f"pin_last ({pin_last}) must be in [0, budget={budget})"
        assert 0 <= n_sinks, f"n_sinks must be >= 0"
        assert n_sinks + pin_last < budget, \
            f"n_sinks ({n_sinks}) + pin_last ({pin_last}) must be < budget ({budget})"
        self.budget = budget
        self.mode = mode
        self.decay = decay
        self.recency_alpha = recency_alpha
        # Default tau: ~a quarter of the budget, so recency fades well
        # before a fresh key becomes a candidate for eviction purely on age.
        self.recency_tau = recency_tau if recency_tau is not None else max(1, budget // 4)
        # Tail pin: last K tokens protected unconditionally. See the Stage 2
        # diagnostic run for why this matters (chunk-tail tokens have score ~0).
        self.pin_last = pin_last
        # Attention sinks: first K tokens protected unconditionally. Classic
        # StreamingLLM mechanism. Pairs with cache-relative positions in
        # generate() to keep the model in its position-embedding distribution.
        self.n_sinks = n_sinks

        self.cache = DynamicCache()
        self.attn_score = None   # [L] float
        self.birth_step = None   # [L] long
        self.token_ids = None    # [L] long; original token id for each slot
        self.step = 0

    @property
    def length(self):
        if not self.cache.layers:
            return 0
        return self.cache.layers[0].get_seq_length()

    def update(self, new_cache, new_token_ids, attentions):
        self.cache = new_cache
        L = self.length
        self.step += 1

        # Track token ids regardless of whether we have attention this step
        new_ids = new_token_ids.detach().view(-1)
        if self.token_ids is None:
            self.token_ids = new_ids.clone()
        else:
            self.token_ids = torch.cat([self.token_ids, new_ids])

        if L == 0 or not attentions:
            return

        device = self.cache.layers[0].keys.device
        prev_L = self.attn_score.shape[0] if self.attn_score is not None else 0
        new_tokens = L - prev_L

        # Per-key attention received this step.
        # attentions[i] shape: [batch, n_heads, q_len, kv_len], kv_len == L.
        n_layers = len(attentions)
        q_len = attentions[0].shape[-2]
        per_key = torch.zeros(L, device=device)
        for layer_attn in attentions:
            per_key = per_key + layer_attn.sum(dim=(0, 1, 2))
        # Normalize: per-query attention over keys sums to 1 (softmax), so
        # dividing by (n_layers * q_len) gives us "fraction of attention mass
        # received by this key, averaged over layers and queries", times n_heads.
        per_key = per_key / (n_layers * q_len)

        # Grow bookkeeping tensors to match the cache, applying decay.
        if self.attn_score is None:
            self.attn_score = torch.zeros(L, device=device)
            self.birth_step = torch.full((L,), self.step, device=device, dtype=torch.long)
        else:
            self.attn_score = torch.cat([
                self.attn_score * self.decay,
                torch.zeros(new_tokens, device=device),
            ])
            self.birth_step = torch.cat([
                self.birth_step,
                torch.full((new_tokens,), self.step, device=device, dtype=torch.long),
            ])

        self.attn_score = self.attn_score + per_key

        if L > self.budget:
            self._evict()

    def _evict(self):
        scores = self.attn_score.clone()
        if self.mode == "attention_recency":
            age = (self.step - self.birth_step).float()
            recency = torch.exp(-age / self.recency_tau)
            scores = scores + self.recency_alpha * recency

        L = scores.shape[0]
        S = self.n_sinks
        T = self.pin_last
        device = scores.device

        # Three-way partition of indices:
        #   [0, S)        -> sinks, always kept
        #   [S, L - T)    -> middle, keep top (budget - S - T) by score
        #   [L - T, L)    -> tail, always kept
        sinks = torch.arange(0, S, device=device) if S > 0 else torch.empty(0, dtype=torch.long, device=device)
        tail = torch.arange(L - T, L, device=device) if T > 0 else torch.empty(0, dtype=torch.long, device=device)

        n_from_mid = self.budget - S - T
        if n_from_mid > 0:
            mid_scores = scores[S: L - T] if T > 0 else scores[S:]
            mid_local = torch.topk(mid_scores, n_from_mid).indices
            from_mid = mid_local + S  # shift to global indices
            keep = torch.cat([sinks, from_mid, tail])
        else:
            keep = torch.cat([sinks, tail])

        # Sort so the surviving cache stays in original sequence order. Not
        # required for attention correctness, but makes the diagnostic
        # output readable.
        keep, _ = keep.sort()

        for layer in self.cache.layers:
            layer.keys = layer.keys.index_select(-2, keep).contiguous()
            layer.values = layer.values.index_select(-2, keep).contiguous()
        self.attn_score = self.attn_score[keep]
        self.birth_step = self.birth_step[keep]
        self.token_ids = self.token_ids[keep]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, cache, collector,
             chunk_size=256, device="cpu", on_prefill_complete=None):
    """Prefill + decode with cache-relative position IDs.

    Positions for new tokens are computed as arange(cache.length, cache.length + N)
    rather than using the token's true index in the overall stream. Combined
    with attention sinks in the cache, this keeps the model's position IDs
    within its trained 1024-slot range no matter how long the input is.

    Requires: cache.budget + chunk_size <= model.config.n_positions.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    n_positions = model.config.n_positions
    assert cache.budget + chunk_size <= n_positions, (
        f"budget ({cache.budget}) + chunk_size ({chunk_size}) exceeds "
        f"n_positions ({n_positions}); reduce one of them."
    )

    last_logits = None

    # Prefill
    for start in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, start:start + chunk_size]
        chunk_len = chunk.shape[1]
        # Cache-relative positions. Sinks keep their baked pos_emb[0..S-1]
        # from when they were first cached; new tokens get positions starting
        # at the current cache length, which is bounded by budget.
        base = cache.length
        positions = torch.arange(base, base + chunk_len, device=device).unsqueeze(0)

        collector.reset()
        out = model(
            input_ids=chunk,
            past_key_values=cache.cache,
            position_ids=positions,
            use_cache=True,
        )
        attentions = collector.get_and_reset()
        cache.update(out.past_key_values, new_token_ids=chunk, attentions=attentions)
        last_logits = out.logits[:, -1, :]

    # Diagnostic hook: fires after prefill is fully processed and evicted,
    # but before any tokens have been generated. This is the state the model
    # will use to predict the answer.
    if on_prefill_complete is not None:
        on_prefill_complete(cache)

    # Decode
    generated = []
    for _ in range(max_new_tokens):
        next_tok = last_logits.argmax(dim=-1, keepdim=True)
        tok_id = next_tok.item()
        generated.append(tok_id)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break
        pos = torch.tensor([[cache.length]], device=device)

        collector.reset()
        out = model(
            input_ids=next_tok,
            past_key_values=cache.cache,
            position_ids=pos,
            use_cache=True,
        )
        attentions = collector.get_and_reset()
        cache.update(out.past_key_values, new_token_ids=next_tok, attentions=attentions)
        last_logits = out.logits[:, -1, :]

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def inspect_cache(cache, tokenizer, label=""):
    """Print what survived in the cache. Called after prefill, before decode.

    Shows:
      - cache fill ratio
      - whether the needle ("73924") is still present
      - a condensed decode of the surviving tokens
      - for scoring caches, the top-5 highest-scored tokens with their ids
    """
    if cache.token_ids is None or len(cache.token_ids) == 0:
        print(f"  [{label}] cache empty")
        return

    ids = cache.token_ids.tolist()
    L = len(ids)
    decoded = tokenizer.decode(ids)
    needle_here = "73924" in decoded

    # Condense long content for display: show head and tail
    if len(decoded) < 240:
        shown = decoded
    else:
        shown = decoded[:120] + "  [...]  " + decoded[-120:]
    shown = shown.replace("\n", " ")

    print(f"  [{label}] L={L}/{cache.budget}  needle_in_cache={needle_here}")
    print(f"    content: {shown!r}")

    # For scoring caches, surface the top-scored tokens
    if hasattr(cache, "attn_score") and cache.attn_score is not None:
        scores = cache.attn_score.clone()
        if getattr(cache, "mode", "") == "attention_recency":
            age = (cache.step - cache.birth_step).float()
            scores = scores + cache.recency_alpha * torch.exp(-age / cache.recency_tau)
        k = min(5, L)
        vals, idxs = torch.topk(scores, k)
        pieces = []
        for s, i in zip(vals.tolist(), idxs.tolist()):
            tok_text = tokenizer.decode([ids[i]]).replace("\n", " ")
            pieces.append(f"(idx={i}, score={s:.2f}, tok={tok_text!r})")
        print(f"    top-{k}: " + " ".join(pieces))


def build_needle_prompt(tokenizer, needle, question, filler, distance_tokens):
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    while len(filler_ids) < distance_tokens:
        filler_ids = filler_ids + filler_ids
    filler_ids = filler_ids[:distance_tokens]

    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids

    full_ids = needle_ids + filler_ids + question_ids
    return tokenizer.decode(full_ids), len(filler_ids)


def make_cache(kind, budget):
    if kind == "fifo":
        return FIFOKVCache(budget)
    if kind == "attn":
        return ScoringKVCache(budget, mode="attention")
    if kind == "attn+recency":
        return ScoringKVCache(budget, mode="attention_recency")
    raise ValueError(kind)


def run_one(model, tokenizer, collector, kind, budget, distance, device, verbose=False):
    needle = "The secret access code is 73924."
    question = (" Question: What is the secret access code?"
                " Answer: The secret access code is")
    filler = ("The weather today is pleasant. Birds are singing in the trees. "
              "The market opened flat this morning. A light breeze moves through the park. ")
    prompt, actual_d = build_needle_prompt(tokenizer, needle, question, filler, distance)
    cache = make_cache(kind, budget)

    cb = None
    if verbose:
        pin = getattr(cache, "pin_last", 0)
        sinks = getattr(cache, "n_sinks", 0)
        label = f"{kind} budget={budget} sinks={sinks} pin={pin} d={actual_d}"
        cb = lambda c: inspect_cache(c, tokenizer, label=label)

    out = generate(model, tokenizer, prompt, max_new_tokens=10,
                   cache=cache, collector=collector, device=device,
                   on_prefill_complete=cb)
    return actual_d, out, "73924" in out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  transformers={transformers.__version__}")

    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained(
        "distilgpt2",
        attn_implementation="eager",  # required so attn_weights aren't None
    ).to(device).eval()

    collector = AttentionCollector(model)

    distances = [50, 200, 500, 900, 1500, 3000, 6000]
    budgets = [256, 512]
    kinds = ["fifo", "attn", "attn+recency"]

    all_results = {k: {b: {} for b in budgets} for k in kinds}

    for kind in kinds:
        for budget in budgets:
            print(f"\n=== kind={kind}  budget={budget} ===")
            for d in distances:
                actual_d, out, hit = run_one(
                    model, tokenizer, collector, kind, budget, d, device,
                    verbose=True,
                )
                all_results[kind][budget][actual_d] = {"hit": hit, "output": out.strip()}
                print(f"  distance={actual_d:>5}  hit={hit}  output={out.strip()!r}")

    collector.remove()

    # Summary grid
    print("\nhit grid:")
    header = f"{'kind':>14} {'budget':>8}  " + "  ".join(f"{d:>5}" for d in distances)
    print(header)
    for kind in kinds:
        for budget in budgets:
            row = f"{kind:>14} {budget:>8}  " + "  ".join(
                f"{'  hit' if all_results[kind][budget].get(d, {}).get('hit') else ' miss':>5}"
                for d in distances
            )
            print(row)


if __name__ == "__main__":
    main()
