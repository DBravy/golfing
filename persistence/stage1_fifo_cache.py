"""
Stage 1: Fixed-budget FIFO KV cache for distilgpt2.

Requires transformers >= 5.0, which removed the legacy tuple past_key_values
format. We now work with DynamicCache directly: each layer exposes .keys
and .values tensors of shape [batch, n_heads, seq_len, head_dim], and we
just slice those in place to evict.

The cache keeps at most `budget` tokens; when full, the oldest are dropped.
Position IDs are clamped at n_positions - 1 (1023 for GPT-2). This is a
known weakness we fix in a later stage; for now it means we can process
sequences of any length without crashing, at the cost of quality past
position 1024.

Run:
    pip install 'torch' 'transformers>=5.0'
    python stage1_fifo_cache.py
"""

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class FIFOKVCache:
    """Fixed-budget KV cache with first-in-first-out eviction.

    Wraps a DynamicCache. The model mutates the underlying cache in place
    during each forward pass; after the pass we check the length and evict
    from the front if needed by slicing each layer's key/value tensors.

    Future stages will keep the same interface (length / cache / update)
    but change _evict to score-based removal.
    """

    def __init__(self, budget):
        self.budget = budget
        self.cache = DynamicCache()

    @property
    def length(self):
        if not self.cache.layers:
            return 0
        return self.cache.layers[0].get_seq_length()

    def update(self, new_cache):
        # The model mutates the cache in place, so new_cache is usually the
        # same object we passed in. Reassign for safety.
        self.cache = new_cache
        if self.length > self.budget:
            self._evict()

    def _evict(self):
        keep = self.budget
        for layer in self.cache.layers:
            layer.keys = layer.keys[..., -keep:, :].contiguous()
            layer.values = layer.values[..., -keep:, :].contiguous()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_fifo(model, tokenizer, prompt, max_new_tokens, budget,
                  chunk_size=256, device="cpu"):
    """Stream `prompt` through `model` with a FIFO cache of size `budget`,
    then autoregressively generate up to `max_new_tokens` greedy tokens.
    Returns only the generated continuation as a string.
    """
    cache = FIFOKVCache(budget)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    max_pos = model.config.n_positions - 1  # 1023 for GPT-2

    # --- Prefill: stream the prompt through in chunks ---
    true_position = 0
    last_logits = None
    for start in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, start:start + chunk_size]
        chunk_len = chunk.shape[1]
        positions = torch.arange(
            true_position, true_position + chunk_len, device=device
        ).clamp(max=max_pos).unsqueeze(0)
        out = model(
            input_ids=chunk,
            past_key_values=cache.cache,
            position_ids=positions,
            use_cache=True,
        )
        cache.update(out.past_key_values)
        last_logits = out.logits[:, -1, :]
        true_position += chunk_len

    # --- Decode ---
    generated = []
    for _ in range(max_new_tokens):
        next_tok = last_logits.argmax(dim=-1, keepdim=True)
        tok_id = next_tok.item()
        generated.append(tok_id)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break
        pos = torch.tensor([[min(true_position, max_pos)]], device=device)
        out = model(
            input_ids=next_tok,
            past_key_values=cache.cache,
            position_ids=pos,
            use_cache=True,
        )
        cache.update(out.past_key_values)
        last_logits = out.logits[:, -1, :]
        true_position += 1

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# Needle-in-haystack eval
# ---------------------------------------------------------------------------

def build_needle_prompt(tokenizer, needle, question, filler, distance_tokens):
    """Place `needle`, then roughly `distance_tokens` of `filler`, then `question`.
    Returns (prompt_str, actual_distance_tokens).
    """
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    while len(filler_ids) < distance_tokens:
        filler_ids = filler_ids + filler_ids
    filler_ids = filler_ids[:distance_tokens]

    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids

    full_ids = needle_ids + filler_ids + question_ids
    return tokenizer.decode(full_ids), len(filler_ids)


def needle_eval(model, tokenizer, budget, distances, device="cpu"):
    needle = "The secret access code is 73924."
    question = (" Question: What is the secret access code?"
                " Answer: The secret access code is")
    filler = ("The weather today is pleasant. Birds are singing in the trees. "
              "The market opened flat this morning. A light breeze moves through the park. ")

    results = {}
    for d in distances:
        prompt, actual_d = build_needle_prompt(tokenizer, needle, question, filler, d)
        out = generate_fifo(
            model, tokenizer, prompt,
            max_new_tokens=10, budget=budget, device=device,
        )
        hit = "73924" in out
        results[actual_d] = {"hit": hit, "output": out.strip()}
        print(f"  distance={actual_d:>5}  hit={hit}  output={out.strip()!r}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  transformers={transformers.__version__}")

    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device).eval()

    distances = [50, 200, 500, 900, 1500, 3000]
    budgets = [256, 512, 1024]

    all_results = {}
    for budget in budgets:
        print(f"\n=== budget={budget} ===")
        all_results[budget] = needle_eval(model, tokenizer, budget, distances, device=device)

    # Summary grid
    print("\nhit grid (rows=budget, cols=distance):")
    header = "budget\\dist  " + "  ".join(f"{d:>5}" for d in distances)
    print(header)
    for budget in budgets:
        row = f"{budget:>11}  " + "  ".join(
            f"{'  hit' if all_results[budget][d]['hit'] else ' miss':>5}"
            for d in distances
        )
        print(row)


if __name__ == "__main__":
    main()
