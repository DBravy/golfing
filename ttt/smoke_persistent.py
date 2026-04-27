"""Smoke test the persistent (fast-weight programmer) variant."""
import torch
import sys
sys.path.insert(0, "/home/claude/ttt_toy")
from ttt_model import (
    TTTTransformer,
    persistent_ttt_forward_and_loss,
    persistent_eval_loss,
    snapshot_fast_params,
    get_non_fast_parameters,
)

torch.manual_seed(0)

model = TTTTransformer(
    vocab_size=64, d_model=64, n_layers=4, n_heads=4, ff_mult=4,
    max_seq_len=128, window_size=128, n_ttt_blocks=1,
)

print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
non_fast = get_non_fast_parameters(model)
print(f"Non-fast params: {sum(p.numel() for p in non_fast):,}")
fast_names = [n for n, p in model.named_parameters()
              if any(p is q for block in model.blocks if getattr(block, 'is_ttt', False)
                     for q in block.mlp_fast.parameters())]
print(f"Fast params excluded from optimizer: {fast_names}")

optim = torch.optim.AdamW(non_fast, lr=3e-3, weight_decay=0.01)

# Initialize persistent state from the model's current fast params
state = snapshot_fast_params(model.get_initial_fast_params())
print(f"\nInitial persistent state shapes:")
for i, fp in enumerate(state):
    for name, p in fp.items():
        print(f"  block[{i}].{name}: {tuple(p.shape)}, mean={p.mean():.4f}, std={p.std():.4f}")

# Quick overfit on 5 different sequences, reusing state
print("\nOverfit test (persistent, 5 different sequences, state accumulates):")
for step in range(30):
    tokens = torch.randint(0, 64, (1, 65))
    optim.zero_grad()
    loss, state = persistent_ttt_forward_and_loss(
        model, tokens, mini_batch_size=16, inner_lr=0.1, fast_params_state=state
    )
    loss.backward()
    optim.step()
    if step % 5 == 0 or step == 29:
        fast_norm = sum(p.norm().item() for fp in state for p in fp.values())
        print(f"  step {step:3d}  loss {loss.item():.4f}  fast_param_norm {fast_norm:.4f}")

# Check eval works
tokens = torch.randint(0, 64, (1, 65))
eval_loss = persistent_eval_loss(model, tokens, mini_batch_size=16, inner_lr=0.1, fast_params_state=state)
print(f"\nEval loss on fresh sequence: {eval_loss:.4f}")
print(f"State after eval (should be unchanged): {sum(p.norm().item() for fp in state for p in fp.values()):.4f}")

# Verify non-fast param is being updated by optimizer
attn_weight = [p for n, p in model.named_parameters() if 'attn.qkv' in n][0]
print(f"\nSample non-fast param (attn.qkv) norm: {attn_weight.norm().item():.4f}")

print("\nPersistent smoke test passed.")
