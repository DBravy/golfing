"""Quick smoke test of the TTT model and inner loop."""
import torch
import sys
sys.path.insert(0, "/home/claude/ttt_toy")
from ttt_model import TTTTransformer, ttt_forward_and_loss, standard_forward_and_loss

torch.manual_seed(0)
device = "cpu"

# Build small model
model_ttt = TTTTransformer(
    vocab_size=64,
    d_model=64,
    n_layers=4,
    n_heads=4,
    ff_mult=4,
    max_seq_len=128,
    window_size=128,
    n_ttt_blocks=1,
).to(device)

model_std = TTTTransformer(
    vocab_size=64,
    d_model=64,
    n_layers=4,
    n_heads=4,
    ff_mult=4,
    max_seq_len=128,
    window_size=128,
    n_ttt_blocks=0,  # pure standard
).to(device)

print(f"TTT model params:      {model_ttt.num_params():,}")
print(f"Standard model params: {model_std.num_params():,}")
assert model_ttt.num_params() == model_std.num_params(), "Param count mismatch!"

# Forward smoke test
tokens = torch.randint(0, 64, (1, 65), device=device)  # seq 65, predicts 64 = 4 mini-batches of 16

# Standard forward
loss_std = standard_forward_and_loss(model_std, tokens)
print(f"Standard loss: {loss_std.item():.4f}")
loss_std.backward()
g = next(model_std.parameters()).grad
print(f"Standard grad shape: {g.shape}, nonzero: {(g != 0).any().item()}")

# TTT forward with create_graph=True
opt = torch.optim.SGD(model_ttt.parameters(), lr=0.01)
opt.zero_grad()
loss_ttt = ttt_forward_and_loss(model_ttt, tokens, mini_batch_size=16, inner_lr=1.0, create_graph=True)
print(f"TTT loss (training):   {loss_ttt.item():.4f}")
loss_ttt.backward()
g = next(model_ttt.parameters()).grad
print(f"TTT grad shape: {g.shape}, nonzero: {(g != 0).any().item()}")

# Check inner-loop fast MLP gets gradient
ttt_block = [b for b in model_ttt.blocks if b.is_ttt][0]
fast_grad = ttt_block.mlp_fast.fc1.weight.grad
print(f"Fast MLP grad nonzero: {(fast_grad != 0).any().item()}, norm: {fast_grad.norm().item():.4f}")

# Slow MLP should also get gradient (it's part of the main forward)
slow_grad = ttt_block.mlp_slow.fc1.weight.grad
print(f"Slow MLP grad nonzero: {(slow_grad != 0).any().item()}, norm: {slow_grad.norm().item():.4f}")

# Attention should get gradient
attn_grad = ttt_block.attn.qkv.weight.grad
print(f"Attn QKV grad nonzero: {(attn_grad != 0).any().item()}, norm: {attn_grad.norm().item():.4f}")

# TTT eval (no create_graph)
model_ttt.eval()
loss_eval = ttt_forward_and_loss(model_ttt, tokens, mini_batch_size=16, inner_lr=1.0, create_graph=False)
print(f"TTT loss (eval):       {loss_eval.item():.4f}")

# Quick overfit test: should the TTT loss decrease on the same sequence over a few steps?
model_ttt = TTTTransformer(vocab_size=64, d_model=64, n_layers=4, n_heads=4,
                           ff_mult=4, max_seq_len=128, window_size=128,
                           n_ttt_blocks=1).to(device)
opt = torch.optim.AdamW(model_ttt.parameters(), lr=3e-3)
print("\nOverfit test (TTT):")
for step in range(20):
    opt.zero_grad()
    loss = ttt_forward_and_loss(model_ttt, tokens, mini_batch_size=16, inner_lr=1.0, create_graph=True)
    loss.backward()
    opt.step()
    if step % 5 == 0 or step == 19:
        print(f"  step {step:3d}  loss {loss.item():.4f}")

print("\nOverfit test (standard):")
model_std = TTTTransformer(vocab_size=64, d_model=64, n_layers=4, n_heads=4,
                           ff_mult=4, max_seq_len=128, window_size=128,
                           n_ttt_blocks=0).to(device)
opt = torch.optim.AdamW(model_std.parameters(), lr=3e-3)
for step in range(20):
    opt.zero_grad()
    loss = standard_forward_and_loss(model_std, tokens)
    loss.backward()
    opt.step()
    if step % 5 == 0 or step == 19:
        print(f"  step {step:3d}  loss {loss.item():.4f}")

print("\nSmoke test passed.")
