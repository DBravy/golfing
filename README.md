# Vanilla TTT-E2E vs Standard Transformer

A faithful, small-scale PyTorch implementation of the TTT-E2E mechanism from
Tandon et al., "End-to-End Test-Time Training for Long Context" (2025),
set up for a head-to-head comparison against a standard transformer at
matched parameter count.

## What this is

Two models, same architecture, same parameter count. The only difference:

* `standard` (`n_ttt_blocks=0`): plain transformer with sliding-window attention.
* `ttt` (`n_ttt_blocks=2`): last two blocks are TTT blocks with a fast MLP
  (updated in the inner loop) and a slow MLP (static). Each has half the
  hidden dim of a regular MLP so the total parameter count matches.

During training and evaluation in `ttt` mode, the inner loop runs over
mini-batches of each sequence, updating the fast MLP weights via SGD
(paper Eq. 5). The outer loop backpropagates through these updates using
`torch.autograd.grad(create_graph=True)` (paper Eq. 6).

Character-level tokenization on TinyStories. Model defaults to ~1.6M params.

## Files

* `ttt_model.py`  model definition and the inner-loop loss function
* `train.py`      data loading, training loop, comparison, plotting
* `smoke_test.py` quick sanity check that shapes, gradients, and overfitting all work

## Run

```bash
pip install torch datasets matplotlib
python train.py                    # both modes, ~300 outer steps
python train.py --mode ttt         # TTT only
python train.py --mode standard    # baseline only
python train.py --max_steps 100    # short run for a quick look
```

First run downloads the TinyStories validation split to `./data/` (about
20 MB). Without network access, it falls back to repetitive sample text
so the code still runs end to end.

## Default config

| name              | value | note                                     |
|-------------------|-------|------------------------------------------|
| d_model           | 128   |                                          |
| n_layers          | 8     |                                          |
| n_heads           | 4     |                                          |
| ff_mult           | 4     |                                          |
| n_ttt_blocks      | 2     | last 1/4 of the layers                   |
| seq_len           | 256   |                                          |
| window_size       | 256   | = seq_len, so SWA is effectively full    |
| mini_batch_size   | 32    | gives 7 inner-loop steps per sequence    |
| inner_lr          | 1.0   | SGD step for the fast MLP, likely tune   |
| lr (outer)        | 3e-4  | AdamW on all outer parameters            |
| accum_steps       | 8     | gradient accumulation, batch_size=1      |

`inner_lr` is the most important knob to sweep. It has no published default
so 1.0 is a guess. Try 0.3 and 3.0 if the first run looks off.

## What to look at

* `results.json`   logged train + eval NLL, wall-clock per step
* `results.png`    eval NLL vs step (left) and vs wall-clock (right)

Expected behavior on this setup:

* Standard will train faster in wall-clock by a large factor (roughly an
  order of magnitude per outer step) because TTT unrolls the inner loop
  and backpropagates through it.
* Per outer step (same amount of outer-loop optimization), TTT may or may
  not beat standard. The paper's own result is that at pretraining scale,
  TTT is slightly worse than full-attention on standard NLL benchmarks,
  and the win shows up on long-context retrieval. At this toy scale with
  window = seq_len, we are mainly validating that the mechanism trains
  cleanly and that the second-order gradients through the inner loop do
  not blow up.
* If the TTT curve is much worse than standard per outer step, the
  inner_lr is probably wrong. That is the first thing to sweep.

## Design notes, for reference

* Per-sequence TTT with B=1 and gradient accumulation. The paper's inner
  loop is per-sequence. With B>1 in this code the fast params are shared
  across the batch (a simplification). Using B=1 avoids the ambiguity.
* The two MLPs in a TTT block are combined additively in parallel, not
  in series. The paper is not explicit about this; parallel keeps the
  block structure closest to a regular block and makes the parameter
  match exact.
* Only MLP fast weights are updated in the inner loop. Attention, norms,
  embeddings, slow MLPs are all frozen during TTT, per the paper.
* Weight tying between the embedding and the output head.
* GPT-2-style init (std=0.02) so the initial NLL sits near log(vocab).

## Smoke test

```bash
python smoke_test.py
```

Should report matched parameter counts for TTT vs standard, initial losses
near log(vocab), nonzero gradients on fast MLP / slow MLP / attention, and
loss decreasing over 20 overfitting steps for both modes.
