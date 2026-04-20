# TTT variants vs. Standard Transformer

Three models with identical parameter counts, differing only in how the
fast-path MLP weights are updated.

## The three modes

**`standard`** (`n_ttt_blocks=0`)
Plain transformer with sliding-window attention. Regular backprop.

**`ttt`** (bi-level TTT-E2E, the paper)
Last `n_ttt_blocks` are TTT blocks with a fast MLP and a slow MLP.
At every outer step, for each sequence:
1. Reset fast MLP weights to the current model's values (`W_0`).
2. Inner loop: mini-batch SGD on next-token loss, producing
   `W_0 -> W_1 -> ... -> W_n`.
3. Outer loss is the sum of per-mini-batch NLLs using the weights
   in effect at the start of each mini-batch.
4. Backprop through the whole chain (gradient of gradients) to update `W_0`.
The fast-MLP adaptation is discarded at sequence boundaries.

**`persistent`** (fast-weight programmer variant)
Same architecture as `ttt`, but:
1. The fast MLP weights are NOT reset between sequences. They persist
   across the entire training run as a single accumulating state.
2. The inner loop uses first-order gradients only (no `create_graph`).
   No second-order derivatives.
3. The outer optimizer updates only the non-fast parameters (attention,
   slow MLPs, norms, embeddings). Fast MLPs are excluded from AdamW and
   are updated exclusively by the inner-loop SGD.
4. At eval time, a snapshot of the persistent state is taken before
   evaluation. Each eval sequence starts from that snapshot, runs its
   own inner loop, and the snapshot is restored between eval sequences
   so adaptation does not leak. The snapshot is also restored at the
   end of eval so eval never affects training.

The system being tested by `persistent`: a two-timescale setup where the
fast MLPs are continuously-adapting working memory and everything else
is a learned "mechanism" that shapes how adaptation is used.

## Files

* `ttt_model.py`        model + all three loss functions
* `train.py`            data, training loop, comparison, plotting
* `smoke_test.py`       sanity check for the bi-level TTT path
* `smoke_persistent.py` sanity check for the persistent path

## Run

```bash
pip install torch datasets matplotlib
python train.py                   # all three modes, ~300 outer steps
python train.py --mode ttt        # bi-level only
python train.py --mode persistent # persistent only
python train.py --mode standard   # baseline only
python train.py --max_steps 100   # short run
```

First run caches TinyStories validation split to `./data/`. Without
network access, falls back to repetitive sample text so the pipeline
still runs end-to-end.

## Default config

| name                   | value | applies to          |
|------------------------|-------|---------------------|
| d_model                | 128   | all                 |
| n_layers               | 8     | all                 |
| n_heads                | 4     | all                 |
| ff_mult                | 4     | all                 |
| n_ttt_blocks           | 2     | ttt, persistent     |
| seq_len                | 256   | all                 |
| window_size            | 256   | all                 |
| mini_batch_size        | 32    | ttt, persistent     |
| inner_lr               | 1.0   | ttt                 |
| persistent_inner_lr    | 0.1   | persistent          |
| persistent_inner_wd    | 0.0   | persistent          |
| lr (outer)             | 3e-4  | all                 |
| accum_steps            | 8     | all                 |

`persistent_inner_lr` defaults to an order of magnitude smaller than
`inner_lr` because those updates accumulate across the whole training
run instead of being thrown away per sequence. With the bi-level setup
every inner step starts from `W_0`, so aggressive steps are fine; here
the updates compound, so a smaller step size makes sense as a starting
point. Worth sweeping.

`persistent_inner_wd` is an optional multiplicative decay on the fast
params at each inner step. Set to something like 1e-4 if the fast-param
norm keeps climbing without bound over long runs. 0 by default.

## What to look at

* `results.json`   train + eval NLL, wall-clock per step, config
* `results.png`    eval NLL vs step (left) and vs wall-clock (right)

A fourth thing printed in the training log for persistent: `fast_norm`,
the total norm of the persistent fast-weight state. If this grows
without bound over the run, raise `persistent_inner_wd` or lower
`persistent_inner_lr`.

## Expected shape of the comparison

On the vanilla baseline `ttt` beat `standard` slightly but at roughly
10x wall-clock per outer step because of the unrolled second-order
backward. Per outer step (same outer optimization budget):

* `standard` should be the fastest in wall-clock by a wide margin.
* `ttt` should reach a lower loss per outer step in the best case
  (the meta-learned `W_0` is a good starting point for inner-loop
  adaptation), but at the highest wall-clock cost.
* `persistent` should sit between the two in wall-clock. No
  gradient-of-gradients, so the overhead compared to `standard` is
  just the 7 inner forwards plus 6 first-order inner backwards per
  sequence, not the second-order traversal on top.

Whether `persistent` reaches comparable or better final loss per
outer step is the open question this setup is meant to probe. If
`persistent` is competitive with `ttt` on loss but cheaper in wall-
clock, that is evidence that the bi-level meta-learning is doing
less work than the inner loop itself, which would be interesting.

## Design notes

* Per-sequence runs use B=1 plus gradient accumulation. TTT's inner
  loop is per-sequence in the paper; B>1 would require vmap or would
  share fast params across the batch in a way that departs from the
  paper. B=1 sidesteps the ambiguity.
* In `persistent` mode the fast MLPs are `nn.Parameter`s of the model
  but are excluded from the outer optimizer. Their values are
  overridden per call via `torch.func.functional_call` using the
  persistent state. This means AdamW never touches them, including
  no weight decay from AdamW.
* The two MLPs in a TTT block are combined additively in parallel,
  not in series. Parameter match with a regular block is exact.
* GPT-2-style init (std=0.02), weight tying between embedding and head.

## Smoke tests

```bash
python smoke_test.py        # bi-level TTT
python smoke_persistent.py  # persistent variant
```

Both should report matched parameter counts, initial losses near
`log(vocab)`, nonzero gradients where they should be zero-free, and
loss dropping over a handful of overfitting steps.
