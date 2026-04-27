"""
Train a SentencePiece BPE tokenizer on a Hugging Face dataset.

The output .model file is consumed by prepare_shards.py and train.py via the
same --tokenizer flag (paths ending in .model are auto-detected as SP).

Usage:
    python train_sentencepiece.py --dataset roneneldan/TinyStories \\
        --vocab-size 8192 \\
        --output ./data/tokenizers/tinystories_8192_bpe.model

For larger corpora, use --max-stories to subsample for training (the SP model
doesn't need every token of the corpus to find good merges).
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

try:
    import sentencepiece as spm
except ImportError as e:
    raise SystemExit(
        "sentencepiece is not installed. Run: pip install sentencepiece"
    ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--output", required=True,
                    help="Output path. .model and .vocab files will be written.")
    ap.add_argument("--max-stories", type=int, default=None,
                    help="Cap on training documents.")
    ap.add_argument("--max-sentence-length", type=int, default=4192)
    ap.add_argument("--character-coverage", type=float, default=0.9995)
    ap.add_argument("--input-sentence-size", type=int, default=10_000_000,
                    help="Soft cap on lines fed to the trainer.")
    args = ap.parse_args()

    output_path = Path(args.output)
    output_prefix = (str(output_path.with_suffix(""))
                     if output_path.suffix == ".model" else str(output_path))
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_stories and len(ds) > args.max_stories:
        ds = ds.select(range(args.max_stories))
    print(f"  {len(ds):,} documents")

    # Materialize to a temp text file (one document per line; SP handles multi-line).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                     encoding="utf-8") as tmp:
        tmp_path = tmp.name
        n_lines = 0
        for record in tqdm(ds, desc="materializing"):
            text = record["text"].strip()
            if not text:
                continue
            # Replace internal newlines with spaces so each doc is one line.
            text = text.replace("\n", " ")
            tmp.write(text + "\n")
            n_lines += 1
    print(f"  wrote {n_lines:,} lines to {tmp_path}")

    print(f"\nTraining SentencePiece BPE: vocab_size={args.vocab_size}")
    spm.SentencePieceTrainer.Train(
        input=tmp_path,
        model_prefix=output_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        max_sentence_length=args.max_sentence_length,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        character_coverage=args.character_coverage,
        byte_fallback=True,                  # handle OOV via bytes
        split_digits=True,                   # 1234 -> 1 2 3 4 (helps math/numerics)
        normalization_rule_name="identity",  # don't munge whitespace or case
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        train_extremely_large_corpus=False,
    )

    Path(tmp_path).unlink()

    model_file = f"{output_prefix}.model"
    print(f"\nWrote: {model_file}")
    print(f"Wrote: {output_prefix}.vocab")

    # Sanity check.
    sp = spm.SentencePieceProcessor(model_file=model_file)
    test_text = "Once upon a time, there was a little girl named Lucy. She loved cake!"
    ids = sp.encode_as_ids(test_text)
    pieces = [sp.id_to_piece(i) for i in ids]
    print(f"\nSanity check:")
    print(f"  text:       {test_text!r}")
    print(f"  ids ({len(ids)}): {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"  pieces:     {pieces[:20]}{'...' if len(pieces) > 20 else ''}")
    print(f"  decoded:    {sp.decode(ids)!r}")
    print(f"  vocab_size: {sp.vocab_size()}")
    print(f"  eos_id:     {sp.eos_id()}")
    if sp.piece_to_id("▁") != sp.unk_id():
        print(f"  ▁ token id: {sp.piece_to_id('▁')}  (good for BPB)")
    else:
        print("  warning: ▁ is not a standalone token in this vocab; "
              "BPB calculation will be approximate")


if __name__ == "__main__":
    main()
