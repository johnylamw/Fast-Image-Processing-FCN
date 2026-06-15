"""Generate a tiny synthetic dataset in the repo's inputs/targets + split layout,
purely so the TF reference pipeline (tf_reference.py) can be GPU smoke-tested before
committing to a multi-hour real run.

The image *content* is meaningless (random gradients + a cheap edge-map "operator"
target); the point is only to exercise: TF builds, a train step runs on GPU, a
checkpoint + final .keras are written, and export produces prediction PNGs that
evaluate.py --predictions can then score. Not a quality benchmark.

Layout produced (mirrors ImageOperatorDataset / build_pairs):
    <root>/inputs/<name>.png
    <root>/targets/<name>_pencil.png
    <splits>/<name-of-root>/train_split.txt
    <splits>/<name-of-root>/test_split.txt

Usage:
    python make_toy_dataset.py                     # -> dataset/_toy, data_splits/_toy
    python make_toy_dataset.py --root dataset/_toy --splits data_splits --num 6
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def make_pair(rng: np.random.Generator, h: int, w: int):
    """Return (input_bgr, target_bgr) uint8 images for cv2.imwrite."""
    # input: smooth random gradient so resizing has something to interpolate
    base = rng.random((8, 8, 3), dtype=np.float32)
    inp = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    inp = np.clip(inp, 0.0, 1.0)
    inp_u8 = np.uint8(inp * 255.0)

    # target: cheap "pencil"-like operator = inverted gray edges. Content is
    # irrelevant; we just need a deterministic, learnable-ish mapping.
    gray = cv2.cvtColor(inp_u8, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    tgt_gray = 255 - edges
    tgt_u8 = cv2.cvtColor(tgt_gray, cv2.COLOR_GRAY2BGR)
    return inp_u8, tgt_u8


def main():
    p = argparse.ArgumentParser(description="Generate a toy inputs/targets dataset")
    p.add_argument("--root", default="dataset/_toy",
                   help="Dataset dir to create (gets inputs/ and targets/).")
    p.add_argument("--splits", default="data_splits",
                   help="Splits root; writes <splits>/<root-name>/{train,test}_split.txt")
    p.add_argument("--num", type=int, default=6, help="Total pairs to generate.")
    p.add_argument("--test", type=int, default=2, help="How many go in the test split.")
    p.add_argument("--suffix", default="_pencil", help="Target filename suffix.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root)
    input_dir = root / "inputs"
    target_dir = root / "targets"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.splits) / root.name
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    names = []
    # vary sizes a little (and keep them small) to exercise the short-side resize
    sizes = [(160, 240), (200, 200), (180, 320), (240, 160), (200, 280), (160, 160)]
    for i in range(args.num):
        h, w = sizes[i % len(sizes)]
        name = f"toy{i:02d}"
        inp, tgt = make_pair(rng, h, w)
        cv2.imwrite(str(input_dir / f"{name}.png"), inp)
        cv2.imwrite(str(target_dir / f"{name}{args.suffix}.png"), tgt)
        names.append(name)

    test_names = names[-args.test:] if args.test else []
    train_names = names[:len(names) - len(test_names)]
    (split_dir / "train_split.txt").write_text("\n".join(train_names) + "\n")
    (split_dir / "test_split.txt").write_text("\n".join(test_names) + "\n")

    print(f"Wrote {args.num} pairs to {input_dir} / {target_dir}")
    print(f"Splits: {split_dir} (train={len(train_names)}, test={len(test_names)})")


if __name__ == "__main__":
    main()
