# 4.3 Cross-resolution generalization.
# Evaluates a single checkpoint at several short-edge resolutions on a fixed
# (seeded) subsample of the test split. Writes its own CSV so it never
# clobbers the committed full-split eval files (evaluate.py reuses one path).
import argparse
import csv
import os
import random

from dataset import ImageOperatorDataset
from evaluate import evaluate
from utils import filter_pairs, get_device, load_model

OUTPUT_DIR = "output/evaluate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=[240, 480, 720, 1080, 1440])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = get_device()
    dataset = ImageOperatorDataset(args.dataset)
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join("data_splits", dataset_name, "test_split.txt")

    pairs = sorted(filter_pairs(dataset, split_path), key=lambda pr: pr[0].stem)
    n = min(args.num_samples, len(pairs))
    pairs = random.Random(args.seed).sample(pairs, n)
    print(f"Resolution sweep: {dataset_name} on {n} images at {args.resolutions}")

    model, _, model_name = load_model(args.checkpoint, device)
    rows = []
    for res in args.resolutions:
        m = evaluate(model, pairs, device, res)
        rows.append({
            "dataset": dataset_name, "model": model_name,
            "short_edge": res, "images": n,
            "mse": m["mse"], "psnr": m["psnr"], "ssim": m["ssim"],
            "time_ms": m["time_ms"],
        })
        print(f"  res={res}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
              f"time={m['time_ms']:.2f}ms")

    out = args.out or os.path.join(
        OUTPUT_DIR, f"{dataset_name}_{model_name}_resolution_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
