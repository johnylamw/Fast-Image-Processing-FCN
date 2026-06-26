# 4.4 Training progression over checkpoints.
# Evaluates each model at a fixed set of iteration checkpoints on a fixed
# (seeded) per-dataset subsample, at the 1080p eval resolution. Collects every
# (dataset, model, iteration) row into one CSV for the progression plots.
import argparse
import csv
import os
import random

from dataset import ImageOperatorDataset
from evaluate import evaluate
from utils import filter_pairs, get_device, load_model

OUTPUT_DIR = "output/evaluate"

# (dataset_dir, model_name). The four Adobe variants share the same iteration
# checkpoints; the two cross-dataset models are CAN32+AN.
RUNS = [
    ("datasets/adobe5kA", "CAN24+AN"),
    ("datasets/adobe5kA", "CAN24+AND"),
    ("datasets/adobe5kA", "CAN32+AN"),
    ("datasets/adobe5kA", "CAN32+AND"),
    ("datasets/flickr2k", "CAN32+AN"),
    ("datasets/div2k", "CAN32+AN"),
]
# The only iteration points every model (incl. the CAN24 variants) has saved.
ITERS = [10000, 20000, 50000, 100000, 250000, 500000]


def sampled_pairs(dataset_dir, num_samples, seed):
    dataset = ImageOperatorDataset(dataset_dir)
    name = os.path.basename(os.path.normpath(dataset_dir))
    split = os.path.join("data_splits", name, "test_split.txt")
    pairs = sorted(filter_pairs(dataset, split), key=lambda pr: pr[0].stem)
    n = min(num_samples, len(pairs))
    return random.Random(seed).sample(pairs, n), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-edge", type=int, default=1080)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "progression_eval.csv"))
    args = parser.parse_args()
    device = get_device()

    rows = []
    for dataset_dir, model_name in RUNS:
        name = os.path.basename(os.path.normpath(dataset_dir))
        pairs, n = sampled_pairs(dataset_dir, args.num_samples, args.seed)
        print(f"=== {name}/{model_name} on {n} images @ {args.short_edge}p ===")
        for it in ITERS:
            ckpt = os.path.join("model_runs", name, model_name, f"{model_name}_iter_{it}.pt")
            if not os.path.exists(ckpt):
                print(f"  MISSING {ckpt}, skipping")
                continue
            model, _, _ = load_model(ckpt, device)
            m = evaluate(model, pairs, device, args.short_edge)
            rows.append({
                "dataset": name, "model": model_name, "iteration": it,
                "images": n, "mse": m["mse"], "psnr": m["psnr"], "ssim": m["ssim"],
            })
            print(f"  iter={it}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
