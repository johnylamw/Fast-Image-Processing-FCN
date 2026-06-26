# 4.5 Training-loss plot: MSE training loss vs iteration for every model.
# Batch-1 random-resolution training makes the per-iteration loss extremely
# noisy, so we plot a moving average on a log y-axis (one colour per model).
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("adobe5kA", "CAN24+AN"), ("adobe5kA", "CAN24+AND"),
    ("adobe5kA", "CAN32+AN"), ("adobe5kA", "CAN32+AND"),
    ("flickr2k", "CAN32+AN"), ("div2k", "CAN32+AN"),
]


def load(ds, model):
    path = f"model_runs/{ds}/{model}/{model}_train_log.csv"
    its, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                its.append(int(row["iteration"]))
                losses.append(float(row["loss"]))
            except (ValueError, KeyError):
                continue
    return np.array(its), np.array(losses)


def smooth(y, w):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--out", default="output/figures/training_loss.png")
    args = parser.parse_args()

    plt.figure(figsize=(7, 4.5))
    for ds, model in RUNS:
        its, losses = load(ds, model)
        ys = smooth(losses, args.window)
        xs = its[args.window - 1:] if len(losses) >= args.window else its
        xs = xs[:len(ys)]
        step = max(1, len(xs) // 1500)  # lighten the figure
        plt.plot(xs[::step], ys[::step], linewidth=1.4, label=f"{ds}/{model}")
        print(f"{ds}/{model}: {len(its)} iters, final loss ~{losses[-1]:.4f}")

    plt.yscale("log")
    plt.xlabel("Training iteration")
    plt.ylabel(f"MSE training loss (moving avg, w={args.window})")
    plt.title("Training loss vs iteration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print("Saved", args.out)


if __name__ == "__main__":
    main()
