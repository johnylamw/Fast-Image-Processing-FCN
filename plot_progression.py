# 4.4 Training progression plots.
# Reads progression_eval.csv and draws PSNR-vs-iteration and SSIM-vs-iteration
# curves, one colour-coded line per dataset/model, with markers and a legend.
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    series = defaultdict(lambda: {"it": [], "psnr": [], "ssim": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            key = f"{r['dataset']}/{r['model']}"
            series[key]["it"].append(int(r["iteration"]))
            series[key]["psnr"].append(float(r["psnr"]))
            series[key]["ssim"].append(float(r["ssim"]))
    return series


def plot_metric(series, metric, ylabel, out):
    plt.figure(figsize=(7, 4.5))
    for key in sorted(series):
        s = series[key]
        order = sorted(range(len(s["it"])), key=lambda i: s["it"][i])
        xs = [s["it"][i] for i in order]
        ys = [s[metric][i] for i in order]
        plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=5, label=key)
    plt.xscale("log")
    plt.xlabel("Training iteration")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs training iteration (1080p, 500-image subsample)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/evaluate/progression_eval.csv")
    parser.add_argument("--outdir", default="output/figures")
    args = parser.parse_args()
    series = load(args.csv)
    plot_metric(series, "psnr", "PSNR (dB)",
                os.path.join(args.outdir, "progression_psnr.png"))
    plot_metric(series, "ssim", "SSIM",
                os.path.join(args.outdir, "progression_ssim.png"))


if __name__ == "__main__":
    main()
