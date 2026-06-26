# 4.3 Cross-resolution plot: PSNR and SSIM vs evaluation short-edge resolution
# for the CAN32+AN model, with a twin axis. Companion to the 4.3 table.
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/evaluate/adobe5kA_CAN32+AN_resolution_sweep.csv")
    parser.add_argument("--out", default="output/figures/resolution_quality.png")
    args = parser.parse_args()

    rows = sorted(csv.DictReader(open(args.csv)), key=lambda r: int(r["short_edge"]))
    res = [int(r["short_edge"]) for r in rows]
    psnr = [float(r["psnr"]) for r in rows]
    ssim = [float(r["ssim"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "tab:blue"
    ax1.set_xlabel("Evaluation short-edge resolution (px)")
    ax1.set_ylabel("PSNR (dB)", color=color1)
    ax1.plot(res, psnr, marker="o", color=color1, linewidth=1.8, label="PSNR")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(res)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("SSIM", color=color2)
    ax2.plot(res, ssim, marker="s", color=color2, linewidth=1.8, label="SSIM")
    ax2.tick_params(axis="y", labelcolor=color2)

    # mark the 1080p main-eval resolution and training range edges
    ax1.axvspan(320, 1440, color="green", alpha=0.06)
    ax1.axvline(1080, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title("CAN32+AN quality vs evaluation resolution\n(shaded = 320–1440 training range, dashed = 1080p main eval)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
