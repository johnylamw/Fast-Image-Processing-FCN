# Builds paste-ready markdown tables for the report straight from the eval
# CSVs, so the numbers never get hand-transcribed. Emits to stdout and to
# output/report_tables.md.
#   - 4.3 cross-resolution (adobe5kA CAN32+AN at several short edges)
#   - 4.4 PSNR-by-iteration and SSIM-by-iteration (every model x checkpoint)
import csv
import os
from collections import defaultdict

RES_CSV = "output/evaluate/adobe5kA_CAN32+AN_resolution_sweep.csv"
PROG_CSV = "output/evaluate/progression_eval.csv"
OUT_MD = "output/report_tables.md"

ITERS = [10000, 20000, 50000, 100000, 250000, 500000]
ITER_LABELS = {10000: "10k", 20000: "20k", 50000: "50k",
               100000: "100k", 250000: "250k", 500000: "500k"}
# Stable model ordering for the 4.4 tables.
MODEL_ORDER = [
    ("adobe5kA", "CAN24+AN"), ("adobe5kA", "CAN24+AND"),
    ("adobe5kA", "CAN32+AN"), ("adobe5kA", "CAN32+AND"),
    ("flickr2k", "CAN32+AN"), ("div2k", "CAN32+AN"),
]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def resolution_table():
    rows = sorted(read_rows(RES_CSV), key=lambda r: int(r["short_edge"]))
    out = ["#### 4.3 Cross-Resolution Generalization (`CAN32+AN`, Adobe5kA, 500-image subsample)",
           "",
           "| Resolution | Model | MSE | PSNR (dB) | SSIM | Time (ms) |",
           "|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| {r['short_edge']}p | `{r['model']}` | {float(r['mse']):.6f} | "
            f"{float(r['psnr']):.3f} | {float(r['ssim']):.3f} | {float(r['time_ms']):.1f} |")
    return "\n".join(out)


def progression_tables():
    rows = read_rows(PROG_CSV)
    # (dataset, model) -> iter -> {psnr, ssim}
    data = defaultdict(dict)
    for r in rows:
        data[(r["dataset"], r["model"])][int(r["iteration"])] = {
            "psnr": float(r["psnr"]), "ssim": float(r["ssim"])}

    def one(metric, fmt, title):
        out = [title, "",
               "| Model | " + " | ".join(ITER_LABELS[i] for i in ITERS) + " |",
               "|---|" + "---:|" * len(ITERS)]
        for ds, model in MODEL_ORDER:
            series = data.get((ds, model), {})
            vals = [series.get(i, {}).get(metric) for i in ITERS]
            present = [v for v in vals if v is not None]
            best = max(present) if present else None
            cells = []
            for v in vals:
                if v is None:
                    cells.append("—")
                elif v == best:
                    cells.append(f"**{v:{fmt}}**")  # peak in bold
                else:
                    cells.append(f"{v:{fmt}}")
            out.append(f"| `{ds}/{model}` | " + " | ".join(cells) + " |")
        return "\n".join(out)

    psnr = one("psnr", ".3f",
               "#### 4.4 PSNR (dB) by training iteration (1080p, 500-image subsample; peak in bold)")
    ssim = one("ssim", ".3f",
               "#### 4.4 SSIM by training iteration (1080p, 500-image subsample; peak in bold)")
    return psnr + "\n\n" + ssim


def main():
    blocks = [resolution_table(), progression_tables()]
    md = "\n\n".join(blocks) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print(f"(written to {OUT_MD})")


if __name__ == "__main__":
    main()
