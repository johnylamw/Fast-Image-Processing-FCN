import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
import torchvision.transforms.functional as TF
from PIL import Image

from dataset import ImageOperatorDataset
from utils import checkpoint_dataset_name, checkpoint_iteration, filter_pairs, get_device, load_model, synchronize_device

OUTPUT_DIR = "output/evaluate"

# converts an input/target pair to tensors at the eval resolution
def load_pair(input_path, target_path, short_edge):
    input_image = Image.open(input_path).convert("RGB")
    target_image = Image.open(target_path).convert("RGB")
    if short_edge is not None:
        input_image = TF.resize(input_image, short_edge)
        target_image = TF.resize(target_image, short_edge)
    return TF.to_tensor(input_image), TF.to_tensor(target_image)


# the three torchmetrics objects every code path scores with. Both the live-model
# path and the exported-prediction path go through this so the metric formulas are
# guaranteed identical (e.g. TF predictions are never scored by TF's own PSNR/SSIM).
def make_metrics(device):
    return (
        MeanSquaredError().to(device),
        PeakSignalNoiseRatio(data_range=1.0).to(device),
        StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
    )


def summarize_metrics(metrics, count, total_time):
    mse_metric, psnr_metric, ssim_metric = metrics
    return {
        "images": count,
        "mse": mse_metric.compute().item(),
        "psnr": psnr_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
        # blank for exported predictions: inference happened in another framework,
        # so cross-framework wall time is not comparable here. Read it from the
        # exporting framework's own train/export log instead.
        "time_ms": (1000.0 * total_time / count) if total_time else "",
    }


# runs evaluation and returns averaged metrics in dictionary format
def evaluate(model, pairs, device, short_edge):
    total_time = 0.0
    metrics = make_metrics(device)
    mse_metric, psnr_metric, ssim_metric = metrics
    count = len(pairs)

    for index, (input_path, target_path) in enumerate(pairs, start=1):
        input_tensor, target_tensor = load_pair(input_path, target_path, short_edge)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        target_tensor = target_tensor.unsqueeze(0).to(device)

        synchronize_device(device)
        start_time = time.perf_counter()
        with torch.no_grad():
            pred = model(input_tensor).clamp(0.0, 1.0)
        synchronize_device(device)
        total_time += time.perf_counter() - start_time

        mse_metric.update(pred, target_tensor)
        psnr_metric.update(pred, target_tensor)
        ssim_metric.update(pred, target_tensor)

        if index % 100 == 0 or index == count:
            print(f"Evaluation Progress: {index}/{count} images...")

    return summarize_metrics(metrics, count, total_time)


# scores a directory of pre-exported prediction PNGs (e.g. from tf_reference.py
# export) against the test-split targets, using the exact same metrics as the live
# model path. Predictions are matched to targets by input basename: <stem>.png.
def evaluate_predictions(pred_dir, pairs, device):
    metrics = make_metrics(device)
    mse_metric, psnr_metric, ssim_metric = metrics
    pred_dir = Path(pred_dir)
    total = len(pairs)
    missing = []
    count = 0

    for index, (input_path, target_path) in enumerate(pairs, start=1):
        pred_path = pred_dir / f"{input_path.stem}.png"
        if not pred_path.exists():
            missing.append(input_path.stem)
            continue

        pred_image = Image.open(pred_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        # Resize the target to the prediction's exact (H, W). Predictions were
        # already exported at the eval resolution; matching the grid this way
        # avoids cv2-vs-PIL short-side rounding differences that would otherwise
        # make torchmetrics fail on a 1-pixel size mismatch.
        target_image = TF.resize(target_image, [pred_image.height, pred_image.width])
        pred = TF.to_tensor(pred_image).clamp(0.0, 1.0).unsqueeze(0).to(device)
        target = TF.to_tensor(target_image).unsqueeze(0).to(device)

        mse_metric.update(pred, target)
        psnr_metric.update(pred, target)
        ssim_metric.update(pred, target)
        count += 1

        if index % 100 == 0 or index == total:
            print(f"Scoring Progress: {index}/{total} predictions...")

    if missing:
        preview = ", ".join(missing[:5])
        print(f"WARNING: {len(missing)} of {total} predictions missing (e.g. {preview})")
    if count == 0:
        raise ValueError(f"No predictions in {pred_dir} matched the test split")

    return summarize_metrics(metrics, count, total_time=0.0)

# writes csv output for the evaluated checkpoints
def save_results(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# parses args and evaluates checkpoints
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("checkpoints", nargs="*",
                        help="PyTorch checkpoint(s) to run over the test split. "
                             "Omit when using --predictions.")
    parser.add_argument("--short-edge", type=int, default=1080) # the paper uses 1080 for their experiments
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--predictions", default=None,
                        help="Directory of pre-exported prediction PNGs (named "
                             "<input-stem>.png) to score against the test-split "
                             "targets, instead of running a checkpoint. Used to "
                             "score the TF reference model on identical metrics.")
    parser.add_argument("--pred-label", default="predictions",
                        help="Model label recorded in the results row when using "
                             "--predictions (e.g. CAN32+AN_tf).")
    args = parser.parse_args()

    if not args.checkpoints and not args.predictions:
        parser.error("provide checkpoint(s) or --predictions")
    if args.checkpoints and args.predictions:
        parser.error("use either checkpoint(s) or --predictions, not both")

    device = get_device()
    dataset = ImageOperatorDataset(args.dataset)
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    split_path = os.path.join("data_splits", dataset_name, "test_split.txt")
    pairs = filter_pairs(dataset, split_path)
    if args.num_samples is not None:
        pairs = pairs[:args.num_samples]
    if not pairs:
        raise ValueError("No dataset pairs matched the test split")
    print("Evaluation Config:")
    print(f"Dataset: {args.dataset}")
    print(f"Test split: {split_path}")

    rows = []
    if args.predictions:
        # Score a directory of exported predictions (e.g. the TF reference model).
        print(f"Scoring predictions: {args.predictions}")
        metrics = evaluate_predictions(args.predictions, pairs, device)
        row = {
            "dataset": dataset_name,
            "trained_dataset": dataset_name,
            "model": args.pred_label,
            "checkpoint": args.predictions,
            "checkpoint_iteration": "",
            "test_split": split_path,
            # predictions carry their own export resolution; short_edge is set by
            # the exporter, not this script.
            "short_edge": "",
            **metrics,
        }
        rows.append(row)
        print(
            f"RESULTS FOR {dataset_name}/{args.pred_label}: "
            f"MSE={metrics['mse']:.6f}, "
            f"PSNR={metrics['psnr']:.3f}, "
            f"SSIM={metrics['ssim']:.4f}"
        )
        output_name = f"{dataset_name}_{args.pred_label}_predeval.csv"
    else:
        print(f"Evaluating {len(args.checkpoints)} checkpoint(s) on {len(pairs)} images")
        print(f"Short edge: {args.short_edge}")
        for checkpoint_path in args.checkpoints:
            model, _, model_name = load_model(checkpoint_path, device)
            print(f"\nEvaluating {model_name}: {checkpoint_path}")
            metrics = evaluate(model, pairs, device, args.short_edge)
            row = {
                "dataset": dataset_name,
                "trained_dataset": checkpoint_dataset_name(checkpoint_path),
                "model": model_name,
                "checkpoint": checkpoint_path,
                "checkpoint_iteration": checkpoint_iteration(checkpoint_path),
                "test_split": split_path,
                "short_edge": args.short_edge,
                **metrics,
            }
            rows.append(row)
            print(
                f"RESULTS FOR {checkpoint_dataset_name(checkpoint_path)}/{model_name}: "
                f"MSE={metrics['mse']:.6f}, "
                f"PSNR={metrics['psnr']:.3f}, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"Time={metrics['time_ms']:.2f} ms"
            )
        output_name = f"{dataset_name}_shortedge{args.short_edge}_eval.csv"

    output_path = os.path.join(OUTPUT_DIR, output_name)
    save_results(output_path, rows)
    print(f"\nEvaluation Results: {output_path}")

if __name__ == "__main__":
    main()
