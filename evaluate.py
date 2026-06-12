import argparse
import csv
import os
import time

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


# runs evaluation and returns averaged metrics in dictionary format
def evaluate(model, pairs, device, short_edge):
    total_time = 0.0
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
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

    return {
        "images": count,
        "mse": mse_metric.compute().item(),
        "psnr": psnr_metric.compute().item(),
        "ssim": ssim_metric.compute().item(),
        "time_ms": 1000.0 * total_time / count,
    }

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
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--short-edge", type=int, default=1080) # the paper uses 1080 for their experiments
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

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
    print(f"Evaluating {len(args.checkpoints)} checkpoint(s) on {len(pairs)} images")
    print(f"Dataset: {args.dataset}")
    print(f"Test split: {split_path}")
    print(f"Short edge: {args.short_edge}")

    rows = []
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
