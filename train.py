import torch
import torch.nn as nn

from dataset import ImageOperatorDataset, PairedRandomResizeToTensor
from torch.utils.data import DataLoader, RandomSampler, Subset

from model import CAN24AN, CAN24AND, CAN32AND, CAN32, CAN32AN, CAN32BN

import argparse
import csv
import os
import random
import time

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
MODEL_OUTPUT_DIR = "model_runs"
SPLITS_DIR = "data_splits"
MODEL_VARIANTS = {
    "CAN24+AN": CAN24AN,
    "CAN24+AND": CAN24AND,
    "CAN32": CAN32,
    "CAN32+AN": CAN32AN,
    "CAN32+AND": CAN32AND,
    "CAN32+BN": CAN32BN,
}

# TODO: we can add valid_loader if we hyperparameter tuning. original paper does not.
# can add metrics and valid_loader?
# paper uses "iterations" rather than epochs
def train(model,
    optimizer,
    loss_fn,
    train_loader,
    total_iterations=500_000,
    model_output_dir=MODEL_OUTPUT_DIR,
    split_manifest=None
):
    print(f"Training the model for {total_iterations} iterations")
    model.train()
    os.makedirs(model_output_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(model_output_dir))
    log_path = os.path.join(model_output_dir, f"{run_name}_train_log.csv")
    log_exists = os.path.exists(log_path)
    train_start_time = time.perf_counter()

    log_file = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "iteration",
            "loss",
            "elapsed_seconds",
            "seconds_per_iter",
            "batch_size",
            "height",
            "width",
            "pixels",
            "pixels_per_sec",
        ],
    )
    if not log_exists:
        log_writer.writeheader()
    
    # loop until total_iteraitons is met
    try:
        for iteration, (X_batch, y_batch) in enumerate(train_loader, start=1):
            if torch_device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            X_batch, y_batch = X_batch.to(torch_device), y_batch.to(torch_device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            if torch_device == "cuda":
                torch.cuda.synchronize()
            seconds_per_iter = time.perf_counter() - start_time
            elapsed_seconds = time.perf_counter() - train_start_time

            batch_size = X_batch.shape[0]
            height = X_batch.shape[2]
            width = X_batch.shape[3]
            pixels = batch_size * height * width
            
            print(f"Iteration {iteration}/{total_iterations}, loss={loss.item():.6f}")
            log_writer.writerow(
                {
                    "iteration": iteration,
                    "loss": loss.item(),
                    "elapsed_seconds": elapsed_seconds,
                    "seconds_per_iter": seconds_per_iter,
                    "batch_size": batch_size,
                    "height": height,
                    "width": width,
                    "pixels": pixels,
                    "pixels_per_sec": pixels / seconds_per_iter,
                }
            )
            log_file.flush()
            
            # Save checkpoint every 10k iterations
            if iteration % 10_000 == 0:
                checkpoint_path = os.path.join(model_output_dir, f"{run_name}_iter_{iteration}.pt")
                save_model(checkpoint_path, model, optimizer, iteration, split_manifest)
    finally:
        log_file.close()

def save_model(path, model, optimizer, iteration, split_manifest=None):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "split_manifest": split_manifest,
        },
        path
    )

def make_or_load_split(dataset, split_dir, train_fraction=0.5):
    os.makedirs(split_dir, exist_ok=True)
    train_manifest_path = os.path.join(split_dir, "train_manifest.txt")
    test_manifest_path = os.path.join(split_dir, "test_manifest.txt")
    basename_to_index = {
        input_path.stem: index
        for index, (input_path, _) in enumerate(dataset.pairs)
    }

    if os.path.exists(train_manifest_path) and os.path.exists(test_manifest_path):
        train_basenames = read_manifest(train_manifest_path)
        test_basenames = read_manifest(test_manifest_path)
    else:
        basenames = list(basename_to_index)
        random.Random(SEED).shuffle(basenames)
        train_size = int(len(basenames) * train_fraction)
        train_basenames = basenames[:train_size]
        test_basenames = basenames[train_size:]
        write_manifest(train_manifest_path, train_basenames)
        write_manifest(test_manifest_path, test_basenames)

    split = {
        "dataset_size": len(dataset),
        "seed": SEED,
        "train_fraction": train_fraction,
        "train_manifest": train_manifest_path,
        "test_manifest": test_manifest_path,
        "train_basenames": train_basenames,
        "test_basenames": test_basenames,
    }

    train_indices = manifest_to_indices(train_basenames, basename_to_index)
    test_indices = manifest_to_indices(test_basenames, basename_to_index)
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)
    return train_set, test_set, split


# read the train_test split manifest
def read_manifest(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# write the train/test split manifest for reproducability + evaluation later
def write_manifest(path, basenames):
    with open(path, "w") as f:
        for basename in basenames:
            f.write(f"{basename}\n")


def manifest_to_indices(basenames, basename_to_index):
    missing = [basename for basename in basenames if basename not in basename_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Manifest contains missing dataset items: {preview} (first 5 are shown)")
    return [basename_to_index[basename] for basename in basenames]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/adobe5kA")
    parser.add_argument("--model", choices=MODEL_VARIANTS.keys(), default="CAN24+AN")
    parser.add_argument("--iterations", default=500_000, type=int)
    parser.add_argument("--outputs", default=MODEL_OUTPUT_DIR)
    parser.add_argument("--splits", default=SPLITS_DIR)
    return parser.parse_args()


def dataset_split_dir(splits_dir, dataset_path):
    return os.path.join(splits_dir, dataset_name(dataset_path))


def dataset_name(dataset_path):
    return os.path.basename(os.path.normpath(dataset_path))


def dataset_model_output_dir(outputs_dir, dataset_path, model_name):
    return os.path.join(outputs_dir, dataset_name(dataset_path), model_name)


if __name__ == "__main__":
    """
    NOTES:
    - images of varying resolution resized between 320 - 1440 (preserves aspect ratio)
    - uses Adam
    - 500k iterations
    - no hyperparameter tuning
    - ~ 1 day of training on Nvidia Titan
    """

    args = parse_args()
    checkpoint_dir = dataset_model_output_dir(args.outputs, args.dataset, args.model)
    split_dir = dataset_split_dir(args.splits, args.dataset)

    total_iterations = args.iterations
    random_tensor_transform = PairedRandomResizeToTensor()
    dataset = ImageOperatorDataset(args.dataset, transform=random_tensor_transform)
    train_set, test_set, split_manifest = make_or_load_split(dataset, split_dir)
    sampler = RandomSampler(train_set, replacement=True, num_samples=total_iterations)
    dataloader = DataLoader(
        train_set,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=torch_device == "cuda"
    )

    print("Dataset loaded into dataloader!")

    # TODO: no hyperparameter tuning atm
    model = MODEL_VARIANTS[args.model]().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.MSELoss()

    train(
        model,
        optimizer,
        loss,
        dataloader,
        total_iterations=total_iterations,
        model_output_dir=checkpoint_dir,
        split_manifest=split_manifest
    )

    run_name = os.path.basename(os.path.normpath(checkpoint_dir))
    final_model_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    save_model(final_model_path, model, optimizer, total_iterations, split_manifest)
