import torch
import torch.nn as nn

from dataset import ImageOperatorDataset, PairedRandomResizeToTensor
from torch.utils.data import DataLoader, RandomSampler, Subset

from model import CAN32

import os

torch_device = "cuda"

SEED = 42
CHECKPOINT_DIR = "checkpoint"
SPLIT_FILENAME = "split_indices.pt"

# TODO: we can add valid_loader if we hyperparameter tuning. original paper does not.
# can add metrics and valid_loader?
# paper uses "iterations" rather than epochs
def train(model,
    optimizer,
    loss_fn,
    train_loader,
    total_iterations=500_000,
    checkpoint_dir=CHECKPOINT_DIR,
    split_indices=None,
):
    print(f"Training the model for {total_iterations} iterations")
    model.train()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # loop until total_iteraitons is met
    for iteration, (X_batch, y_batch) in enumerate(train_loader, start=1):
        X_batch, y_batch = X_batch.to(torch_device), y_batch.to(torch_device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {iteration}/{total_iterations}, loss={loss.item():.6f}")
        
        # Save checkpoint every 10k iterations
        if iteration % 10000 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_iter_{iteration}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, iteration, split_indices)

def save_checkpoint(path, model, optimizer, iteration, split_indices=None):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "split_indices": split_indices,
        },
        path
    )

def make_or_load_split(dataset, checkpoint_dir=CHECKPOINT_DIR, train_fraction=0.5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    split_path = os.path.join(checkpoint_dir, SPLIT_FILENAME)

    # loads the splits if it exists for evaluation
    if os.path.exists(split_path):
        split = torch.load(split_path, map_location="cpu")
        if split["dataset_size"] != len(dataset):
            raise ValueError(f"Saved split expects {split['dataset_size']} images but dataset has {len(dataset)} images.")
        train_set = Subset(dataset, split["train_indices"])
        test_set = Subset(dataset, split["test_indices"])
        return train_set, test_set, split

    train_size = int(len(dataset) * train_fraction)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )
    split = {
        "dataset_size": len(dataset),
        "seed": SEED,
        "train_fraction": train_fraction,
        "train_indices": train_set.indices,
        "test_indices": test_set.indices,
    }
    torch.save(split, split_path)
    return train_set, test_set, split


if __name__ == "__main__":
    """
    NOTES:
    - images of varying resolution resized between 320 - 1440 (preserves aspect ratio)
    - uses Adam
    - 500k iterations
    - no hyperparameter tuning
    - ~ 1 day of training on Nvidia Titan
    """

    total_iterations = 500_000
    random_tensor_transform = PairedRandomResizeToTensor()
    dataset = ImageOperatorDataset("datasets/adobe5k_processed", transform=random_tensor_transform)
    train_set, test_set, split_indices = make_or_load_split(dataset, CHECKPOINT_DIR)
    sampler = RandomSampler(train_set, replacement=True, num_samples=total_iterations)
    
    dataloader = DataLoader(
        train_set,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset loaded into dataloader!")

    # TODO: no hyperparameter tuning atm
    model = CAN32().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.MSELoss()

    train(
        model,
        optimizer,
        loss,
        dataloader,
        total_iterations=total_iterations,
        checkpoint_dir=CHECKPOINT_DIR,
        split_indices=split_indices,
    )
