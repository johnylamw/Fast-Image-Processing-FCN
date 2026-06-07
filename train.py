import torch
import torch.nn as nn

from dataset import ImageOperatorDataset, PairedRandomResizeToTensor
from torch.utils.data import DataLoader, RandomSampler

from model import CAN32

import os

torch_device = "cpu"


# TODO: we can add valid_loader if we hyperparameter tuning. original paper does not.
# can add metrics and valid_loader?

# paper uses "iterations" rather than epochs
def train(model, optimizer, loss_fn, train_loader, total_iterations=500_000):
    print(f"Training the model for {total_iterations} iterations")
    model.train()
    iteration = 0
    
    # loop until total_iteraitons is met
    for iteration, (X_batch, y_batch) in enumerate(train_loader, start=1):
        if iteration >= total_iterations:
            break
            
        X_batch, y_batch = X_batch.to(torch_device), y_batch.to(torch_device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {iteration}/{total_iterations}, loss={loss.item():.6f}")
        iteration += 1
        
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}/{total_iterations}, loss={loss.item():.6f}")
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoint/model_iter_{iteration}.pt")

        # Save checkpoint every 50k iterations
        if iteration % 50000 == 0:
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoint/model_LARGE_iter_{iteration}.pt")
            
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
    sampler = RandomSampler(dataset, replacement=True, num_samples=total_iterations)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset loaded into dataloader!")

    # TODO: no hyperparameter tuning atm
    model = CAN32().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.MSELoss()

    train(model, optimizer, loss, dataloader, total_iterations=total_iterations)
