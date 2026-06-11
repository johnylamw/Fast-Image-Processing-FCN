import os
import torch
from model import CAN24AN, CAN24AND, CAN32, CAN32AN, CAN32AND, CAN32BN

MODEL_VARIANTS = {
    "CAN24+AN": CAN24AN,
    "CAN24+AND": CAN24AND,
    "CAN32": CAN32,
    "CAN32+AN": CAN32AN,
    "CAN32+AND": CAN32AND,
    "CAN32+BN": CAN32BN,
}

# get gpu/mps/cpu based on what is available
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# syncs device timing when gpu/mps runs asynchronously
def synchronize_device(device):
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()

# gets the model name from the checkpoint directory
def checkpoint_model_name(checkpoint_path):
    return os.path.basename(os.path.dirname(checkpoint_path))

# gets the dataset name from the checkpoint directory
def checkpoint_dataset_name(checkpoint_path):
    return os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))

# formats the checkpoint label used in demo figures
def checkpoint_display_name(checkpoint_path):
    return f"{checkpoint_dataset_name(checkpoint_path)}/{checkpoint_model_name(checkpoint_path)}"

# loads model weights based on checkpoint path
def load_model(checkpoint_path, device):
    model_name = checkpoint_model_name(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MODEL_VARIANTS[model_name]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, model_name

# reads the basenames in a split file
def read_split(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

# filters dataset pairs by basename
def pairs_matching_basenames(dataset, basenames):
    basenames = set(basenames)
    return [
        (input_path, target_path)
        for input_path, target_path in dataset.pairs
        if input_path.stem in basenames
    ]

# filters dataset pairs by a split file
def filter_pairs(dataset, split_path):
    if split_path is None:
        return dataset.pairs
    pairs = pairs_matching_basenames(dataset, read_split(split_path))
    if not pairs:
        raise ValueError(f"No dataset pairs matched {split_path}")
    return pairs
