import sys
import os
import json
import argparse

# =====================================================================
# 1. Setup Command Line Arguments
# =====================================================================
parser = argparse.ArgumentParser(description="Download a subset of the MIT-Adobe FiveK dataset.")
parser.add_argument("--size", type=int, default=50, 
                    help="Number of images to download. (Default: 50)")
parser.add_argument("--split", type=str, choices=["train", "val", "test", "debug"], default="train", 
                    help="Which pre-defined split to pull from if not bypassing. (Default: train)")
parser.add_argument("--bypass-splits", action="store_true", 
                    help="If set, ignores the chosen split, pools all 5,000 images together, and pulls the target size from the entire dataset.")
parser.add_argument("--experts", type=str, nargs="*", default=["a"], 
                    help="Experts to download. E.g. --experts a b c. Pass nothing for RAW only. (Default: a)")
args = parser.parse_args()

# =====================================================================
# 2. Point Python to the submodule folder
# =====================================================================
submodule_path = os.path.join(os.path.dirname(__file__), 'third_party', 'mit-adobe-fivek-dataset')
sys.path.append(submodule_path)

# =====================================================================
# 3. Intercept JSON loading to limit size AND handle split bypassing
# =====================================================================
original_load = json.load

def truncate_data(data):
    """Truncates FiveK metadata while preserving its original shape."""
    if isinstance(data, list) and len(data) > args.size:
        return data[:args.size]
    if isinstance(data, dict):
        if len(data) > args.size:
            return dict(list(data.items())[:args.size])

        # Depending on how the repo structures the JSON, it might be in a dict key.
        truncated = data.copy()
        for key, value in truncated.items():
            if isinstance(value, list) and len(value) > args.size:
                truncated[key] = value[:args.size]
                break
        return truncated
    return data

def patched_load(fp, *args_json, **kwargs):
    # Check if the file being loaded is one of the FiveK metadata JSONs
    is_fivek_json = hasattr(fp, 'name') and any(
        x in fp.name for x in ['training.json', 'validation.json', 'testing.json', 'debugging.json']
    )

    # If it is a FiveK JSON and the user wants to bypass splits
    if is_fivek_json and args.bypass_splits:
        print("Bypass flag detected! Pooling train, val, and test splits together...")
        all_data = {}
        metadata_dir = os.path.dirname(fp.name)
        for split_file in ['training.json', 'validation.json', 'testing.json']:
            path = os.path.join(metadata_dir, split_file)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    split_data = original_load(f)
                if isinstance(split_data, dict):
                    all_data.update(split_data)
                else:
                    for item in split_data:
                        basename = item.get("basename") or item.get("id")
                        if basename is None:
                            raise ValueError(f"Cannot infer metadata key for item in {path}")
                        all_data[str(basename)] = item
        return truncate_data(all_data)

    # Normal behavior (but still truncated if it is a FiveK JSON)
    data = original_load(fp, *args_json, **kwargs)
    if is_fivek_json:
        return truncate_data(data)
    
    return data

# Apply the patch
json.load = patched_load

# =====================================================================
# 4. Import from the submodule and run the download
# =====================================================================
from datasets.fivek import MITAboveFiveK
from torch.utils.data.dataloader import DataLoader

if __name__ == "__main__":
    print(f"Initializing dataset download...")
    print(f"Target Size: {args.size}")
    print(f"Base Split: {args.split}")
    print(f"Experts: {args.experts}")
    
    # Initialize the dataset. The class still requires a 'split' argument to pass 
    # its internal checks, but our JSON interceptor will override what it actually loads.
    fivek_dataset = MITAboveFiveK(
        root="./fivek_data", 
        split=args.split,       
        download=True,       
        experts=args.experts,       
        download_workers=4   
    )

    data_loader = DataLoader(fivek_dataset, batch_size=None)

    print("\nVerifying local files:")
    count = 0
    for item in data_loader:
        print(f"[{count+1}/{args.size}] Ready: {item['basename']} | Path: {item['files']['dng']}")
        count += 1
        
    print(f"\nSuccess! Processed {count} images.")
