from torch.utils.data import Dataset
from PIL import Image

import random
import torchvision.transforms.functional as F

from pathlib import Path

class ImageOperatorDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, operator_suffix="_pencil"):
        dataset_dir = Path(dataset_dir)
        input_dir = dataset_dir / "inputs"
        target_dir = dataset_dir / "targets"
        if not input_dir.exists():
            raise FileNotFoundError(f"Input Image Directory {input_dir} does not exist")
        if not target_dir.exists():
            raise FileNotFoundError(f"Target (Image Operator) Directory {target_dir} does not exist")

        image_extensions = {".jpg", ".jpeg", ".png"}
        input_paths = sorted(
            path for path in input_dir.iterdir()
            if path.suffix.lower() in image_extensions
        )
        # pairs inputs -> targets
        # note: stem retrieves just the filename
        # ex: img01 : "./datasets/img01_pencil.png"
        target_by_stem = {
            target.stem.removesuffix(operator_suffix): target
            for target in target_dir.iterdir()
            if target.suffix.lower() in image_extensions
        }

        # check + create absolute pairs
        pairs = []
        missing_targets = []
        for input_path in input_paths:
            target_path = target_by_stem.get(input_path.stem)
            if target_path is None:
                missing_targets.append(input_path.name)
                continue
            pairs.append((input_path, target_path))

        if missing_targets:
            preview = ", ".join(missing_targets[:5])
            raise FileNotFoundError(f"Missing targets for {preview} (first 5 is shown)")

        self.pairs = pairs
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        img = Image.open(input_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        if self.transform:
            img, target = self.transform(img, target)

        return img, target


class PairedRandomResizeToTensor:
    """
    Apply the same random resize to an input/target image pair, then convert both
    images to tensors.
    """
    # paper uses 320-1440 for Adobe5k and Raise -> change for other datasets
    def __init__(self, min_res=320, max_res=1440):
        self.min_res = min_res
        self.max_res = max_res

    def __call__(self, img, target):
        target_res = random.randint(self.min_res, self.max_res)
        img = F.resize(img, target_res)
        target = F.resize(target, target_res)
        return F.to_tensor(img), F.to_tensor(target)

class PairedFixedResizeToTensor:
    """
    Converts both input/target image pair to Tensor without explicit resizing
    """
    def __init__(self, res=640):
        self.res = res

    def __call__(self, img, target):
        img = F.resize(img, self.res)
        target = F.resize(target, self.res)
        return F.to_tensor(img), F.to_tensor(target)

class PairedToTensor:
    """
    Converts both input/target image pair to Tensor without explicit resizing
    """
    def __call__(self, img, target):
        return F.to_tensor(img), F.to_tensor(target)

if __name__ == "__main__":
    test = ImageOperatorDataset("datasets/div2k", transform=None)
    pass
