from torch.utils.data import Dataset
from PIL import Image
import os

import random
import torchvision.transforms.functional as F

class ImageOperatorDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        if os.path.exists(dataset_dir):
            input_dir = os.path.join(dataset_dir, "inputs")
            target_dir = os.path.join(dataset_dir, "targets")
            if not os.path.exists(input_dir):
                raise FileNotFoundError(f"Input Image Directory {input_dir} does not exist")
            if not os.path.exists(target_dir):
                raise FileNotFoundError(f"Target (Image Operator) Directory {target_dir} does not exist")
            
            self.input_paths = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir)])
            self.target_paths = sorted([os.path.join(target_dir, file) for file in os.listdir(target_dir)])
        else:
            raise FileNotFoundError(f"Dataset Directory {dataset_dir} does not exist")
        
        self.transform = transform
        
    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        img = Image.open(self.input_paths[index]).convert("RGB")
        target = Image.open(self.target_paths[index]).convert("RGB")

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

if __name__ == "__main__":
    # test = ImageOperatorDataset("dataset/div2k", transform=None)
    pass
