#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from concurrent.futures import ThreadPoolExecutor


def make_dataset(root, subset: str) -> list[tuple[Path, Path]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))

    return list(zip(images, full_labels))

def make_volume_dataset(root, subset: str) -> list[tuple[Path, Path]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images = sorted(img_path.glob("*.png"))
    full_labels = sorted(full_path.glob("*.png"))
    # till here same as above    

    
    # get all slices for a volume
    volumes = [[]]
    labels = [[]]
    
    patient_id = int(images[0].stem.split("_")[1])
    for image, label in zip(images, full_labels):
        if patient_id == int(image.stem.split("_")[1]):
            assert patient_id == int(label.stem.split("_")[1]), f"Patient ID mismatch: {image.stem} != {label.stem}"
            volumes[-1].append(image)
            labels[-1].append(label)
        else:
            patient_id = int(image.stem.split("_")[1])
            assert patient_id == int(label.stem.split("_")[1]), f"Patient ID mismatch: {image.stem} != {label.stem}"
            volumes.append([image])
            labels.append([label])
        
    return list(zip(volumes, labels))
        

class SliceDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir: Path,
        img_transform: Callable = None,
        gt_transform: Callable = None,
        augment: bool = False,
        equalize: bool = False,
        debug: bool = False,
    ):
        self.root_dir: Path = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        subset = f"'{subset.capitalize()}'"
        print(f">> Created {subset:<7} dataset with {len(self)} images!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(
            read_image(str(img_path), mode=ImageReadMode.GRAY)
        )
        gt: Tensor = self.gt_transform(
            read_image(str(gt_path), mode=ImageReadMode.GRAY)
        )

        # img: Tensor = self.img_transform(Image.open(str(img_path)))
        # gt: Tensor = self.gt_transform(Image.open(str(gt_path)))

        print(img.shape, gt.shape)
        _, W, H = img.shape
        K, _, _ = gt.shape
        # / assert gt.shape == (K, W, H)

        return {"images": img, "gts": gt, "stems": img_path.stem, "shape": (K, W, H)}
    
    
class VolumeDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir: Path,
        img_transform: Callable = None,
        gt_transform: Callable = None,
        augment: bool = False,
        equalize: bool = False,
        debug: bool = False,
        num_workers: int = 6,  # Add num_workers parameter
    ):
        self.root_dir: Path = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.num_workers = num_workers

        self.files = make_volume_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        subset = f"'{subset.capitalize()}'"
        print(f">> Created {subset:<7} dataset with {len(self)} volumes!")

    def __len__(self):
        return len(self.files)

    def _load_slice(self, img_gt_paths: tuple[Path, Path]):
        """Helper function to load a single slice image and ground truth."""
        img_path, gt_path = img_gt_paths
        img = self.img_transform(
            read_image(str(img_path), mode=ImageReadMode.GRAY)
        )
        gt = self.gt_transform(
            read_image(str(gt_path), mode=ImageReadMode.GRAY)
        )
        return img, gt

    def __getitem__(self, index) -> dict[str, Union[torch.Tensor, int, str]]:
        img_paths, gt_paths = self.files[index]

        # Use a ThreadPoolExecutor to load slices in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_slice, zip(img_paths, gt_paths)))

        # Reconstruct the volume and ground truth from slices
        volume = torch.cat([img for img, _ in results], dim=0)
        volume_gt = torch.cat([gt for _, gt in results], dim=0)

        D, W, H = volume.shape
        K, _, _ = volume_gt.shape
        assert volume_gt.shape == (K, W, H), "Mismatch in ground truth and volume shape"

        return {"volume": volume, "gt": volume_gt, "shape": (D, W, H)}