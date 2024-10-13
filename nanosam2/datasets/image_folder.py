# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
import PIL.Image
import os, numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, Resize
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple

class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
def default_transform():
    transform = Compose([
        Resize((1024, 1024)),
        ToTensor(),
        Normalize(
            mean=[123.675/255, 116.28/255, 103.53/255],
            std=[58.395/255, 57.12/255, 57.375/255]
        )
    ])
    return transform


def static_transform(x, img_size=1024):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    transform = ResizeLongestSide(img_size)
    x = transform.apply_image(x)
    x = torch.as_tensor(x)
    x = x.permute(2, 0, 1).contiguous() 

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x / 255.0

class ImageFolder:
    def __init__(self, root: str, transform = None):
        self.root = root
        image_paths = glob.glob(os.path.join(root, "*.jpg"))
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.image_paths[index]).convert("RGB")
        image = self.transform(image)
        return self.image_paths[index], image
    
class SA1Folder:
    def __init__(self, sa1_datasets: list, root: str):
        self.root = root
        train_dirs = ["sa_" + str(i).zfill(6) for i in sa1_datasets]
        self.jpg_paths = []
        for train_dir in train_dirs:
            self.jpg_paths += glob.glob(os.path.join(root, train_dir, "*.jpg"))

        print(f"Found {len(self.jpg_paths)} jpg files in {root}")

        self.resolution = 1024
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transform = torch.jit.script(
            torch.nn.Sequential(
                RandomResizedCrop((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __len__(self) -> int:
        return len(self.jpg_paths)
    
    def __getitem__(self, index):
        image = np.asarray(PIL.Image.open(self.jpg_paths[index]).convert("RGB"))
        # Make a writable copy of the array
        image = np.copy(image)
        image_t = self.to_tensor(image)
        image_t = self.transform(image_t)
        return image_t
    
class NpySA1Folder:
    def __init__(self, sa1_datasets: list, root: str):
        self.root = root
        train_dirs = ["sa_" + str(i).zfill(6) for i in sa1_datasets]
        self.npy_paths = []
        for train_dir in train_dirs:
            self.npy_paths += glob.glob(os.path.join(root, train_dir, "*.npy"))

        print(f"Found {len(self.npy_paths)} npy files in {root}")
        # self.transform = default_transform()

    def __len__(self) -> int:
        return len(self.npy_paths)
    
    def __getitem__(self, index):
        image = np.asarray(PIL.Image.open(self.npy_paths[index].replace("npy", "jpg")).convert("RGB"))
        image = static_transform(image)
        return image, np.load(self.npy_paths[index]).squeeze()   

class NpySA1ImageFolder:
    def __init__(self, sa1_datasets: list, root: str):
        self.root = root
        train_dirs = ["sa_" + str(i).zfill(6) for i in sa1_datasets]
        self.npy_paths = []
        for train_dir in train_dirs:
            self.npy_paths += glob.glob(os.path.join(root, train_dir, "*.jpg"))

        print(f"Found {len(self.npy_paths)} npy files in {root}")

    def __len__(self) -> int:
        return len(self.npy_paths)
    
    def __getitem__(self, index):
        img_path = self.npy_paths[index].replace("npy", "jpg")
        image = np.asarray(PIL.Image.open(img_path).convert("RGB"))
        image = static_transform(image)
        return img_path, image


class NpyFolder:
    def __init__(self, root: str):
        self.root = root
        self.npy_paths = glob.glob(os.path.join(root, "*.npy"))

        self.transform = default_transform()
    def __len__(self) -> int:
        return len(self.npy_paths)
    
    def __getitem__(self, index):
        image = PIL.Image.open(self.npy_paths[index].replace("npy", "jpg")).convert("RGB")
        image = self.transform(image)
        return image, np.load(self.npy_paths[index]).squeeze()   