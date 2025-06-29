import os
import torch
import torchvision
import numpy as np
from PIL import Image
from segmentation.data.comma10k import comma10_dataset


class Comma10k(torch.utils.data.Dataset):
    colors = np.asarray(
        [
            (0x00, 0xFF, 0x66),  # movable
            (0x40, 0x20, 0x20),  # road
            (0x80, 0x80, 0x60),  # undrivable
            (0xCC, 0x00, 0xFF),  # my car
            (0xFF, 0x00, 0x00),  # lane markings
        ],
        dtype=np.uint8,
    )

    def __init__(self, root, split="train", transform=None):
        self.transform = transform
        if split != "train":
            split = "val"
        self.files = comma10_dataset[split]
        self.image_root = os.path.join(root, "imgs")
        self.label_root = os.path.join(root, "masks")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def getdata(self, idx):
        file = self.files[idx]
        image_path = os.path.join(self.image_root, file)
        label_path = os.path.join(self.label_root, file)
        image = Image.open(image_path)
        label = Image.open(label_path).convert("RGB")
        image = np.asarray(image)
        label = np.asarray(label)
        label = (label[:, :, 0].astype(np.int64) + 1) >> 6
        return image, label

    @classmethod
    def get_color(cls, label):
        return cls.colors[label]


class Cityscapes(torch.utils.data.Dataset):
    mapping_20 = {
        7: 1,
        8: 2,
        11: 3,
        12: 4,
        13: 5,
        17: 6,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        31: 17,
        32: 18,
        33: 19,
    }

    colors = np.asarray(
        [
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
        dtype=np.uint8,
    )

    def __init__(self, root, split="train", transform=None):
        super().__init__()
        self.transform = transform
        self.dataset = torchvision.datasets.Cityscapes(root, split, "fine", "semantic")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def getdata(self, idx):
        image, target = self.dataset[idx]
        image = np.asarray(image)
        target = np.asarray(target)
        label = np.zeros_like(target, dtype=np.int64)
        for k in self.mapping_20:
            label[target == k] = self.mapping_20[k]
        return image, label

    @classmethod
    def get_color(cls, label):
        return cls.colors[label]
