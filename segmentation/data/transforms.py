import cv2
import torch
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = np.fliplr(image)
            target = np.fliplr(target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, self.size, interpolation=cv2.INTER_NEAREST)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        h, w, c = image.shape
        ratio = w / h
        hr = random.randint(self.min_size, self.max_size)
        wr = round(hr * ratio)
        size = (wr, hr)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, size, interpolation=cv2.INTER_NEAREST)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        h, w, c = image.shape
        xs = random.randint(0, w - self.size)
        ys = random.randint(0, h - self.size)
        xe = xs + self.size
        ye = ys + self.size
        image = image[ys:ye, xs:xe]
        target = target[ys:ye, xs:xe]
        return image, target


class RandomResizedCrop:
    def __init__(self, size, scale=(0.5, 2.0)):
        self.size = size
        self.scale = scale

    def __call__(self, image, target):
        h, w, c = image.shape

        s = random.uniform(self.scale[0], self.scale[1])
        w_crop, h_crop = round(self.size[0] / s), round(self.size[1] / s)
        xs = random.randint(0, w - w_crop)
        ys = random.randint(0, h - h_crop)
        xe = xs + w_crop
        ye = ys + h_crop
        image = image[ys:ye, xs:xe]
        target = target[ys:ye, xs:xe]

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, self.size, interpolation=cv2.INTER_NEAREST)

        return image, target


class ToTensor:
    def __init__(self):
        return

    def __call__(self, image, target):
        image = image.transpose(2, 0, 1)
        image = image.astype(np.float32) / 255
        target = target.astype(np.int64)
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).reshape(-1, 1, 1)
        self.std = torch.FloatTensor(std).reshape(-1, 1, 1)

    def __call__(self, image, target):
        image = (image - self.mean) / self.std
        return image, target
