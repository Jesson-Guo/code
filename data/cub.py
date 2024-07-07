import os
from typing import Callable, Optional

import numpy as np
import imageio
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg


class CUB(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = os.path.join(root, 'cub')

        img_name_list = []
        for line in open(os.path.join(self._base_folder, 'images.txt')):
            img_name_list.append(line[:-1].split(' ')[-1])

        label_list = []
        for line in open(os.path.join(self._base_folder, 'image_class_labels.txt')):
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)

        train_test_list = []
        for line in open(os.path.join(self._base_folder, 'train_test_split.txt')):
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        if self._split == "train":
            file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
            self.imgs = [imageio.imread(os.path.join(self._base_folder, 'images', f)) for f in file_list]
            self.labels = [x for i, x in zip(train_test_list, label_list) if i]
        else:
            file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
            self.imgs = [imageio.imread(os.path.join(self._base_folder, 'images', f)) for f in file_list]
            self.labels = [x for i, x in zip(train_test_list, label_list) if not i]

        self._samples = [
            (self.imgs[i], self.labels[i])
            for i in range(len(self.imgs))
        ]

        self.classes = []
        for line in open(os.path.join(self._base_folder, 'classes.txt')):
            self.classes.append(line[:-1].split(' ')[-1].split('.')[1])

        self.class_to_idx = {}
        for i in range(len(self.classes)):
            self.class_to_idx[self.classes[i]] = i

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, target = self._samples[idx]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
