from asyncio.constants import ACCEPT_RETRY_DELAY
from curses import meta
import os
from turtle import down
from typing import Any, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100, ImageNet
import pandas as pd
import numpy as np


class ImageNetDataset(Dataset):
    ARCHIVE_META = {
        "train": "LOC_train_solution.csv",
        "test": "LOC_val_solution.csv",
    }
    IMAGE_DIR = "ILSVRC/Data/CLS-LOC"

    def __init__(
        self,
        root: str,
        train: Optional[str] = "train",
        image_size: int = 224,
    ):
        super().__init__()

        path = os.path.join(root, self.ARCHIVE_META[train])
        df = pd.read_csv(path)

        image_id = df["ImageId"].tolist()
        label = list(map(lambda x: x.split()[0], df["PredictionString"].tolist()))

        metadata = list(zip(image_id, label))

        if train == "train":
            self.data = [
                (os.path.join(root, self.IMAGE_DIR, "train", l, p + ".JPEG"), l)
                for p, l in metadata
            ]
        else:
            self.data = [
                (os.path.join(root, self.IMAGE_DIR, "val", p + ".JPEG"), l)
                for p, l in metadata
            ]

        self.transform = A.Compose(
            [
                A.Resize(256, 256),
                A.RandomCrop(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensor(),
            ],
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.data[index]
        image = np.array(Image.open(path))
        image = self.transform(image=image)["image"]
        return image, target

    def __len__(self):
        return len(self.data)


