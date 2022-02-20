from typing import List, Tuple, Union
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image

__all__ = ["BaseTransforms"]


class BaseTransforms:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        image_shape: List[int],
        train: Union[int, bool, str] = False,
        mean: Tuple[float, float, float] = None,
        std: Tuple[float, float, float] = None,
    ) -> None:
        c, image_size, _ = image_shape
        train = True if isinstance(train, str) and train == "train" else False

        mean = self.mean if mean is None else mean
        std = self.std if std is None else std

        assert c == len(mean)
        assert c == len(std)

        self.transforms = A.Compose(
            [
                A.RandomResizedCrop(
                    height=image_size,
                    width=image_size,
                    p=1.0 if train else 0.0,
                ),
                A.HorizontalFlip(p=1.0 if train else 0.0),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        image = np.array(image) if not isinstance(image, np.ndarray) else image
        return self.transforms(image=image)["image"]