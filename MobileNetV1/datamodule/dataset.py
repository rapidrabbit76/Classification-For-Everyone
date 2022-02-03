from typing import Any, Tuple, Callable
from torchvision.datasets import CIFAR10, CIFAR100


class CIFAR10DataSet(CIFAR10):

    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = False,
            transform: Callable = None,
            image_size: int = 224,
    ) -> None:
        super().__init__(
            root,
            train=train,
            download=download,
        )
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, target


class CIFAR100DataSet(CIFAR100):

    def __init__(
            self,
            root: str,
            train: bool = True,
            download: bool = False,
            transform: Callable = None,
            image_size: int = 224,
    ) -> None:
        super().__init__(
            root,
            train=train,
            download=download,
        )
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return image, target