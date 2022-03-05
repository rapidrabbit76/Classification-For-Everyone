from .base import *


TRANSFORMS_TABLE: Dict["str", Callable] = {
    "BASE": BaseTransforms,
}

__all__ = [
    "BaseTransforms",
    "TRANSFORMS_TABLE",
]
