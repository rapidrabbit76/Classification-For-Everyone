from .base import *
from .medium import *

TRANSFORMS_TABLE: Dict["str", Callable] = {
    "BASE": BaseTransforms,
    "MEDUIM": MediumTransforms,
}

__all__ = [
    "BaseTransforms",
    "MediumTransforms",
    "TRANSFORMS_TABLE",
]
