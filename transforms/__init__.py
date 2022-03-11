from .base import *
from .light import *
from .medium import *

TRANSFORMS_TABLE: Dict["str", Callable] = {
    "BASE": BaseTransforms,
    "LIGHT": LightTransforms,
    "MEDIUM": MediumTransforms,
}

__all__ = [
    "BaseTransforms",
    "LightTransforms",
    "MediumTransforms",
    "TRANSFORMS_TABLE",
]
