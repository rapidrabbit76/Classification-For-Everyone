import pytorch_lightning as pl

from .VGG import *
from .LeNet5 import *
from .SqueezeNet import *
from .DenseNet import *
from .ResNeXt import *
from .WideResNet import *
from .ShuffleNet import *
from .EfficientNetV2 import *
from .Xception import *
from .InceptionNet import *
from .AlexNet import *
from .ResNet import *
from .GoogLeNet import *
from .MobileNetV1 import *
from .MobileNetV2 import *
from .MobileNetV3 import *
from .MNASNet import *
from .EfficientNetV1 import *


MODEL_TABLE: Dict["str", pl.LightningModule] = {
    "VGG": LitVGG,
    "LeNet5": LitLeNet5,
    "SqueezeNet": LitSqueezeNet,
    "DenseNet": LitDenseNet,
    "ResNeXt": LitResNeXt,
    "WideResNet": LitWideResNet,
    "ShuffleNetV2": LitShuffleNetV2,
    "EfficientNetV2": LitEfficientNetV2,
    "XceptionNet": LitXceptionNet,
    "Inception": LitInceptionV3,
    "AlexNet": LitAlexNet,
    "GoogLeNet": LitGoogLeNet,
    "ResNet": LitResNet,
    "MobileNetV1": LitMobileNetV1,
    "MobileNetV2": LitMobileNetV2,
    "MobileNetV3": LitMobileNetV3,
    "MNASNet": LitMNASNet,
    "EfficientNetV1": LitEfficientNet,
}

__all__ = [
    "LitVGG",
    "LitLeNet5",
    "LitSqueezeNet",
    "LitDenseNet",
    "LitResNeXt",
    "LitWideResNet",
    "LitShuffleNetV2",
    "LitEfficientNetV2",
    "LitXceptionNet",
    "LitInceptionV3",
    "LitAlexNet",
    "LitResNet",
    "LitGoogLeNet",
    "LitMobileNetV1",
    "LitMobileNetV2",
    "LitMobileNetV3",
    "LitMNASNet",
    "LitEfficientNet",
    "MODEL_TABLE",
]
