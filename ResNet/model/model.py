import torch
import torch.nn as nn
from .block import ConvBlock, ResidualBlock, Classifier


RESNET_TYPE = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}


class ResNet(nn.Module):

    def __init__(
            self,
            image_channels: int,
            n_classes: int,
            model_type: int,
    ):
        super().__init__()
        dim = 64
        layers = []
        layers += [ConvBlock(image_channels, dim, kernel_size=7, stride=2, padding=3)]
        layers += [nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # stack blocks
        listBlocks = RESNET_TYPE[model_type]
        for idx, nblock in enumerate(listBlocks):
            layers += [ResidualBlock(model_type, idx, nblock, dim)]
            dim *= 2

        layers += [nn.AvgPool2d(kernel_size=7)]

        if model_type < 50:
            dim = dim // 2
        else:
            dim = dim * 2

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = Classifier(in_features=int(dim), out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)