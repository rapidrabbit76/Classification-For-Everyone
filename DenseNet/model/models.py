import torch
from torch import nn

from .blocks import ConvBlock, DenseBlock, TransitionBlock

DENSE_NET_TYPE = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '265': [6, 12, 64, 48],
}


class DenseNet(nn.Module):

    def __init__(
        self,
        model_type: str = "121",
        image_channals: int = 3,
        n_classes: int = 1000,
        growth_rate: int = 12,
    ) -> None:
        super().__init__()
        layers = []
        dim = growth_rate * 2
        layers += [nn.Conv2d(image_channals, dim, 7, 2, 3, bias=False)]
        layers += [nn.MaxPool2d(3, 2, padding=1)]

        model_type = DENSE_NET_TYPE[model_type]

        for idx, nblock in enumerate(model_type):
            layers += [DenseBlock(nblock, dim, growth_rate)]
            dim += growth_rate * nblock
            if idx == len(model_type) - 1:
                continue
            layers += [TransitionBlock(dim, dim // 2)]
            dim = dim // 2
        self.feature_extractor = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
