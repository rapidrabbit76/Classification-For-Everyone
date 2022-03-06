import torch
from torch import nn

Activation = nn.ReLU


class ConvBlock(nn.Module):
    def __init__(self, inp: int, outp: int, k: int = 3, s: int = 1, p: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(
            inp,
            outp,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False,
        )
        self.act = Activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.act(x)


class FireModule(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        e1x1: int,
        e3x3: int,
    ):
        super().__init__()
        self.squeeze = ConvBlock(inp, outp, 1)
        self.e1x1 = ConvBlock(outp, e1x1, 1)
        self.e3x3 = ConvBlock(outp, e3x3, 3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.e1x1(x), self.e3x3(x)], 1)
