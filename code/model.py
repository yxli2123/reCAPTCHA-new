from resnet import resnet50, resnet101
import torch.nn as nn
import torch
from transformers import AutoModel


class ResNet(nn.Module):
    def __init__(self, num_cls=3000):
        super().__init__()
        self.num_character = num_character
        self.num_cls = num_cls

        # Initialize the vision model
        self.resnet = resnet50(pretrained=False, num_classes=num_cls)

    def forward(self, x):
        """
        x: (B, num_char, C, H, W)
        """
        B, C, H, W = x.shape
        y = self.resnet(x)                         # (B, num_cls)
        return y


if __name__ == '__main__':
    batch = torch.randn(4, 3, 40, 80)
    model = ResNet(num_character=1)
    out = model(batch)
