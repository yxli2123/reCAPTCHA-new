from resnet import resnet50, resnet101
import torch.nn as nn
import torch
from transformers import AutoModel


class ResNet(nn.Module):
    def __init__(self, num_character, num_cls=200):
        super().__init__()
        self.num_character = num_character
        self.num_cls = num_cls

        # Initialize the vision model
        self.resnet = resnet50(pretrained=False, num_classes=num_cls)

    def forward(self, x):
        """
        x: (B, num_char, C, H, W)
        """
        B, num_char, C, H, W = x.shape
        x = x.reshape(B*num_char, C, H, W)         # (num_char*B, C, H, W)
        y = self.resnet(x)                         # (num_char*B, num_cls)
        y = y.reshape(B, num_char, self.num_cls)   # (B, num_char, num_cls)
        return y


if __name__ == '__main__':
    batch = torch.randn(4, 3, 40, 80)
    model = ResNet(num_character=1)
    out = model(batch)
