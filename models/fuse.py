import torch
import torch.nn as nn
import torch.nn.functional as F



# 将rgb图和光流图融合
class FeatureFusionBlock(nn.Module):
    def __init__(self, transform=None):
        super(FeatureFusionBlock, self).__init__()
        self.transform = transform
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, g):
        _, _, x_h, x_w = x.size()
        b, c, g_h, g_w = g.size()
        if x_h != g_h or x_w != g_w:
            g = F.interpolate(g, size=(x_h, x_w), mode='bilinear')

        # x = x + self.beta * g  # 当beta为正数的时候虽然也有提升，但不大
        x = x + (-0.01) * g
        if self.transform is not None:
            x = self.transform(x)
        return x
    