import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = in_channel // squeeze_factor
        self.fc1 = nn.Conv2d(in_channel, squeeze_c, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_c, in_channel, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        
        return x * scale 


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=-1, squeeze_factor=4):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride
        self.se = SqueezeExcitation(out_channel, squeeze_factor)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel=3, width=1,
                 groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.in_channel = 64 * width
        self.base = int(64 * width)
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(in_channel, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 64, 128, 128))
        self.layer1 = self._make_layer(block, self.base * 1, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=1)
        self.layer4 = self._make_layer(block, self.base * 4, layers[3], stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        if block is BasicBlock:
            gamma_name = "bn2.weight"
        else:
            raise RuntimeError(f"block {block} not supported")
        for name, value in self.named_parameters():
            if name.endswith(gamma_name):
                value.data.zero_()
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channel != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_channel, planes, stride, downsample, self.groups, self.base_width, dilation)]
        self.in_channel = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, planes, groups=self.groups, base_width=self.base_width, dilation=dilation))

        return nn.Sequential(*layers)
    
    def forward(self, x, ape=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if ape:
            pos_embed1 = F.interpolate(self.pos_embed1, size=(x.size(2), x.size(3)), mode='bilinear')
            x = x + pos_embed1
            
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        return c5
    
    
def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  


if __name__ == "__main__":
    model = resnet18()
    tensor = torch.randn(1, 3, 256, 256)
    
    c4, c5 = model(tensor)
    
    print(c4.size(), c5.size())