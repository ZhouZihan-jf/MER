import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, Swin_V2_T_Weights


class ResNetModel(nn.Module):
    def __init__(self, weights=ResNet50_Weights.DEFAULT):
        super(ResNetModel, self).__init__()
        # Load a pre-trained ResNet model from torchvision
        self.resnet = models.resnet50(weights=weights)
        self.pre_conv = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        
        self.freeze_weight()
        self.eval()
        
    def freeze_weight(self):
        for params in self.resnet.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.pre_conv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3
    
    
class SwinTransformer(nn.Module):
    def __init__(self, weights=Swin_V2_T_Weights.DEFAULT):
        super(SwinTransformer, self).__init__()
        self.swin = models.swin_v2_t(weights=weights)
        self.features = self.swin.features
        self.norm = self.swin.norm
        self.permute = self.swin.permute
        self.freeze_weight()
        self.eval()
        
    def freeze_weight(self):
        for params in self.swin.parameters():
            params.requires_grad = False
            
    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        return x
        

if __name__ == '__main__':
    # model = ResNetModel(weights=ResNet50_Weights.DEFAULT)
    model = SwinTransformer(weights=Swin_V2_T_Weights.DEFAULT)
    tensor = torch.randn(1, 3, 480, 910)
    x1 = model(tensor)
    print(x1.shape)
    