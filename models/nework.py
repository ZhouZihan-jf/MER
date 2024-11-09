import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet18
from .colorizer import Colorizer
        

class Nework(nn.Module):
    def __init__(self, args):
        super(Nework, self).__init__()
        self.C = 7
        self.args = args
        if args.training:
            self.R = 6
            self.freeze = args.freeze
        else:
            self.R = 14
        self.training = args.training
        self.feature_extraction = resnet18()
        self.post_convolution = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.colorizer = Colorizer(self.R, self.C, self.training)
           
    def forward(self, lab_r, lab_t, quantized_r, negatives):
        origin_fr = [self.feature_extraction(lab) for lab in lab_r]
        origin_ft = self.feature_extraction(lab_t)
        
        feats_r = [self.post_convolution(fr).contiguous() for fr in origin_fr]
        feats_t = self.post_convolution(origin_ft).contiguous()
        
        if self.training:
            del origin_fr, origin_ft
            quantized_t, info_nce_loss = self.colorizer(feats_r, feats_t, quantized_r)
            return quantized_t, info_nce_loss
        else:
            del origin_fr, origin_ft
            quantized_t = self.colorizer(feats_r, feats_t, quantized_r)
            return quantized_t
    
    def compute_loss(self, lab, quantized, ch, negatives=None):
        b, c, h, w = lab[0].size()
        
        lab_r = [l for l in lab[:-1]]
        lab_t = lab[-1]
        quantized_r = [q[:, ch] for q in quantized[:-1]]
        result_t = quantized[-1][:,ch]
        
        quantized_t, info_nce_loss = self.forward(lab_r, lab_t, quantized_r, negatives)
        
        output = F.interpolate(quantized_t, (h, w), mode='bilinear')
        loss = F.smooth_l1_loss(output * 30, result_t * 30, reduction='mean')
        
        loss += 0.5 * info_nce_loss
 
        return loss        
        

        