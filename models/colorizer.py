import torch
import torch.nn as nn
import torch.nn.functional as F
from .target_and_multi_sampler import CorrelationCalculator
from spatial_correlation_sampler import SpatialCorrelationSampler
from .correlation import *


def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target


def make_gaussian(size, sigma=1, center=None):
    """ Make a square gaussian kernel.
        size: is the dimensions of the output gaussian
        sigma: is full-width-half-maximum, which
        can be thought of as an effective radius.
    """
    x = torch.arange(start=0, end=size[3], step=1, dtype=float).cuda().repeat(size[1], 1).repeat(size[0], 1, 1)
    y = torch.arange(start=0, end=size[2], step=1, dtype=float).cuda().repeat(size[1], 1).repeat(size[0], 1, 1)
    x0 = center[0]
    y0 = center[1]
    return torch.exp(-4 * torch.log(torch.tensor(2.0)) * (
                (x - x0).unsqueeze(2) ** 2 + (y - y0).unsqueeze(3) ** 2) / sigma ** 2).float()


class Colorizer(nn.Module):
    def __init__(self, R=6, C=32, trianing=False):
        super(Colorizer, self).__init__()
        self.D = 4
        self.R = R  # window size
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0
        
        self.training = trianing
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.corr_calculater = CorrelationCalculator(grid_size=self.R)
        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)
        
    def prep(self, image, h, w):
        _, c, H, W = image.size()
        scale = H // h
        x = image.float()[:, :, ::scale, ::scale]

        if c == 1 and not self.training:
            x = one_hot(x.long(), self.C)
        return x
    
    def prep2(self, image, H, W):
        x = F.interpolate(image.float(), size=(H,W), mode='bilinear', align_corners=True)
        return x
    
    def get_out(self, prep, quantized_r, corr, b, h, w, kernel_size, padding):
        qr = [prep(qr, h, w) for qr in quantized_r]
        image_uf = [F.unfold(r, kernel_size=kernel_size, padding=padding) for r in qr]  # 滑动窗口展开张量
        image_uf = [uf.reshape([b, qr[0].size(1), kernel_size * kernel_size, h * w]) for uf in image_uf]
        image_uf = torch.cat(image_uf, dim=2)  # [1, 7, 841, 27360]
        if not self.training:
            image_uf[:, 1:] *= 1.15
        product = corr * image_uf
        sum_result = product.sum(2)
        result = sum_result.reshape([b, qr[0].size(1), h, w])
        return result
    
    def compact_verify(self, corr, nref, h, w):
        corr_compact = corr[:, 0].reshape(-1, nref, self.P * self.P, h * w).permute(0, 1, 3, 2)
        corr_compact = corr_compact.contiguous().reshape(-1, nref * h * w, self.P * self.P)
        value_axis_, axis = torch.topk(corr_compact.detach(), 2, dim=-1, largest=True, sorted=False)
        value_axis = torch.softmax(value_axis_, dim=-1)
        heats = []
        for ii in range(2):
            axis_ys = axis[:,:,ii:ii+1] // self.P
            axis_xs = axis[:,:,ii:ii+1] % self.P
            heats.append(make_gaussian(size=[axis_ys.size(0),axis_ys.size(1),self.P,self.P], center=[axis_xs, axis_ys]))
        corr_heat = (heats[0].flatten(2)*value_axis[:,:,0:1]+heats[1].flatten(2)*value_axis[:,:,1:2])
        value_axis_, _ = torch.max(value_axis_, dim=-1, keepdim=True) #b, nref*h*w, 1
        
        return corr_compact, corr_heat, value_axis_

    def forward(self, feats_r, feats_t, quantized_r):
        nref = len(feats_r)
        b,c,h,w = feats_t.size()
        
        # 局部采样
        corrs = []
        corrs_small = []
        for ind in range(nref):
            corr = self.correlation_sampler(feats_t, feats_r[ind])
            multi_scale_corr, corr_small = self.corr_calculater(feats_t, feats_r[ind])
            # corr += self.gamma * multi_scale_corr
            corrs.append(corr.reshape([b, self.P * self.P, h * w]))
            corrs_small.append(corr_small.reshape([b, (self.R + 1) * (self.R + 1), h * w]))
            
        corr = torch.cat(corrs, 1)  # b,nref*N,HW
        corr = F.softmax(corr, dim=1)
        corr = corr.unsqueeze(1)
        
        corr_small = torch.cat(corrs_small, 1)  # b,nref*N,HW
        corr_small = F.softmax(corr_small, dim=1)
        corr_small = corr_small.unsqueeze(1)
        
        if self.training == True:
            # 正常采样
            quantized_t = self.get_out(self.prep, quantized_r, corr, b, h, w, self.P, self.R)
            quantized_t_small = self.get_out(self.prep2, quantized_r, corr_small, b, h, w, self.R+1, self.R//2)
            quantized_t = quantized_t + self.beta * quantized_t_small
            return quantized_t, 0
        else:
            # 紧凑性验证
            corr_compact, corr_heat, value_axis_ = self.compact_verify(corr, nref, h, w)
            indic = value_axis_ > 0.1
            corr = corr_compact * (~indic) + corr_heat * indic
            corr = corr.reshape(b, nref, h * w, self.P * self.P).permute(0, 1, 3, 2).contiguous().reshape(b, nref * self.P * self.P, h * w).unsqueeze(1)
            
            quantized_t = self.get_out(self.prep, quantized_r, corr, b, h, w, self.P, self.R)
            # quantized_t_small = self.get_out(self.prep2, quantized_r, corr_small, b, h, w, self.R+1, self.R//2)
            # quantized_t = quantized_t + self.beta * quantized_t_small
            return quantized_t

            
            

