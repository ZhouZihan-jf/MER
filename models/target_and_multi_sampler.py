import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler
from .heat_center import HeatCenter


class CorrelationCalculator(nn.Module):
    def __init__(self, grid_size=14):
        super(CorrelationCalculator, self).__init__()
        self.grid_size = grid_size
        self.patch = 2 * self.grid_size + 1
        
        self.corr_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.grid_size + 1,
            stride=1,
            padding=0,
            dilation=1
        )
        
        self.corr_sampler_for_crop = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.patch,
            stride=1,
            padding=0,
            dilation=1
        )
        
        self.predictor = HeatCenter()
    
    def get_position(self, feature):  
        position = self.predictor(feature)
        return position
    
    def crop(self, feature, position):
        b, c, h, w = feature.size()
        s = self.patch

        if b != 1:
            crop = []
            for i in range(0, b):
                c0 = torch.zeros((1, c, h, w))
                
                x = int(position[i][0])
                y = int(position[i][1])
                
                x1 = max(0, x - s // 2)  # 左上角 x 坐标
                y1 = max(0, y - s // 2)  # 左上角 y 坐标
                x2 = min(w, x + s // 2)  # 右下角 x 坐标
                y2 = min(h, y + s // 2)  # 右下角 y 坐标
                
                c1 = feature[i, :, y1:y2, x1:x2].unsqueeze(0)
                c0[:, :, y1:y2, x1:x2] = c1  
                crop.append(c0)
                
            crop = torch.cat(crop)
        else:
            c0 = torch.zeros((1, c, h, w))
            
            x = int(position[0][0])
            y = int(position[0][1])

            x1 = max(0, x - s // 2)  # 左上角 x 坐标
            y1 = max(0, y - s // 2)  # 左上角 y 坐标
            x2 = min(w, x + s // 2)  # 右下角 x 坐标
            y2 = min(h, y + s // 2)  # 右下角 y 坐标
            
            c1 = feature[:, :, y1:y2, x1:x2] 
            # c1 = F.interpolate(c1, size=(self.patch, self.patch), mode='bilinear', align_corners=True)        
            c0[:, :, y1:y2, x1:x2] = c1
            crop = c0

        crop = crop.to(feature.device)
        return crop
    
    def get_enhance(self, frame1, frame2):  # frame1是当前帧, frame2是之前帧,对热力图最关键点周围重采样
        _, _, h, w = frame1.size()
        
        position = self.get_position(frame1)
        crop = self.crop(frame1, position)# .contiguous()
        sim_matrix = self.corr_sampler_for_crop(crop, frame2)
        corr = sim_matrix
        
        return corr
        
    def forward(self, frame1, frame2):  # frame1是当前帧, frame2是之前帧, 缩小采样尺度
        b, c, h, w = frame1.size()

        similarity_matrix = self.corr_sampler(frame1, frame2)
        similarity = similarity_matrix
        similarity_matrix = similarity_matrix.permute(0, 3, 4, 1, 2)

        padding = ((self.patch - self.grid_size - 1) // 2, (self.patch - self.grid_size) // 2,
                   (self.patch - self.grid_size - 1) // 2, (self.patch - self.grid_size) // 2)
        corr = F.pad(similarity_matrix, pad=padding, mode='constant').permute(0, 3, 4, 1, 2)  

        return corr, similarity


if __name__ == '__main__':
    f1 = torch.randn(4, 64, 128, 128)
    f2 = torch.randn(4, 64, 128, 128)

    calculator = CorrelationCalculator()

    corr, _ = calculator(f1, f2)
    print(corr.size())
    