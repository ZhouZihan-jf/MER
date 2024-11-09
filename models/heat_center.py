import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatCenter(nn.Module):
    def __init__(self, sigma=1.0):
        super(HeatCenter, self).__init__()
        self.sigma = sigma
        
    def gaussian_heat(self, feature_map, sigma=1.0):
        # 对每个通道进行求和
        sumed_feature_map = torch.sum(feature_map, dim=1, keepdim=True)
        # 找到平均特征图中的最大值位置
        max_indices = torch.argmax(sumed_feature_map.view(sumed_feature_map.size(0), -1), dim=1)
        
        height, width = feature_map.size(2), feature_map.size(3)
        max_row_indices = max_indices // width
        max_col_indices = max_indices % width
        
        x0 = max_col_indices.view(-1, 1, 1).float()
        y0 = max_row_indices.view(-1, 1, 1).float()

        # 获取特征图的高度和宽度
        x_axis = torch.arange(0, width, 1, dtype=torch.float32, device=feature_map.device)
        y_axis = torch.arange(0, height, 1, dtype=torch.float32, device=feature_map.device)
        y, x = torch.meshgrid(y_axis, x_axis)
        
        # 归一化坐标范围至 [0, 1]
        x_normalized = x / (width - 1)
        y_normalized = y / (height - 1)
        x0_normalized = x0 / (width - 1)
        y0_normalized = y0 / (height - 1)

        # 计算高斯热力图
        gaussian = torch.exp(-((x_normalized - x0_normalized) ** 2 +
                            (y_normalized - y0_normalized) ** 2) / (2 * sigma ** 2))
        gaussian = gaussian.unsqueeze(1)  # 添加 channel 维度

        return gaussian
        
    def forward(self, feature):
        b, c, h, w = feature.size()
        
        # 使用高斯模糊层对求和后的特征图进行处理
        gaussian = self.gaussian_heat(feature, self.sigma)
        
        position = []
        for i in range(b):        
            max_position = torch.argmax(gaussian[i]).item()
            x = max_position % w
            y = max_position // w
            point = [x, y]
            position.append(point)
            
        return position
    
    
if __name__ == '__main__':
    tensor = torch.randn(4, 64, 120, 228)
    
    heat = HeatCenter(sigma=1.0)
    
    print(heat(tensor))
