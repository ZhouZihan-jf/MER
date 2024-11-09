# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial

from mamba_ssm import Mamba
import torch.nn.functional as F 

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=3, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              LayerNorm(dims[0], eps=1e-6)
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            if i == 0 or i == 2:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                )
            else:
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=1, padding=1),
                )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        # self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6)
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            # self.mlps.append(MlpChannel(dims[i_layer], 4 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            # print(f"stage {i} output shape: {x.shape}")
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                # x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x
    
    
    
if __name__ == "__main__":
    model = MambaEncoder(in_chans=3, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512]).cuda()
    tensor = torch.randn(1, 3, 256, 256).cuda()
    output = model(tensor)
    print(output[3].shape)