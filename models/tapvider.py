import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbone.resnet import resnet18
import math
import collections
from itertools import repeat


def video2images(imgs):
    batches, channels, clip_len = imgs.shape[:3]
    if clip_len == 1:
        new_imgs = imgs.squeeze(2).reshape(batches, channels, *imgs.shape[3:])
    else:
        new_imgs = imgs.transpose(1, 2).contiguous().reshape(
            batches * clip_len, channels, *imgs.shape[3:])

    return new_imgs

def images2video(imgs, clip_len):
    batches, channels = imgs.shape[:2]
    if clip_len == 1:
        new_imgs = imgs.unsqueeze(2)
    else:
        new_imgs = imgs.reshape(batches // clip_len, clip_len, channels,
                                *imgs.shape[2:]).transpose(1, 2).contiguous()

    return new_imgs

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")

def coords_grid(batch, xx, yy):
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1)  # shape(batch, 2, H, W)

def cat(tensors, dim: int = 0):
    """Efficient version of torch.cat that avoids a copy if there is only a
    single element in a list."""
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def spatial_neighbor(batches,
                     height,
                     width,
                     neighbor_range,
                     device,
                     dtype,
                     dim=1,
                     mode='circle'):
    assert dim in [1, 2]
    assert mode in ['circle', 'square']
    if mode == 'square':
        neighbor_range = _pair(neighbor_range)
        mask = torch.zeros(
            batches, height, width, height, width, device=device, dtype=dtype)
        for i in range(height):
            for j in range(width):
                top = max(0, i - neighbor_range[0] // 2)
                left = max(0, j - neighbor_range[1] // 2)
                bottom = min(height, i + neighbor_range[0] // 2 + 1)
                right = min(width, j + neighbor_range[1] // 2 + 1)
                mask[:, top:bottom, left:right, i, j] = 1

        mask = mask.view(batches, height * width, height * width)
        if dim == 2:
            mask = mask.transpose(1, 2).contiguous()
    else:
        radius = neighbor_range // 2
        grid_x, grid_y = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype))
        dist_mat = ((grid_x.view(height, width, 1, 1) -
                     grid_x.view(1, 1, height, width))**2 +
                    (grid_y.view(height, width, 1, 1) -
                     grid_y.view(1, 1, height, width))**2)**0.5
        mask = dist_mat < radius
        mask = mask.view(height * width, height * width)
        mask = mask.to(device=device, dtype=dtype)
    return mask.bool()

def masked_attention_efficient(query,
                               key,
                               value,
                               mask,
                               temperature=0.07,
                               topk=10,
                               normalize=True,
                               step=512,
                               non_mask_len=0,
                               mode='softmax',
                               sim_mode='dot_product'):
    """

    Args:
        query (torch.Tensor): Query tensor, shape (N, C, H, W)
        key (torch.Tensor): Key tensor, shape (N, C, T, H, W)
        value (torch.Tensor): Value tensor, shape (N, C, T, H, W)
        temperature (float): Temperature
        topk (int): Top-k
        normalize (bool): Whether normalize feature
        step (int): Step for computing affinity
        non_mask_len (int): Length of video that do not apply mask
        mode (str): Affinity mode

    Returns:

    """
    assert mode in ['softmax', 'cosine']
    batches = query.size(0)
    assert query.size(0) == key.size(0) == value.size(0)
    assert value.shape[2:] == key.shape[2:], f'{value.shape} {key.shape}'
    if key.ndim == 4:
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)
    assert value.ndim == key.ndim == 5
    clip_len = key.size(2)
    assert 0 <= non_mask_len < clip_len
    # assert query.shape[2:] == key.shape[3:]
    att_channels, query_height, query_width = query.shape[1:]
    key_height, key_width = key.shape[3:]
    C = value.size(1)
    if normalize:
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
    query_vec = query.view(batches, att_channels, query.shape[2:].numel())
    key_vec = key.view(batches, att_channels, key.shape[2:].numel())
    value_vec = value.view(batches, C, value.shape[2:].numel())
    output = torch.zeros(batches, C,
                         query_height * query_width).to(query)
    if step is None:
        step = query_height * query_width
    for ptr in range(0, query_height * query_width, step):
        # [N, TxHxW, step]
        if sim_mode == 'dot_product':
            cur_affinity = torch.einsum('bci,bcj->bij', key_vec,
                                        query_vec[...,
                                                ptr:ptr + step]) / temperature
        elif sim_mode == 'l2-distance':
            a_sq = key_vec.pow(2).sum(1).unsqueeze(2)
            ab = key_vec.transpose(1, 2) @ query_vec[...,ptr:ptr + step]
            cur_affinity = (2*ab-a_sq) / math.sqrt(att_channels)

        if mask is not None:
            if mask.ndim == 2:
                assert mask.shape == (key_height * key_width,
                                      query_height * query_width)
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + step].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))
            else:
                cur_mask = mask.view(1, 1, key_height * key_width,
                                     query_height *
                                     query_width)[..., ptr:ptr + step].expand(
                                         batches, clip_len - non_mask_len, -1,
                                         -1).reshape(batches, -1,
                                                     cur_affinity.size(2))

            if non_mask_len > 0:
                cur_mask = cat([
                    torch.ones(batches, non_mask_len * key_height * key_width,
                               cur_affinity.size(2)).to(cur_mask), cur_mask
                ],
                               dim=1)
            cur_affinity.masked_fill_(~cur_mask.bool(), float('-inf'))
        if topk is not None:
            # [N, topk, step]
            topk_affinity, topk_indices = cur_affinity.topk(k=topk, dim=1)
            # cur_affinity, idx = cur_affinity.sort(descending=True, dim=1)
            # topk_affinity, topk_indices = cur_affinity[:, :topk], idx[:,
            # :topk]
            topk_value = value_vec.transpose(0, 1).reshape(
                C, -1).index_select(
                    dim=1, index=topk_indices.reshape(-1))
            # [N, C, topk, step]
            topk_value = topk_value.reshape(C,
                                            *topk_indices.shape).transpose(
                                                           0, 1)

            if mode == 'softmax':
                topk_affinity = topk_affinity.softmax(dim=1)
            elif mode == 'cosine':
                topk_affinity = topk_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bcks,bks->bcs', topk_value,
                                      topk_affinity)
        else:
            if mode == 'softmax':
                cur_affinity = cur_affinity.softmax(dim=1)
            elif mode == 'cosine':
                cur_affinity = cur_affinity.clamp(min=0)**2
            else:
                raise ValueError
            cur_output = torch.einsum('bck,bks->bcs', value_vec, cur_affinity)
        output[..., ptr:ptr + step] = cur_output

    output = output.reshape(batches, C, query_height,
                            query_width)

    return output




class TapTracker(nn.Module):
    def __init__(self, args):
        super(TapTracker, self).__init__()
        self.args = args
        self.feature_extraction = resnet18()
        self.post_convolution = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        
    def img2coord(self, imgs, num_poses, topk=5):
            
        clip_len = len(imgs)
        height, width = imgs.shape[2:]
        assert imgs.shape[:2] == (clip_len, num_poses)
        coords = np.zeros((2, num_poses, clip_len), dtype=float)
        imgs = imgs.reshape(clip_len, num_poses, -1)
        assert imgs.shape[-1] == height * width
        # [clip_len, NUM_KEYPOINTS, topk]
        topk_indices = np.argsort(imgs, axis=-1)[..., -topk:]
        topk_values = np.take_along_axis(imgs, topk_indices, axis=-1)
        topk_values = topk_values / (np.sum(topk_values, keepdims=True, axis=-1)+1e-9)
        topk_x = topk_indices % width
        topk_y = topk_indices // width
        # [clip_len, NUM_KEYPOINTS]
        coords[0] = np.sum(topk_x * topk_values, axis=-1).T
        coords[1] = np.sum(topk_y * topk_values, axis=-1).T
        coords[:, np.sum(imgs.transpose(1, 0, 2), axis=-1) == 0] = -1 

        return coords
    
    def get_coords_grid(self, shape, feat_shape):
        xx = torch.arange(0, shape[-1]).cuda()
        yy = torch.arange(0, shape[1]).cuda()
        grid = coords_grid(shape[0], xx, yy)
        stride = shape[1] // feat_shape[0] 
        
        # B x 2 x H x W
        grid_ = grid[:,:,::stride,::stride]
        
        return grid, grid_, stride
    
    def draw_gaussion_map_online(self, coord, grid, sigma=6, stride=8):
            
        B, _, H, W = grid.shape
        # B x P x 2 x H x W
        grid = grid.unsqueeze(1).repeat(coord.shape[0] // B, coord.shape[1], 1, 1, 1)
        coord = coord[:,:,:,None,None]
        
        # B x P x H x W
        g = torch.exp(-((grid[:,:,0,:,:] - coord[:,:,0])**2 + (grid[:,:,1,:,:] - coord[:,:,1])**2) / (2 * sigma**2))
        
        # a = g[0,0].detach().cpu().numpy()
        # a2 = g[0,3].detach().cpu().numpy()
        
        
        resize_g = g[:,:,::stride,::stride]
        # a3 = g[0,0].detach().cpu().numpy()
 
        return g.to(torch.float32), resize_g.to(torch.float32)
    
    def get_feats(self, imgs, num_feats):
        assert imgs.shape[0] == 1
        batch_step = 5
        feat_bank = [[] for _ in range(num_feats)]
        clip_len = imgs.size(2)
        imgs = video2images(imgs)
        for batch_ptr in range(0, clip_len, batch_step):
            feats = self.feature_extraction(imgs[batch_ptr:batch_ptr +
                                                batch_step])
            feats = self.post_convolution(feats)
            if isinstance(feats, tuple):
                assert len(feats) == len(feat_bank)
                for i in range(len(feats)):
                    feat_bank[i].append(feats[i].cpu())
            else:
                feat_bank[0].append(feats.cpu())
        for i in range(num_feats):
            feat_bank[i] = images2video(
                torch.cat(feat_bank[i], dim=0), clip_len)
            assert feat_bank[i].size(2) == clip_len

        return feat_bank
    
    def forward(self, rgbs, query_points, trajectories, visibilities,
                    save_image=False,
                    save_path=None,
                    iteration=None):
        imgs = rgbs.transpose(1,2)
        
        B,  C, clip_len, h, w = imgs.shape
        
        clip_len = imgs.size(2)
        
        # get target shape
        dummy_feat = self.post_convolution(self.feature_extraction(imgs[0:1, :, 0]))
        feat_shape = dummy_feat.shape

        feat_bank = self.get_feats(imgs, len(dummy_feat))[0]
        
        
        grid, grid_, stride = self.get_coords_grid((B,h,w), feat_shape[-2:])
        ref_seg_map, resized_seg_map = self.draw_gaussion_map_online(query_points[:,:,1:], grid, stride=stride)
        
        
        C = resized_seg_map.shape[1]    
            
        seg_bank = []

        seg_preds = [ref_seg_map.detach()]
        spatial_neighbor_mask = spatial_neighbor(
            feat_shape[0],
            *feat_shape[2:],
            neighbor_range=30,
            device=imgs.device,
            dtype=imgs.dtype,)


        seg_bank.append(resized_seg_map.detach())
        for frame_idx in range(1, clip_len):
            key_start = max(0, frame_idx - 5)
            query_feat = feat_bank[:, :, frame_idx].to(imgs.device)
            key_feat = feat_bank[:, :, key_start:frame_idx].to(imgs.device)
            value_logits = torch.stack(seg_bank[key_start:frame_idx], dim=2).to(imgs.device)
            key_feat = torch.cat([feat_bank[:, :, 0:1].to(imgs.device), key_feat], dim=2)
            value_logits = cat([seg_bank[0].unsqueeze(2).to(imgs.device), value_logits], dim=2)
            
            seg_logit = masked_attention_efficient(
                query_feat,
                key_feat,
                value_logits,
                spatial_neighbor_mask,)

            
            seg_bank.append(seg_logit)

            seg_pred = F.interpolate(
                seg_logit,
                size=(h,w),
                mode='bilinear',
                align_corners=False)
            
            seg_preds.append(seg_pred.detach())

        seg_preds = torch.stack(seg_preds, 1).cpu().numpy()

        trajectories_pred = self.img2coord(seg_preds[0], num_poses=C)
        trajectories_pred = torch.from_numpy(trajectories_pred).permute(2,1,0).unsqueeze(0).cuda()
        
        visibilities_pred = torch.zeros_like(visibilities).cuda()
            
            
        return trajectories, visibilities, trajectories_pred, visibilities_pred, query_points