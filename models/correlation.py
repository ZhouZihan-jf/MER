import torch.nn.functional as F
import torch
import torch.nn as nn



def frame_transform(corr, gts, scale=4):
    if isinstance(gts, list):
        gts = torch.cat(gts, 1)
        gts = gts.float()[:, :, ::scale, ::scale]
        gts = gts.flatten(1,3).unsqueeze(1)
    out =  torch.einsum('bij,bcj -> bic', [corr, gts])      
    return out


def non_local_correlation(tar, refs, temprature=1.0, mask=None, scaling=True, norm=False, att_only=False): 
    if isinstance(refs, list):
        refs = torch.stack(refs, 1)
        
    tar = tar.flatten(2).permute(0, 2, 1)
    _, t, feat_dim, w_, h_ = refs.shape
    refs = refs.flatten(3).permute(0, 1, 3, 2)
    if norm:
        tar = F.normalize(tar, dim=-1)
        refs = F.normalize(refs, dim=-1)
        
    # calc correlation
    corr = torch.einsum("bic,btjc -> btij", (tar, refs)) / temprature 
    
    if att_only: return corr
    
    if scaling:
        # scaling
        corr = corr / torch.sqrt(torch.tensor(feat_dim).float()) 

    if mask is not None:
        # att *= mask
        corr.masked_fill_(~mask.bool(), float('-inf'))

    corr_ = corr.permute(0, 2, 1, 3).flatten(2)
    corr_ = F.softmax(corr_, -1)
    return corr_


def get_top_k_offset_region(frames1, frames2, k, sigma=1.0):
    b, c, h, w = frames2.shape
    
    masks = []
    for i in range(b):
        frame1 = frames1[i]
        frame2 = frames2[i]
    
        # Step 1: Calculate the absolute difference between the frames
        offset = torch.abs(frame1 - frame2)
        
        # Step 2: Sum the offsets across the channel dimension
        offset_sum = torch.sum(offset, dim=1)  # shape: [batch_size, height, width]
    
        # Step 3: Flatten the offset_sum to get the top-k indices
        offset_flat = offset_sum.view(-1)
        max_values, max_index = torch.max(offset_flat, dim=0)
        top_k_values, top_k_indices = torch.topk(offset_flat, k)
        
        # Step 4: Convert the flat indices to 2D coordinates
        top_k_coords = [(idx // w, idx % w) for idx in top_k_indices]
        
        # Step 5: Find the bounding box that covers all top-k points
        min_h = min([coord[0] for coord in top_k_coords])
        max_h = max([coord[0] for coord in top_k_coords])
        min_w = min([coord[1] for coord in top_k_coords])
        max_w = max([coord[1] for coord in top_k_coords])
        
        # top_left = (min_h.item(), min_w.item())
        # bottom_right = (max_h.item(), max_w.item())
    
        max_h = max_index // w
        max_w = max_index % w
        max_coord = (max_h.item(), max_w.item())
    
        mask = torch.zeros((h, w))
        # x = torch.arange(min_w, max_w + 1)
        # y = torch.arange(min_h, max_h + 1)
        # y, x = torch.meshgrid(y, x)
    
        # gauss = torch.exp(-((x - max_coord[1])**2 + (y - max_coord[0])**2) / (2 * sigma**2))
        mask[min_h:max_h + 1, min_w:max_w + 1] = frame2[:, min_h:max_h + 1, min_w:max_w + 1].sum(dim=0)

        masks.append(mask.unsqueeze(0))
        
    mask = torch.stack(masks, dim=0).to(frames2.device)
    return mask


if __name__ == '__main__':
    batch_size, channels, height, width = 1, 3, 256, 256
    frame1 = torch.rand((batch_size, channels, height, width))
    frame2 = torch.rand((batch_size, channels, height, width))
    k = 15

    mask = get_top_k_offset_region(frame1, frame2, k)
    print("mask:", mask.size())