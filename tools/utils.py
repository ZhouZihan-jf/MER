import numpy as np
import torch
import math
from torchvision.transforms.functional import gaussian_blur
from torch.nn.functional import mse_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
  
        
def adjust_lr_cosine(optimizer, epoch, batch, n_b, args):
    iteration = (batch + epoch * n_b) * args.bsize
    total_iterations = args.epochs * n_b * args.bsize
    
    cosine_decay = 0.5 * (1 + np.cos(np.pi * iteration / total_iterations))
    
    min_lr_ratio = 0.2
    lr = args.lr * cosine_decay * (1 - min_lr_ratio) + min_lr_ratio * args.lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def dropout2d_lab(arr): # drop same layers for all images

    drop_ch_num = int(np.random.choice(np.arange(1, 2), 1))
    drop_ch_ind = np.random.choice(np.arange(1,3), drop_ch_num, replace=False)

    for a in arr:
        for dropout_ch in drop_ch_ind:
            a[:, dropout_ch] = 0
        a *= (3 / (3 - drop_ch_num))

    return arr, drop_ch_ind # return channels not masked


def get_short_list(images, i, mem_gap=1, count=5, sigma=1.0):
    short_list = list(filter(lambda x : x > 0, range(i, i - mem_gap * count, -mem_gap)))[::-1]
    short_score = []
    smoothed_image = gaussian_blur(images[i + 1], kernel_size=23, sigma=sigma)
    for i in short_list:
        mse = mse_loss(gaussian_blur(images[i], kernel_size=23, sigma=sigma), smoothed_image)
        short_score.append(mse)
    combine_data = list(zip(short_list, short_score))
    sorted_data = sorted(combine_data, key=lambda x: x[1])
    short_term = [i[0] for i in sorted_data[:3]]
    return short_term


def calulate_ref(args, images, i, mem_gap=1):
    long_list = [0, 5, 10]
    long_term = list(filter(lambda x : x <= i, long_list))
    
    short_term1 = get_short_list(images, i, mem_gap, count=5)
    short_term2 = get_short_list(images, i, 2, count=5)
    short_term = short_term1 + short_term2
    short_term.append(i)

    if args.ref == 0:
        # ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
        # ref_index = sorted(list(set(ref_index)))
        ref_index = long_term + short_term  # 按间隔mem_gap打印再逆序输出为list
        ref_index = sorted(list(set(ref_index)))
        if len(ref_index) > 6:
            remove_count = len(ref_index) - 6
            del ref_index[3:3 + remove_count]
    elif args.ref == 1:
        ref_index = [0] + short_term
        ref_index = sorted(list(set(ref_index)))
    elif args.ref == 2:
        ref_index = short_term
        ref_index = sorted(list(set(ref_index)))
    else:
         raise NotImplementedError
    # print(f"ref_index = {ref_index}")
    return ref_index



"""
def get_gaussian_pos(image):
    sigma = 1.0
    smoothed_image = gaussian_blur(image, kernel_size=23, sigma=sigma)

    max_pos = torch.argmax(smoothed_image[:, 0, :, :]).item()
    max_pos = divmod(max_pos, smoothed_image.shape[2])
    
    return max_pos
def calculate_distance(point1, point2):
    y1, x1 = point1
    y2, x2 = point2
    
    distance = math.sqrt(float((x2 - x1)**2 + (y2 - y1)**2))
    
    return distance < 10
""" 
    

    