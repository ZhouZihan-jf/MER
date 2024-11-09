import os.path as osp
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import glob
from .davis_io import *
import torch

M = 1
        

class YTBVAL(Dataset):
    
    def __init__(self, filepath, flow_gap=2):
        self.root = filepath
        self.flow_gap = flow_gap
        self.get_data()

    def get_data(self):
        self.samples = []
        list_path = osp.join(self.root, 'name_part.txt')
        with open(list_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                sample = dict()
                vname = line.strip('\n')
                sample['vname'] = vname
                sample['frames_path'] = sorted(glob.glob(osp.join(self.root, 'JPEGImages', vname, '*.jpg')))
                sample['anno_path'] = sorted(glob.glob(osp.join(self.root, 'Annotations', vname, '*.png')))
                sample['num_frames'] = len(sample['frames_path'])
                
                self.samples.append(sample)
        print(f"Total samples: {len(self.samples)}")
                
    def prep_img(self, path):
        image = cv2.imread(path)
        image = np.float32(image) / 255.0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 
        image = transforms.ToTensor()(image)
        # image = transforms.Resize((256,256))(image)
        image = transforms.Normalize([50,0,0], [50,127,127])(image)

        h,w = image.shape[1], image.shape[2]
        if w%M != 0: image = image[:,:,:-(w%M)]
        if h%M != 0: image = image[:,:,-(h%M)]

        return image
    
    def prep_anno(self, path):
        image, _ = imread_indexed(path)
        h,w = image.shape[0], image.shape[1]
        if w % M != 0: image = image[:,:-(w%M)]
        if h % M != 0: image = image[:-(h%M),:]
        image = np.expand_dims(image, 0)
        
        anno = torch.Tensor(image).contiguous().long()
        # anno = transforms.Resize((256,256))(anno)
        return anno

    def __getitem__(self, index: int):
        sample = self.samples[index]
        num_frames = sample['num_frames']
        anno_path = sample['anno_path']
        frames_path = sample['frames_path']
        vname = sample['vname']
        
        imgs = [self.prep_img(path) for path in frames_path]
        annos = [self.prep_anno(path) for path in anno_path]
        
        return annos, imgs, vname
        
    def __len__(self):
        return len(self.samples)
    
    
if __name__ == "__main__":
    davis = YTBVAL(filepath='/dataset2/zzh/valid')
    anno, imgs, vname = davis[0]
    print(f"anno = {anno[0].shape}, imgs = {imgs[0].shape}")
    print(f"vname = {vname}")