import os
import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from .davis_io import *


M = 1


class DAVIS(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.annos, self.imgs, self.catnames = self.get_data()
        
    def get_data(self):
        global catnames
        catname_txt = self.filepath + '/ImageSets/2017/val.txt'
        
        catnames = open(catname_txt).readlines()
        
        annotation_all = []
        jpg_all = []
        
        for catname in catnames:
            anno_path = os.path.join(self.filepath, 'Annotations/480p/' + catname.strip())
            cat_annos = [os.path.join(anno_path, file) for file in sorted(os.listdir(anno_path))]
            annotation_all.append(cat_annos)
            
            jpg_path = os.path.join(self.filepath, 'JPEGImages/480p/' + catname.strip())
            cat_jpgs = [os.path.join(jpg_path, file) for file in sorted(os.listdir(jpg_path))]
            jpg_all.append(cat_jpgs)
            
        return annotation_all, jpg_all, catnames
    
    def prep(self, path, tag):
        if tag == 'a':
            image, _ = imread_indexed(path)
            
            h, w = image.shape[0], image.shape[1]
            if w % M != 0: image = image[:,:-(w%M)]
            if h % M != 0: image = image[:-(h%M),:]
            image = np.expand_dims(image, 0)

            return torch.Tensor(image).contiguous().long() 
        elif tag == 'i':
            image = cv2.imread(path)
            image = np.float32(image) / 255.0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            image = transforms.ToTensor()(image)
            image = transforms.Normalize([50,0,0], [50,127,127])(image)

            h,w = image.shape[1], image.shape[2]
            if w%M != 0: image = image[:,:,:-(w%M)]
            if h%M != 0: image = image[:,:,-(h%M)]

            return image
        else:
            raise ValueError('Invalid tag')
        
    def __getitem__(self, index):
        annos = self.annos[index]
        imgs = self.imgs[index]
        catname = self.catnames[index]
        
        annotations = [self.prep(anno, 'a') for anno in annos]
        images = [self.prep(img, 'i') for img in imgs]
        
        return annotations, images, catname
    
    def __len__(self):
        return len(self.annos)
    
    
    