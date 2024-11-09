import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class YTVOS(Dataset):
    def __init__(self, datapath, csv_path, transforms=None):
        super().__init__()
        self.datapath = datapath
        self.csv_path = csv_path
        self.transforms = transforms
        self.pos_num = 2
        self.neg_num = self.pos_num - 1
        self.imgs, self.video_list = self.get_data()
        
    def get_data(self):
        file_names = open(self.csv_path).readlines()
        
        video_name = [file_name.split(',')[0].strip() for file_name in file_names]
        start_frame = [int(file_name.split(',')[1].strip()) for file_name in file_names]
        n_frames = [int(file_name.split(',')[2].strip()) for file_name in file_names]
        
        all_index = np.arange(len(n_frames))
        np.random.shuffle(all_index)
        
        refs = []
        for i in range(len(all_index)):
            index1 = all_index[i]
            if(i == len(all_index) - 1):
                index2 = all_index[0]
            else:
                index2 = all_index[i+1]

            frame_interval = np.random.choice([2, 3, 5], p=[0.4, 0.4, 0.2])
            
            refs_imgs = []
            n1 = n_frames[index1]
            n2 = n_frames[index2]
            start1 = start_frame[index1]
            start2 = start_frame[index2]
            frame1_indices = np.arange(start1, start1 + n1, max(frame_interval, 2))
            frame2_indices = np.random.choice(np.arange(start2, start2 + n2), size=self.neg_num)
            
            total_batch, batch_mod = divmod(len(frame1_indices), self.pos_num)  # 取商取余操作
            if batch_mod > 0:
                frame1_indices = frame1_indices[:-batch_mod]
            frame1_indices_batches = np.split(frame1_indices, total_batch)
            
            for batches1 in frame1_indices_batches:
                refs_img2 = [os.path.join(video_name[index2], '{:05d}.jpg'.format(frame))
                        for frame in list(frame2_indices)]
                    
                refs_img1 = [os.path.join(video_name[index1], '{:05d}.jpg'.format(frame))
                            for frame in list(batches1)]
                refs_imgs.append(refs_img2 + refs_img1)
            refs.extend(refs_imgs)
     
        return refs, video_name
    
    def prep(self, path):
        image = cv2.imread(path)
        image = np.float32(image) / 255.0
        image = cv2.resize(image, (256, 256))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([50, 0, 0], [50, 127, 127])(image)
        return image
    
    def __getitem__(self, index):
        img = self.imgs[index]
        negative = img[:self.neg_num]
        img = img[-self.pos_num:]
        
        lab = [self.prep(os.path.join(self.datapath, "JPEGImages", i)) for i in img]
        negative_lab = [self.prep(os.path.join(self.datapath, "JPEGImages", n)) for n in negative]
        refs = self.video_list.index(img[0].split('/')[0])
        
        return lab, negative_lab, refs
    
    def __len__(self):
        return len(self.imgs)
    
    
if __name__ == "__main__":
    dataset = YTVOS("/dataset/zzh/train_all_frames", "/home/zzh/proj/NewWork/datasets/csv/ytvos_mid.csv")
    refs, _ = dataset.get_data()
    print(len(refs))
             