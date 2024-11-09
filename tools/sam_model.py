import torch
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_l"](checkpoint='/home/zzh/proj/NewWork/results/train/sam_vit_l_0b3195.pth')


if __name__ == "__main__":
    image = torch.randn(1, 3, 1024, 1024)
    print(sam.image_encoder(image).shape)

