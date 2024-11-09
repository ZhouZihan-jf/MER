import argparse
from tools.utils import *
import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from models.nework import Nework
from datasets.youtube import *
import logger
from trainer import Trainer
torch.cuda.set_device(1)

# 设置seed
torch.manual_seed(2024)

def main(args):
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMR')
    # Data options
    parser.add_argument('--datapath', default='/dataset/zzh/YTVOS/',
                        help='Data path for Kinetics')
    parser.add_argument('--validpath',
                        help='Data path for Davis')
    parser.add_argument('--csvpath', default='/home/zzh/proj/NewWork/datasets/csv/ytvos.csv',
                        help='Path for csv file')
    parser.add_argument('--savepath', type=str, default='results/train/',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default="/home/zzh/proj/NewWork/results/train/best_checkpoint.pt",
                        help='resumed checkpoint file')
    # Training options
    parser.add_argument('--training', type=bool, default=True,
                        help='number of epochs to train')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--step_size', type=int, default=15,
                        help='learning rate')
    parser.add_argument('--bsize', type=int, default=32,
                        help='batch size for training (default: 12)')
    parser.add_argument('--freeze', type=bool, default=False, help="冻结参数")
    parser.add_argument('--worker', type=int, default=8,
                        help='number of dataloader threads')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    main(args)
