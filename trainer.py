import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tools.utils import *
from models.nework import Nework
from datasets.youtube import YTVOS
import logger

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup()

    def setup(self):
        if not os.path.isdir(self.args.savepath):
            os.makedirs(self.args.savepath)
        self.log = logger.setup_logger(self.args.savepath + '/training.log')
        for key, value in sorted(vars(self.args).items()):
            self.log.info(str(key) + ': ' + str(value))

        self.model = Nework(self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999),  weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        if self.args.freeze:
            for name, param in self.model.named_parameters():
                if 'gamma' in name or 'beta' in name or 'se' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if self.args.resume:
            self.load_checkpoint()

        self.model = nn.DataParallel(self.model).to(self.device)
        self.best_loss = 1e10

    def load_checkpoint(self):
        if os.path.isfile(self.args.resume):
            self.log.info(f"=> loading checkpoint '{self.args.resume}'")
            checkpoint = torch.load(self.args.resume)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.log.info(f"=> loaded checkpoint '{self.args.resume}'")
        else:
            self.log.info(f"=> No checkpoint found at '{self.args.resume}'")

    def freeze_bn(self, m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    def train_epoch(self, epoch, dataloader):
        self.model.train()
        self.model.apply(self.freeze_bn)
        _loss = AverageMeter()
        n_b = len(dataloader)
        b_s = time.perf_counter()

        for b_i, (lab, negatives, refs) in enumerate(dataloader):
            adjust_lr_cosine(self.optimizer, epoch, b_i, n_b, self.args)

            quantized = [r.clone().to(self.device) for r in lab]
            lab = [r.to(self.device) for r in lab]
            negatives = [n.to(self.device) for n in negatives]

            _, ch = dropout2d_lab(arr=lab)
            loss = self.model.module.compute_loss(lab, quantized, ch, negatives)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _loss.update(loss.item())
            b_t = time.perf_counter() - b_s
            b_s = time.perf_counter()

            lr_now = self.optimizer.param_groups[0]['lr']
            self.log.info(f'Epoch {epoch} [{b_i}/{n_b}] Loss = {_loss.val:.3f}({_loss.avg:.3f}) T={b_t:.2f} LR={lr_now:.6f}')
        
        self.scheduler.step()

        self.save_checkpoint(epoch, _loss.avg)

    def save_checkpoint(self, epoch, avg_loss):
        save_last_path = os.path.join(self.args.savepath, 'last_checkpoint.pt')
        state_dict = self.model.module.state_dict()
        torch.save({
            'state_dict': state_dict,
        }, save_last_path)
        self.log.info("Saved last checkpoint.")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            save_best_path = os.path.join(self.args.savepath, 'best_checkpoint.pt')
            torch.save({
                'state_dict': state_dict,
            }, save_best_path)
            self.log.info("Saved best checkpoint.")

    def train(self):
        start_full_time = time.time()
        self.log.info(f'Number of model parameters: {sum(p.data.nelement() for p in self.model.parameters())}')

        for epoch in range(self.args.epochs):
            train_dataset = YTVOS(self.args.datapath, self.args.csvpath)
            train_sampler = RandomSampler(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=self.args.bsize, num_workers=self.args.worker,
                                      pin_memory=True, sampler=train_sampler, drop_last=True)
            self.log.info(f'This is {epoch}-th epoch')
            self.train_epoch(epoch, train_loader)

        self.log.info(f'Full training time = {(time.time() - start_full_time) / 3600:.2f} Hours')

