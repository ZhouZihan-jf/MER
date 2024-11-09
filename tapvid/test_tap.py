import os, time
import argparse
import torch
import torch.nn.parallel
import torch.backends.cudnn
from tqdm import tqdm
from datasets.tapvid import TapVidDAVIS
from torch.utils.data import DataLoader
from models.tapvider import TapTracker
from tools.utils import *
import logger
from tapvid.save_tap import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 根据你的需求设置GPU设备编号


def main(args):
    args.training = False
    args.pad_divisible = 4
    
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
        
    TestDataset = TapVidDAVIS(args.filepath)
    TestDataLoader = DataLoader(TestDataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    model = TapTracker(args).cuda()
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)  # ['state_dict']
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
        
    start_full_time = time.time()
    test(TestDataLoader, model, log)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def test(dataloader, model, log):
    model.eval()
    torch.backends.cudnn.benchmark = True
    log.info("Start testing.")
    outputs = []
    for data in tqdm(dataloader):
        rgbs, query_points, trajectories, visibilities = data
        
        rgbs = rgbs.cuda()
        query_points = query_points.cuda()
        trajectories = trajectories.cuda()
        visibilities = visibilities.cuda()

        with torch.no_grad():
            output = model(rgbs, query_points, trajectories, visibilities)  
            outputs.append(output)
            
        del output

    evaluate(outputs, origin_dir=args.filepath, output_dir=args.savepath)
    
    return 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMR')
    # Data options
    parser.add_argument('--ref', type=int, default=0)
    parser.add_argument('--filepath', help='Data path for Davis', default='/dataset/zzh/tiny_tap/')
    parser.add_argument('--savepath', type=str, default='/home/zzh/proj/NewWork/results/tap',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='/home/zzh/proj/NewWork/results/train/722.pt',
                        help='Checkpoint file to resume') 
    args = parser.parse_args()
    
    main(args)