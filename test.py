import os, time
import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np
from datasets.davis import DAVIS
from torch.utils.data import DataLoader
from metric.f_boundary import db_eval_boundary
from metric.jaccard import db_eval_iou
from datasets.davis_io import imwrite_indexed
from models.nework import Nework
from tools.utils import *
import logger
torch.cuda.set_device(0)

def main(args):
    args.training = False
    args.pad_divisible = 4
    
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
        
    TestDataset = DAVIS(args.filepath)
    TestDataLoader = DataLoader(TestDataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    model = Nework(args).cuda()
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])  # ['state_dict']
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
    
    Fs = AverageMeter()
    Js = AverageMeter()
    
    n_b = len(dataloader)
    log.info("Start testing.")
    
    for b_i, (annotations, images, catname) in enumerate(dataloader):
        fb = AverageMeter()
        jb = AverageMeter()
        
        images = [r.cuda() for r in images]
        annotations = [a.cuda() for a in annotations]
                
        N = len(images)
        outputs = [annotations[0].contiguous()]
    
        for i in range(N - 1):
            ref_index = calulate_ref(args, images, i)  # 采用长短帧
            # print(f"ref_index: {ref_index}")

            rgb_0 = [images[ind] for ind in ref_index]
            rgb_1 = images[i + 1]

            anno_0 = [outputs[ind] for ind in ref_index]
            anno_1 = annotations[i + 1]

            _, _, h, w = anno_0[0].size()

            max_class = int(anno_1.max())
            with torch.no_grad():
                _output = model(rgb_0, rgb_1, anno_0, None)  
                _output = F.interpolate(_output, (h, w), mode='bilinear')
                # _output = dense_crf_torch(rgb_1, _output)
                output = torch.argmax(_output, 1, keepdim=True).float()
                outputs.append(output)


            for classid in range(1, max_class + 1):
                obj_true = (anno_1 == classid).cpu().numpy()[0, 0]
                obj_pred = (output == classid).cpu().numpy()[0, 0]

                f = db_eval_boundary(obj_true, obj_pred)
                j = db_eval_iou(obj_true, obj_pred)

                fb.update(f)
                jb.update(j)
                Fs.update(f)
                Js.update(j)

            ###
            folder = args.savepath
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, catname[0].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad = ((0, 0), (0, 0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, '%s.png' % str(0).zfill(5))
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)
                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img)

            output_file = os.path.join(output_folder, '%s.png' % str(i + 1).zfill(5))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img)

        info = '\t'.join(['Js: ({:.3f}). Fs: ({:.3f}).  current Js: ({:.3f}). Fs: ({:.3f})'
                         .format(Js.avg, Fs.avg, jb.avg, fb.avg)])

        log.info('[{}/{}] {}'.format(b_i, n_b, info))

    return Js.avg
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMR')
    # Data options
    parser.add_argument('--ref', type=int, default=0)
    parser.add_argument('--filepath', help='Data path for Davis', default='/dataset/zzh/DAVIS')
    parser.add_argument('--savepath', type=str, default='/home/zzh/proj/NewWork/results/davis',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='/home/zzh/proj/NewWork/results/train/copy_epoch_9.pt',
                        help='Checkpoint file to resume') 
    args = parser.parse_args()
    
    main(args)