import argparse
import os
import ipdb
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore", UserWarning)

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

from collections import defaultdict
from utility import mask_iou, Eval_Fmeasure, AverageMeter, MetricLogger, save_single_mask



def make_data_loader(tag='', args=None):
    if args.subset == 'synthetic':
        from datasets.AVSSynthetic_dataloader import SyntheticDataset
        dataset = SyntheticDataset(split=tag, args=args)
    elif args.subset == 'ms3':
        from datasets.avsb_dataloader_vggish_ms3_eval import S4Dataset
        dataset = S4Dataset(split=tag, args=args)
    elif args.subset == 's4':        
        from datasets.avsb_dataloader_vggish import S4Dataset
        dataset = S4Dataset(split=tag, args=args)

    print('{} dataset: size={}'.format(tag, len(dataset)))
        
    loader = DataLoader(dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.n_threads, pin_memory=True)
    return loader



def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_avs_synthetic(loader, model, args):
    model.eval()
    device = model.device

    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch in tqdm(metric_logger.log_every(loader, 10, header)):
        img, spec, mask, category, video_name = batch
        # img: Bx5xCxHxW, spec: Bx5x1xHxW, mask: BxTx1xHxW -> BxTxHxW
        bs, T = img.size()[:2]
        mask = mask.squeeze(dim=2).to(device)
        bs, T, H, W = mask.size()

        all_pred_masks = [] 

        # Assume the test batch_size is 1
        for idx in range(bs):
            img_i = img[idx].to(device)
            spec_i = spec[idx].to(device)
            mask_i = mask[idx].to(device)

            with torch.no_grad():
                mask_pred = model.infer(img_i, spec_i)
            all_pred_masks.append(mask_pred)
        all_pred_masks = torch.stack(all_pred_masks, dim=0)  # BxTxHxW

        gt_masks = mask.reshape(bs*T, H, W)
        pred_masks = all_pred_masks.reshape(bs*T, H, W)
        
        miou = mask_iou(pred_masks, gt_masks)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(pred_masks, gt_masks)
        avg_meter_F.add({'F_score': F_score})

        if args.save_mask:
            assert bs == 1, "Please set batch_size to 1 when saving test masks!"
            save_dir = os.path.join(os.path.dirname(args.eval), "syn_masks")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(pred_masks.size(0)):
                each_mask = pred_masks[i]
                each_mask_name = video_name[0] + '.png'
                save_single_mask(each_mask, os.path.join(save_dir, each_mask_name))

    miou = (avg_meter_miou.pop('miou'))
    F_score = (avg_meter_F.pop('F_score'))
    eval_metrics = {'miou': miou.item(),
                    'F_score': F_score  }
    
    return eval_metrics



@torch.no_grad()
def evaluate_avs_S4(loader, model, args):
    model.eval()
    device = model.device

    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for batch in tqdm(metric_logger.log_every(loader, 10, header)):
        img, spec, mask, category, video_name = batch
        # img: Bx5xCxHxW, spec: Bx5x1xHxW, mask: BxTx1xHxW -> BxTxHxW
        # import ipdb; ipdb.set_trace()
        bs, T = img.size()[:2]
        mask = mask.squeeze(dim=2).to(device)
        bs, T, H, W = mask.size()

        all_pred_masks = [] 

        # Assume the test batch_size is 1
        for idx in range(bs):
            img_i = img[idx].to(device)
            spec_i = spec[idx].to(device)
            mask_i = mask[idx].to(device)

            with torch.no_grad():
                mask_pred = model.infer(img_i, spec_i)
            all_pred_masks.append(mask_pred)
        all_pred_masks = torch.stack(all_pred_masks, dim=0)  # BxTxHxW

        gt_masks = mask.reshape(bs*T, H, W)
        pred_masks = all_pred_masks.reshape(bs*T, H, W)
        
        miou = mask_iou(pred_masks, gt_masks)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(pred_masks, gt_masks)
        avg_meter_F.add({'F_score': F_score})

        if args.save_mask:
            assert bs == 1, "Please set batch_size to 1 when saving test masks!"
            save_dir = os.path.join(os.path.dirname(args.eval), "masks", category[0], video_name[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(pred_masks.size(0)):
                each_mask = pred_masks[i]
                each_mask_name = video_name[0] + '_{}'.format(i+1) + '.png'
                save_single_mask(each_mask, os.path.join(save_dir, each_mask_name))

    miou = (avg_meter_miou.pop('miou'))
    F_score = (avg_meter_F.pop('F_score'))
    eval_metrics = {'miou': miou.item(),
                    'F_score': F_score
                    }
    
    return eval_metrics





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam_avs_adapter.yaml")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument("--n_threads", type=int, default=8, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--inp_size", type=int, default=1024, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument('--trainset', type=str, default="AVSBench")
    parser.add_argument('--dir_prefix', type=str, default="/", help="")
    parser.add_argument('--eval', type=str, default="", help="The path to trained model weights")
    parser.add_argument('--save_mask', default=False, action='store_true', help='Save mask in the test stage')
    parser.add_argument('--save_res_by_cat', default=False, action='store_true', help='Save test results by category')
    parser.add_argument('--subset', type=str, default="s4", help="which subset of avsbench: s4 | ms3 | synthetic")
    parser.add_argument('--openset', default=False, action='store_true', help='Open set traing and evaluation of S4 subset ')
    parser.add_argument('--infer', default=False, action='store_true', help='Open set traing and evaluation of S4 subset ')
    args = parser.parse_args()

    if args.eval:
        # args.config = os.path.join(os.path.dirname(args.eval), 'config.yaml')
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            args.config = config
    
    
    spec = config['test_dataset']
    loader = make_data_loader(tag='test', args=args)

    model = models.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.eval, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    
    if args.subset == "synthetic":
        results = evaluate_avs_synthetic(loader, model, args)
        with open(os.path.join(os.path.dirname(args.eval), 'test_res_synthetic.txt'),'w') as f:
            f.writelines("mIoU: " + str(results['miou']) + '\n')
            f.writelines("F-score: " + str(results['F_score']) + '\n')
    else:
        results = evaluate_avs_S4(loader, model, args)
        with open(os.path.join(os.path.dirname(args.eval), 'test_res.txt'),'w') as f:
            f.writelines("mIoU: " + str(results['miou']) + '\n')
            f.writelines("F-score: " + str(results['F_score']) + '\n')

    print('Metric-MIoU: {:.4f}'.format(results['miou']))
    print('Metric-FScore: {:.4f}'.format(results['F_score']))
   
