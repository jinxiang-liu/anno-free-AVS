import argparse
import os
import time
import ipdb
import pdb
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.simplefilter("ignore", UserWarning)
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl')
# torch.distributed.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
from collections import defaultdict
from utility import mask_iou, Eval_Fmeasure, AverageMeter, MetricLogger


def make_data_loader(spec, tag='', args=None):
    if args.subset == 'ms3':
        if tag =='train':
            from datasets.avsb_dataloader_vggish_ms3_train import S4Dataset
            dataset = S4Dataset(split=tag, args=args)
        else:
            from datasets.avsb_dataloader_vggish_ms3_eval import S4Dataset
            dataset = S4Dataset(split=tag, args=args)
    elif args.subset == 's4':
        from datasets.avsb_dataloader_vggish import S4Dataset
        dataset = S4Dataset(split=tag, args=args)
    elif args.subset == 'synthetic':
        from datasets.AVSSynthetic_dataloader import SyntheticDataset
        dataset = SyntheticDataset(split=tag, args=args)
    else:
        raise NotImplementedError("To be implemented")
        
    
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=args.n_threads, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders(args=None):
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', args=args)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', args=args)
    return train_loader, val_loader


@torch.no_grad()
def validate(loader, model):
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
        print('n_iter: {}, iou: {}, F_score: {}'.format(idx, miou, F_score))
    
    miou = (avg_meter_miou.pop('miou'))
    F_score = (avg_meter_F.pop('F_score'))
    eval_metrics = {'miou': miou.item(),
                    'F_score': F_score
                    }
    return eval_metrics


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()
    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        img, spec, mask, category, video_name = batch
        model.set_input(img, spec, mask)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    if args.local_rank == 0:
        log, writer = utils.set_save_path(save_path, remove=False)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders(args)
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    if os.path.isfile(args.pretrained_weights):
        ckpt = torch.load(args.pretrained_weights, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt, strict=False)
        print(f"Successfully load pretrained model weights from {args.pretrained_weights}!")

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = 1e-8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()
            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            eval_results = validate(val_loader, model)
            metric1 = 'miou'
            result1 = eval_results[metric1]
            metric2 = 'F_score'
            result2 = eval_results[metric2]
            
            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                
                if result1 > max_val_v:
                    max_val_v = result1
                    save(config, model, save_path, 'best') 

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam_avs_adapter.yaml")
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument("--n_threads", type=int, default=8, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--inp_size", type=int, default=1024, help="")
    parser.add_argument('--dir_prefix', type=str, default="/home/", help="")
    parser.add_argument('--subset', type=str, default="s4", help="which subset of avsbench: s4 | ms3 | synthetic")
    parser.add_argument('--pretrained_weights', type=str, default="", help="Load pretrained weights")
    parser.add_argument('--trainset_shuffle', default=False, action='store_true', help=' ')
    parser.add_argument("--trainset_ratio", type=float, default=1, help="Use the ratio of S4 subset")
    parser.add_argument('--openset', default=False, action='store_true', help='Open set traing and evaluation of S4 subset ')
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')
    
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    save_name = current_time + '_' + args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./ckpts', save_name)
    args.config = config

    main(config, save_path, args=args)
