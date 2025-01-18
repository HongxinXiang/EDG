# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import ImageED.models_mae_multi_view_RGBD as models_mae
import ImageED.util.misc as misc
from ImageED.engine_pretrain import train_one_epoch
from ImageED.util.misc import NativeScalerWithGradNormCount as NativeScaler


class ImageDataset(Dataset):
    def __init__(self, filepaths, labels, data_index_list=None, transform=None, target_transform=None, ret_index=False):
        assert len(filepaths) == len(labels)
        self.samples = list(zip(filepaths, labels))
        self.data_index_list = data_index_list
        self.total = len(filepaths)
        self.transform = transform
        self.transform_Depth = transforms.Compose([transforms.ToTensor()])
        self.target_transform = target_transform
        self.ret_index = ret_index

    def __getitem__(self, index):
        item_path, target = self.samples[index]
        if isinstance(item_path, str):  # 一个图片
            RGBD = np.array(Image.open(item_path).convert('RGBA'))
            RGB = Image.fromarray(RGBD[:, :, :3])
            Depth = Image.fromarray(RGBD[:, :, 3])
            if self.transform is not None:
                RGB = self.transform(RGB)
            if self.transform_Depth is not None:
                Depth = self.transform_Depth(Depth)
            if self.target_transform is not None:
                target = self.target_transform(target)
            sample = torch.vstack([RGB, Depth])
        elif isinstance(item_path, list):  # 多个视角
            sample_list = []
            for path in item_path:
                RGBD = np.array(Image.open(path).convert('RGBA'))
                RGB = Image.fromarray(RGBD[:, :, :3])
                Depth = Image.fromarray(RGBD[:, :, 3])
                if self.transform is not None:
                    RGB = self.transform(RGB)
                if self.transform_Depth is not None:
                    Depth = self.transform_Depth(Depth)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                single_sample = torch.vstack([RGB, Depth])
                sample_list.append(single_sample)
            sample = torch.stack(sample_list)
        else:
            raise NotImplementedError

        if self.ret_index:
            return self.data_index_list[index], sample, target
        else:
            return sample, target

    def __len__(self):
        return self.total


def get_6_view_image_path_list(data_root, ED_image_root="", flatten=False):
    protein_name_list = os.listdir(data_root)
    ED_image_path_list = [[f"{data_root}/{protein_name}/{ED_image_root}/x_0.png", f"{data_root}/{protein_name}/{ED_image_root}/x_90.png",
                        f"{data_root}/{protein_name}/{ED_image_root}/x_-90.png",
                        f"{data_root}/{protein_name}/{ED_image_root}/x_180.png", f"{data_root}/{protein_name}/{ED_image_root}/y_90.png",
                        f"{data_root}/{protein_name}/{ED_image_root}/y_-90.png"]
                       for protein_name in protein_name_list]
    if flatten:
        image_path_list = np.array(ED_image_path_list).flatten().tolist()
    else:
        image_path_list = ED_image_path_list
    return image_path_list


def get_args_parser():
    parser = argparse.ArgumentParser('ImageED pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--mask_loss_weight', type=float, default=1, help='')
    parser.add_argument('--recovery_loss_weight', type=float, default=1, help='')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # In order to match the depth map and RGB image, do not use any image augmentation
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_path_list = get_6_view_image_path_list(args.data_path, ED_image_root="density")
    labels = [-1] * len(image_path_list)
    dataset_train = ImageDataset(image_path_list, labels, transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae.__dict__[args.model](in_chans=4, norm_pix_loss=args.norm_pix_loss,
                                            mask_loss_weight=args.mask_loss_weight,
                                            recovery_loss_weight=args.recovery_loss_weight)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write('job dir: {}\n'.format(os.path.dirname(os.path.realpath(__file__))))
        f.write("{}".format(args).replace(', ', ',\n'))
        f.write("{}\n".format("base lr: %.2e" % (args.lr * 256 / eff_batch_size)))
        f.write("{}\n".format("actual lr: %.2e" % args.lr))
        f.write("{}\n".format("Model = %s" % str(model_without_ddp)))
        f.write("{}\n".format("accumulate grad iterations: %d" % args.accum_iter))
        f.write("{}\n".format("effective batch size: %d" % eff_batch_size))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
