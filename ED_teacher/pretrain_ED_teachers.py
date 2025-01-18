#coding=UTF-8
import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ED_teacher.dataloader.data_utils import transforms_for_train, transforms_no_aug
from ED_teacher.dataloader.dataset import PretrainQCDataset
from ED_teacher.model.model_utils import write_result_dict_to_tb, save_checkpoint, load_checkpoint
from ED_teacher.model.predictor import Backboneredictor, Predictor
from ED_teacher.pretrain_utils import train_one_epoch, evaluate
from ED_teacher.utils.logger import Logger
from ED_teacher.utils.public_utils import fix_train_random_seed, cal_torch_model_params


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of ED-teacher')

    # basic
    parser.add_argument('--dataroot', type=str, default="./datasets/pre-training", help='data root, e.g. ./datasets/pre-training')
    parser.add_argument('--dataset', type=str, default="image-200w", help='dataset name')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')

    # ddp
    parser.add_argument("--ngpus", default=8, type=int, help="number of nodes for distributed training")

    # model params
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')

    # knowledge-related
    parser.add_argument('--use_ED', action='store_true', default=False, help='whether use electronic density')
    parser.add_argument('--ED_path', type=str, default="./1000_density_feats.pkl", help='filename')

    # optimizer
    parser.add_argument("--warmup_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--weighted_loss', action='store_true', help='add regularization for multi-task loss')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2024, help='random seed to run model (default: 2024)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument('--n_batch_step_optim', default=1, type=int, help='update model parameters every n batches')
    parser.add_argument('--n_sub_checkpoints_each_epoch', type=int, default=4,
                        help='save the sub-checkpoints in an epoch, 0 represent this param is not active. e.g. n=4, will save epoch.2, epoch.4, epoch.6, epoch.8')

    # log
    parser.add_argument('--log_dir', default='./logs/pretrain/', help='path to log')
    parser.add_argument('--tb_step_num', default=100, type=int, help='The training results of every n steps are recorded in tb')

    # Parse arguments
    return parser.parse_args()


def print_only_rank0(text, logger=None):
    log = print if logger is None else logger.info
    log(text)


def is_rank0():
    return True
    # return dist.get_rank() == 0


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def main(args):

    assert args.use_ED, "use_ED must be true"

    device, device_ids = setup_device(args.ngpus)

    # fix random seeds
    fix_train_random_seed(args.runseed)

    # initializing logger
    args.log_dir = args.log_dir / Path(args.dataset) / Path(args.model_name) / Path("seed" + str(args.seed))
    args.tb_dir = os.path.join(args.log_dir, "tb")
    args.tb_step_dir = os.path.join(args.log_dir, "tb_step")
    log_filename = "logs.log"
    log_path = args.log_dir / Path(log_filename)
    try:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    except:
        pass
    logMaster = Logger(str(log_path))
    logger = logMaster.get_logger('main')
    print_only_rank0("run command: " + " ".join(sys.argv), logger=logger)
    print_only_rank0("log_dir: {}".format(args.log_dir), logger=logger)

    ########################## load dataset
    # transforms
    train_transforms = transforms_for_train(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transforms = transforms_no_aug(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset", logger=logger)

    train_dataset = PretrainQCDataset(args.dataroot, args.dataset, use_ED=args.use_ED, ED_path=args.ED_path,
                                      split="train", transforms=train_transforms, ret_index=True, idx_frame_list=[19, 9, 29, 49],  # 对应的是 x_0, x_180, y_180, z_180
                                      logger=logger)
    valid_dataset = PretrainQCDataset(args.dataroot, args.dataset, use_ED=args.use_ED, ED_path=args.ED_path,
                                      split="valid", transforms=valid_transforms, ret_index=True, idx_frame_list=[19, 9, 29, 49],
                                      logger=logger)

    # initialize data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    ED_teacher = Backboneredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    num_features = ED_teacher.in_features
    
    EDPredictor = Predictor(in_features=num_features, out_features=768)

    structural_image_model_params_num = cal_torch_model_params(ED_teacher, unit="M")
    EDPredictor_params_num = cal_torch_model_params(EDPredictor, unit="M")

    print_only_rank0("structural_image teacher: {}".format(structural_image_model_params_num), logger=logger)
    print_only_rank0("EDPredictor: {}".format(EDPredictor_params_num), logger=logger)

    # Loss and optimizer
    optim_params = [{"params": ED_teacher.parameters()},
                    {"params": EDPredictor.parameters()}]
    optimizer = SGD(optim_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    criterionL1 = nn.L1Loss()
    criterionL1_none = nn.L1Loss(reduction="none")

    # lr scheduler
    lr_scheduler = None

    # Resume weights
    if args.resume is not None:
        flag, resume_desc = load_checkpoint(args.resume, ED_teacher, EDPredictor, optimizer=None, lr_scheduler=None, logger=logger)
        args.start_epoch = int(resume_desc['epoch'])
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc), logger=logger)

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=args.tb_dir)
    tb_step_writer = SummaryWriter(log_dir=args.tb_step_dir)

    ED_teacher = ED_teacher.to(device)
    EDPredictor = EDPredictor.to(device)

    ########################## train
    best_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        train_dict = train_one_epoch(ED_teacher,
                                     EDPredictor=EDPredictor,
                                     optimizer=optimizer, data_loader=train_loader,
                                     criterionReg=(criterionL1, criterionL1_none),
                                     device=device, epoch=epoch, lr_scheduler=lr_scheduler, tb_writer=tb_step_writer,
                                     args=args, weighted_loss=args.weighted_loss, logger=logger, is_ddp=True)

        print_only_rank0(str(train_dict), logger=logger)

        evaluate_valid_results = evaluate(ED_teacher, EDPredictor, valid_loader,
                                          (criterionL1, criterionL1_none), device, epoch, args=args, logger=logger,
                                          is_ddp=True)
        print_only_rank0("[valid evaluation] epoch: {} | {}".format(epoch, evaluate_valid_results), logger=logger)

        # save model
        model_dict = {"ED_teacher": ED_teacher, "EDPredictor": EDPredictor}
        optimizer_dict = {"optimizer": optimizer}
        lr_scheduler_dict = {"lr_scheduler": lr_scheduler} if lr_scheduler is not None else None

        cur_loss = train_dict["total_loss"]
        # save best model
        if is_rank0() and best_loss > cur_loss:
            files2remove = glob.glob(os.path.join(args.log_dir, "ckpts", "best_epoch*"))
            for _i in files2remove:
                os.remove(_i)
            best_loss = cur_loss
            best_pre = "best_epoch={}_loss={:.2f}".format(epoch, best_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=best_pre, name_post="", logger=logger)

        if is_rank0() and args.n_ckpt_save > 0 and epoch % args.n_ckpt_save == 0:
            ckpt_pre = "ckpt_epoch={}_loss={:.2f}".format(epoch, cur_loss)
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                            train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                            name_pre=ckpt_pre, name_post="", logger=logger)

        write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict={"optimizer": optimizer})


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许数据副本，提高数据加载的效率
    args = parse_args()
    main(args)
