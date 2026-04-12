#coding=UTF-8
import sys
import os
import glob
from argparse import ArgumentParser
import sys
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.public_utils import fix_train_random_seed, cal_torch_model_params
from model.frame.frame_model import Predictor
from dataloader.data_utils import simple_transforms_for_train_1, simple_transforms_no_aug
from dataloader.dataset import PretrainQCDataset
import numpy as np
from pathlib import Path
from utils.logger import Logger
from pretrain.pretrain_utils import train_one_epoch, evaluate
from loss.losses import SupConLoss
from model.dual_teacherl_utils import write_result_dict_to_tb, save_checkpoint, load_checkpoint
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.base.predictor import FramePredictor


def parse_args():
    # Parse arguments
    parser = ArgumentParser(description='PyTorch Implementation of IEMv2')

    # basic
    parser.add_argument('--dataroot', type=str, default="/data1/xianghongxin/IEMv2/pre-training", help='data root, e.g. /data1/xianghongxin/datasets/IEMv2/pre-training')
    parser.add_argument('--dataset', type=str, default="200w-224x224", help='dataset name, e.g. 200w')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')

    # ddp
    parser.add_argument("--ngpus", default=8, type=int, help="number of nodes for distributed training")

    # model params
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')

    # knowledge-related
    parser.add_argument('--use_ED', action='store_true', default=False, help='whether use electronic density')
    parser.add_argument('--ED_path', type=str, default="./1000_density_feats.pkl", help='filename')
    parser.add_argument('--use_EK', action='store_true', default=False, help='electronics-related knowledge')
    parser.add_argument('--EK_path', type=str, default="./quantum_knowledge_01normalized.csv", help='filename')
    parser.add_argument('--use_evaluator', action='store_true', default=False, help='whether use evaluator')

    # optimizer
    parser.add_argument("--warmup_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--weighted_loss', action='store_true', help='add regularization for multi-task loss')

    # train
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42) to split dataset')
    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument('--base_temperature', type=float, default=0.1, help="temperature required by contrastive loss")
    parser.add_argument("--resume", type=str, default=None, help='Resume training from a path of checkpoint')
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split")
    parser.add_argument('--n_ckpt_save', default=1, type=int, help='save a checkpoint every n epochs, n_ckpt_save=0: no save')
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

    assert args.use_ED or args.use_EK, "At least one of ED or EK is true"

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
    train_transforms = simple_transforms_for_train_1(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transforms = simple_transforms_no_aug(resize=args.imageSize, mean_std=(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    # Load dataset
    print_only_rank0("load dataset", logger=logger)

    train_dataset = PretrainQCDataset(args.dataroot, args.dataset, use_ED=args.use_ED, ED_path=args.ED_path,
                                      use_EK=args.use_EK, EK_path=args.EK_path, split="train",
                                      transforms=train_transforms, ret_index=True, idx_frame_list=[19, 9, 29, 49],  # 对应的是 x_0, x_180, y_180, z_180
                                      logger=logger)
    valid_dataset = PretrainQCDataset(args.dataroot, args.dataset, use_ED=args.use_ED, ED_path=args.ED_path,
                                      use_EK=args.use_EK, EK_path=args.EK_path, split="valid",
                                      transforms=valid_transforms, ret_index=True, idx_frame_list=[19, 9, 29, 49],
                                      logger=logger)

    # initialize data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    ########################## load model
    # Load model
    video_teacher = FramePredictor(model_name=args.model_name, head_arch="none", num_tasks=None, pretrained=True)
    num_features = video_teacher.in_features
    
    EDPredictor = Predictor(in_features=num_features, out_features=768)
    EKPredictor = Predictor(in_features=num_features, out_features=22)
    EDEvaluator = Predictor(in_features=num_features, out_features=1)
    EKEvaluator = Predictor(in_features=num_features, out_features=22)

    video_model_params_num = cal_torch_model_params(video_teacher, unit="M")
    EDPredictor_params_num = cal_torch_model_params(EDPredictor, unit="M")
    EKPredictor_params_num = cal_torch_model_params(EKPredictor, unit="M")
    EDEvaluator_params_num = cal_torch_model_params(EDEvaluator, unit="M")
    EKEvaluator_params_num = cal_torch_model_params(EKEvaluator, unit="M")

    print_only_rank0("video teacher: {}".format(video_model_params_num), logger=logger)
    print_only_rank0("EDPredictor: {}".format(EDPredictor_params_num), logger=logger)
    print_only_rank0("EKPredictor: {}".format(EKPredictor_params_num), logger=logger)
    print_only_rank0("EDEvaluator: {}".format(EDEvaluator_params_num), logger=logger)
    print_only_rank0("EKEvaluator: {}".format(EKEvaluator_params_num), logger=logger)

    # Loss and optimizer
    optim_params = [{"params": video_teacher.parameters()},
                    {"params": EDPredictor.parameters()},
                    {"params": EKPredictor.parameters()},
                    {"params": EDEvaluator.parameters()},
                    {"params": EKEvaluator.parameters()}]
    optimizer = SGD(optim_params, momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    criterionCL = SupConLoss(temperature=args.temperature, base_temperature=args.base_temperature, contrast_mode='all')  # mutual information
    criterionL1 = nn.L1Loss()
    criterionL1_none = nn.L1Loss(reduction="none")

    # lr scheduler
    lr_scheduler = None  # 每个视频60次，至少要60个epoch再降学习率，所以一开始训练先不要学习率的调度
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4, last_epoch=-1, verbose=True)

    # Resume weights
    if args.resume is not None:
        flag, resume_desc = load_checkpoint(args.resume, video_teacher, EDPredictor, EKPredictor, EDEvaluator,
                                            EKEvaluator, optimizer=None, lr_scheduler=None, logger=logger)
        args.start_epoch = int(resume_desc['epoch'])
        assert flag, "error in loading pretrained model {}.".format(args.resume)
        print_only_rank0("[resume description] {}".format(resume_desc), logger=logger)

    # Tensorboard
    tb_writer = SummaryWriter(log_dir=args.tb_dir)
    tb_step_writer = SummaryWriter(log_dir=args.tb_step_dir)

    video_teacher = video_teacher.to(device)
    EDPredictor = EDPredictor.to(device)
    EKPredictor = EKPredictor.to(device)
    EDEvaluator = EDEvaluator.to(device)
    EKEvaluator = EKEvaluator.to(device)

    ########################## train
    best_loss = np.Inf
    for epoch in range(args.start_epoch, args.epochs):
        train_dict = train_one_epoch(video_teacher,
                                     EDPredictor=EDPredictor, EKPredictor=EKPredictor, EDEvaluator=EDEvaluator, EKEvaluator=EKEvaluator,
                                     optimizer=optimizer, data_loader=train_loader,
                                     criterionCL=criterionCL, criterionReg=(criterionL1, criterionL1_none),
                                     device=device, epoch=epoch, lr_scheduler=lr_scheduler, tb_writer=tb_step_writer,
                                     args=args, weighted_loss=args.weighted_loss, logger=logger, is_ddp=True)

        print_only_rank0(str(train_dict), logger=logger)

        evaluate_valid_results = evaluate(video_teacher, EDPredictor, EKPredictor,
                                          EDEvaluator, EKEvaluator, valid_loader, criterionCL,
                                          (criterionL1, criterionL1_none), device, epoch, args=args, logger=logger,
                                          is_ddp=True)
        print_only_rank0("[valid evaluation] epoch: {} | {}".format(epoch, evaluate_valid_results), logger=logger)

        # save model
        model_dict = {"video_teacher": video_teacher, "EDPredictor": EDPredictor, "EKPredictor": EKPredictor,
                      "EDEvaluator": EDEvaluator, "EKEvaluator": EKEvaluator}
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
