import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from ED_teacher.model.model_utils import save_checkpoint, write_result_dict_to_tb


def train_one_epoch(ED_teacher, EDPredictor,
                    optimizer, data_loader, criterionReg, device, epoch, weighted_loss=False,
                    lr_scheduler=None, tb_writer=None, args=None, logger=None, is_rank0=True, is_ddp=False):
    criterionReg, criterionReg_none = criterionReg
    n_sub_ckpt_list_step = (
                (np.arange(1, args.n_sub_checkpoints_each_epoch + 1) / (args.n_sub_checkpoints_each_epoch + 1)) * len(
            data_loader)).astype(int)

    ED_teacher.train()
    EDPredictor.train()

    accu_loss = 0
    accu_ED_loss = 0

    optimizer.zero_grad()

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, (structural_image, ED_feat, ED_mask, indexs) in enumerate(data_loader):
        n_samples, n_views, n_chanel, h, w = structural_image.shape
        if n_samples <= 1:
            continue
        ED_feat = ED_feat[(ED_mask == 1).flatten(), :]
        structural_image, indexs = structural_image.to(device), indexs.to(device)

        if len(ED_feat) != 0:
            ED_feat = ED_feat.to(device)

        # forward
        feat_structural_image = ED_teacher(structural_image.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
        feat_structural_image_mean = feat_structural_image.mean(1)

        # calculating loss
        loss_dict = {
            "ED_loss": torch.zeros(1).to(device),
        }

        if args.use_ED and len(ED_feat) != 0:
            feat_structural_image_mean_for_ED = feat_structural_image_mean[(ED_mask == 1).flatten(), :]  # ED 可能有缺失，因此这个也需要 align 一下
            bar_ED_feat = EDPredictor(feat_structural_image_mean_for_ED)
            loss_dict["ED_loss"] = criterionReg(bar_ED_feat, ED_feat)

        # backward
        loss = loss_dict["ED_loss"]
        loss.backward()

        # logger
        accu_loss += loss.item()
        accu_ED_loss += loss_dict["ED_loss"].item()

        data_loader.desc = "[train epoch {}] total loss: {:.3f}; ED loss: {:.3f}".format(
            epoch, accu_loss / (step + 1), accu_ED_loss / (step + 1))

        if is_rank0 and logger is not None:
            msg_step = len(data_loader) // 10
            if msg_step == 0:
                msg_step = 1
            if (epoch == 0 and step == 0) or step % msg_step == 0:
                msg = ("=== [train epoch {}, step {}] total loss: {:.3f}; ED loss: {:.3f}").format(
                    epoch, step, accu_loss / (step + 1), accu_ED_loss / (step + 1))
                logger.info(msg)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if step % args.n_batch_step_optim == 0:
            optimizer.step()
            optimizer.zero_grad()

        accu_loss += loss.item()
        accu_ED_loss += loss_dict["ED_loss"].item()

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss / (step + 1),
            "ED_loss": accu_ED_loss / (step + 1),
        }
        if is_rank0 and step != 0 and step in n_sub_ckpt_list_step:
            ckpt_pre = "ckpt_epoch={}[{:.2f}%]_loss={:.2f}".format(epoch, step / len(data_loader) * 100, train_dict["total_loss"])
            model_dict = {"ED_teacher": ED_teacher, "EDPredictor": EDPredictor}

            optimizer_dict = {"optimizer": optimizer}
            lr_scheduler_dict = {"lr_scheduler": lr_scheduler} if lr_scheduler is not None else None
            save_checkpoint(model_dict, optimizer_dict, lr_scheduler_dict,
                             train_dict, epoch, save_path=os.path.join(args.log_dir, "ckpts"),
                             name_pre=ckpt_pre, name_post="", logger=logger)

        if is_rank0 and step in np.arange(0, len(data_loader), args.tb_step_num).tolist():
            if tb_writer is not None:
                write_result_dict_to_tb(tb_writer, train_dict, optimizer_dict={"optimizer": optimizer}, show_epoch=False)

    # Update learning rates
    if lr_scheduler is not None:
        lr_scheduler.step()

    return train_dict


@torch.no_grad()
def evaluate(ED_teacher, EDPredictor,
             data_loader, criterionReg, device, epoch, args=None, logger=None, is_rank0=True, is_ddp=False):
    criterionReg, criterionReg_none = criterionReg

    ED_teacher.eval()
    EDPredictor.eval()
    accu_loss = 0
    accu_ED_loss = 0

    train_dict = {}
    data_loader = tqdm(data_loader, total=len(data_loader))
    for step, (structural_image, ED_feat, ED_mask, indexs) in enumerate(data_loader):
        n_samples, n_views, n_chanel, h, w = structural_image.shape
        if n_samples <= 1:
            continue

        ED_feat = ED_feat[(ED_mask == 1).flatten(), :]
        structural_image, indexs = structural_image.to(device), indexs.to(device)

        if len(ED_feat) != 0:
            ED_feat = ED_feat.to(device)

        with torch.no_grad():
            # forward
            feat_structural_image = ED_teacher(structural_image.reshape(n_samples * n_views, n_chanel, h, w)).reshape(n_samples, n_views, -1)
            feat_structural_image_mean = feat_structural_image.mean(1)

            # calculating loss
            loss_dict = {
                "ED_loss": torch.zeros(1).to(device),
            }
            if args.use_ED and len(ED_feat) != 0:
                feat_structural_image_mean_for_ED = feat_structural_image_mean[(ED_mask == 1).flatten(), :]  # ED 可能有缺失，因此这个也需要 align 一下
                bar_ED_feat = EDPredictor(feat_structural_image_mean_for_ED)
                loss_dict["ED_loss"] = criterionReg(bar_ED_feat, ED_feat)

        # backward
        loss = loss_dict["ED_loss"]

        # logger
        accu_loss += loss.item()
        accu_ED_loss += loss_dict["ED_loss"].item()

        data_loader.desc = "[evaluation epoch {}] total loss: {:.3f}; ED loss: {:.3f}".format(
            epoch, accu_loss / (step + 1), accu_ED_loss / (step + 1))

        train_dict = {
            "step": (step + 1) + len(data_loader) * epoch,
            "epoch": epoch + (step + 1) / len(data_loader),
            "total_loss": accu_loss / (step + 1),
            "ED_loss": accu_ED_loss / (step + 1),
        }

    return train_dict