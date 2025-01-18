import sys
sys.path.append("./Geom3D")  # add Geom3D to envs
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Geom3D.dataloaders import DataLoaderGemNet
from Geom3D.datasets import MoleculeDataset3DRadius, MoleculeDataset3DFull, MoleculeDatasetOneAtom
from Geom3D.datasets.dataset_QM9_distillation import MoleculeDatasetQM9
from Geom3D.models import SchNet, EGNN, SphereNet, EquiformerEnergy
from EDG.config_distillation import args
from EDG.data_utils import DualCollater
from EDG.distillation_utils import *
from EDG.logger import Logger
from EDG.splitters import qm9_random_customized_01, qm9_random_customized_02
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).float()


def split(dataset, data_root, logger=None):
    log = print if logger is None else logger.info
    if args.split == "customized_01" and ("qm9" in args.dataset or "QM9" in args.dataset):
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_01(
            dataset, null_value=0, seed=args.seed, logger=None
        )
        log("customized random (01) on QM9")
    elif args.split == "customized_02" and ("qm9" in args.dataset or "QM9" in args.dataset):
        train_dataset, valid_dataset, test_dataset = qm9_random_customized_02(
            dataset, null_value=0, seed=args.seed, logger=None
        )
        log("customized random (02) on QM9")
    else:
        raise ValueError("Invalid split option on {}.".format(args.dataset))
    log(f"{len(train_dataset)}\t{len(valid_dataset)}\t{len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset


def model_setup_for_kd():
    if args.model_3d == "SchNet":
        model = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
            node_class=node_class,
        )
        graph_pred_linear = torch.nn.Linear(args.emb_dim, num_tasks)
        hidden_dim = args.emb_dim

    elif args.model_3d == "EGNN":
        in_node_nf = node_class * (1 + args.EGNN_charge_power)
        model = EGNN(
            in_node_nf=in_node_nf,
            in_edge_nf=0,
            hidden_nf=args.emb_dim,
            n_layers=args.EGNN_n_layers,
            positions_weight=args.EGNN_positions_weight,
            attention=args.EGNN_attention,
            node_attr=args.EGNN_node_attr,
        )
        graph_pred_linear = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.SiLU(),
            nn.Linear(args.emb_dim, num_tasks),
        )
        hidden_dim = args.emb_dim

    elif args.model_3d == "SphereNet":
        model = SphereNet(
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            energy_and_force=False,
            cutoff=args.SphereNet_cutoff,
            num_layers=args.SphereNet_num_layers,
            int_emb_size=args.SphereNet_int_emb_size,
            basis_emb_size_dist=args.SphereNet_basis_emb_size_dist,
            basis_emb_size_angle=args.SphereNet_basis_emb_size_angle,
            basis_emb_size_torsion=args.SphereNet_basis_emb_size_torsion,
            out_emb_channels=args.SphereNet_out_emb_channels,
            num_spherical=args.SphereNet_num_spherical,
            num_radial=args.SphereNet_num_radial,
            envelope_exponent=args.SphereNet_envelope_exponent,
            num_before_skip=args.SphereNet_num_before_skip,
            num_after_skip=args.SphereNet_num_after_skip,
            num_output_layers=args.SphereNet_num_output_layers,
        )
        graph_pred_linear = nn.Linear(args.SphereNet_out_emb_channels, num_tasks, bias=False)
        hidden_dim = args.SphereNet_out_emb_channels

    elif args.model_3d == "Equiformer":
        if args.Equiformer_hyperparameter == 0:
            # This follows the hyper in Equiformer_l2
            model = EquiformerEnergy(
                irreps_in=args.Equiformer_irreps_in,
                max_radius=args.Equiformer_radius,
                node_class=node_class,
                number_of_basis=args.Equiformer_num_basis,
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64],
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=False,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        elif args.Equiformer_hyperparameter == 1:
            # This follows the hyper in Equiformer_nonlinear_bessel_l2_drop00
            model = EquiformerEnergy(
                irreps_in=args.Equiformer_irreps_in,
                max_radius=args.Equiformer_radius,
                node_class=node_class,
                number_of_basis=args.Equiformer_num_basis,
                irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
                irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
                fc_neurons=[64, 64], basis_type='bessel',
                irreps_feature='512x0e',
                irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
                rescale_degree=False, nonlinear_message=True,
                irreps_mlp_mid='384x0e+192x1e+96x2e',
                norm_layer='layer',
                alpha_drop=0.0, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0)
        graph_pred_linear = None
        hidden_dim = 512

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model, graph_pred_linear, hidden_dim


def load_model(model, graph_pred_linear, model_weight_file, logger=None):
    log = print if logger is None else logger.info
    log("Loading from {}".format(model_weight_file))
    if "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model_3D"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    elif "JAE" in model_weight_file:
        if "SchNet_02" in args.output_model_dir:
            tag = "3D_model_02"
        else:
            tag = "3D_model_01"
        log("Loading model from {} ...".format(tag))
        model_weight = torch.load(model_weight_file)

        model.load_state_dict(model_weight[tag])
        
    else:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    return


def save_model(save_best, logger=None):
    log = print if logger is None else logger.info
    if not args.output_model_dir == "":
        if save_best:
            log("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            log("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def get_loss(y_graph, y_image):
    assert y_graph.shape[0] == y_image.shape[0]
    loss = F.smooth_l1_loss(y_graph, y_image, reduction='mean')
    return loss


def train(epoch, device, loader, optimizer, args):
    model.train()
    alignMapper.train()
    taskPredictor.train()

    EDPredictor.eval()

    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc_dict = {
        "total_loss": 0,
        "g_loss": 0,
        "ED_loss": 0,
    }

    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader, ncols=180)
    else:
        L = loader

    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                batch.x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch, extract_representation=True)

        elif args.model_3d == "Equiformer":
            molecule_3D_repr, molecule_3D_y_pred = model(node_atom=batch.x, pos=batch.positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_y_pred.squeeze()
        B = pred.size()[0]

        ED_loss = torch.zeros(1).to(device)
        if args.use_ED:
            molecule_3D_repr = alignMapper(molecule_3D_repr)  # 映射到和 image 相同的维度

        if args.use_ED:
            ED_graph = EDPredictor(molecule_3D_repr)
            ED_image = EDPredictor(batch.img_feat)
            ED_loss = get_loss(ED_graph, ED_image)

        y = batch.y.view(B, -1)[:, task_id]
        # normalize
        y = (y - TRAIN_mean) / TRAIN_std
        g_loss = criterion(pred, y)

        loss = g_loss + args.weight_ED * ED_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc_dict = {
            "total_loss": loss_acc_dict["total_loss"] + loss.cpu().detach().item(),
            "g_loss": loss_acc_dict["g_loss"] + g_loss.cpu().detach().item(),
            "ED_loss": loss_acc_dict["ED_loss"] + args.weight_ED * ED_loss.cpu().detach().item(),
        }

        if args.verbose:
            L.desc = (f"[train epoch {epoch}] "
                      f"total_loss: {loss_acc_dict['total_loss'] / (step + 1):.8f}; "
                      f"g_loss: {loss_acc_dict['g_loss'] / (step + 1):.8f}; "
                      f"ED loss: {loss_acc_dict['ED_loss'] / (step + 1):.8f}")

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    for key in loss_acc_dict.keys():
        loss_acc_dict[key] /= len(loader)

    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc_dict["total_loss"])

    return loss_acc_dict


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)

        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(batch.x, num_classes=node_class).float()
            x = preprocess_input(
                x_one_hot,
                batch.x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=batch.positions,
                edge_index=batch.full_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)

        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch, extract_representation=True)

        elif args.model_3d == "Equiformer":
            molecule_3D_repr, molecule_3D_y_pred = model(node_atom=batch.x, pos=batch.positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_y_pred.squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        # denormalize
        pred = pred * TRAIN_std + TRAIN_mean

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    mae = mean_absolute_error(y_scores, y_true)
    return mae, y_true, y_scores


if __name__ == "__main__":
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)

    log_path = f"{args.output_model_dir}/logs.log"
    logMaster = Logger(str(log_path))
    logger = logMaster.get_logger('main')
    logger.info("run command: " + " ".join(sys.argv))
    logger.info("log_dir: {}".format(args.output_model_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rotation_transform = None
    if args.use_rotation_transform:
        rotation_transform = RandomRotation()

    num_tasks = 1
    assert args.dataset == "QM9"
    data_root = f"{args.dataroot}/{args.dataset}"
    dataset = MoleculeDatasetQM9(
        data_root,
        dataset=args.dataset,
        task=args.task,
        rotation_transform=rotation_transform,
        img_feat_path=args.img_feat_path
    )
    task_id = dataset.task_id

    ##### Dataset wrapper for graph with radius. #####
    if args.model_3d == "EGNN":
        data_root = f"{args.dataroot}/{args.dataset}_full"
        dataset = MoleculeDataset3DFull(
            data_root,
            preprcessed_dataset=dataset
        )
    elif args.model_3d == "SEGNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.SEGNN_radius}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.SEGNN_radius
        )
    elif args.model_3d == "PaiNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.PaiNN_radius_cutoff}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.PaiNN_radius_cutoff
        )
    elif args.model_3d in ["NequIP", "Allegro"]:
        data_root = f"{args.dataroot}/{args.dataset}_{args.NequIP_radius_cutoff}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.NequIP_radius_cutoff
        )
    
    if args.only_one_atom_type:
        data_root = "{}_one_atom".format(dataset.root)
        dataset = MoleculeDatasetOneAtom(
            data_root,
            preprcessed_dataset=dataset
        )

    train_dataset, valid_dataset, test_dataset = split(dataset, data_root, logger=logger)
    TRAIN_mean, TRAIN_std = (
        train_dataset.mean()[task_id].item(),
        train_dataset.std()[task_id].item(),
    )
    logger.info("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    DataLoaderClass = DataLoader
    collate_fn = DualCollater(follow_batch=[], multigpu=False)
    dataloader_kwargs = {"collate_fn": collate_fn}
    if args.model_3d == "GemNet":
        DataLoaderClass = DataLoaderGemNet
        dataloader_kwargs = {"cutoff": args.GemNet_cutoff, "int_cutoff": args.GemNet_int_cutoff, "triplets_only": args.GemNet_triplets_only}

    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    # print("================= train")
    # train_idx, val_idx, test_idx = [], [], []
    # for item in train_loader:
    #     train_idx.extend(item)
    # print("================= val")
    # for item in val_loader:
    #     val_idx.extend(item)
    # print("================= test")
    # for item in test_loader:
    #     test_idx.extend(item)

    node_class, edge_class = 119, 5
    model, graph_pred_linear, hidden_dim = model_setup_for_kd()

    # TODO: 没有测试过，需要测试
    # 加载模型
    image_dim = 512
    alignMapper = get_classifier("arch3", hidden_dim, image_dim)  # graph->image
    taskPredictor = get_classifier("arch3", image_dim, num_tasks)  # image->prediction

    # 从预训练模型中加载
    EDPredictor = Predictor(in_features=image_dim, out_features=768)

    if args.pretrained_pth is not None:
        flag, resume_desc = load_checkpoint(args.pretrained_pth, EDPredictor)
        assert flag
        logger.info(resume_desc)

    for my_model in [alignMapper, taskPredictor, EDPredictor]:
        my_model.to(device)

    if args.input_model_file != "":
        load_model(model, graph_pred_linear, args.input_model_file, logger=logger)
    model.to(device)
    logger.info(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    logger.info(graph_pred_linear)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr},
                         {"params": alignMapper.parameters(), "lr": args.lr},
                         {"params": taskPredictor.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        logger.info("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        logger.info("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        logger.info("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        logger.info("Apply lr scheduler ReduceLROnPlateau")
    else:
        logger.info("lr scheduler {} is not included.".format(args.lr_scheduler))

    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_val_mae, best_val_idx = 1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc_dict = train(epoch, device, train_loader, optimizer, args)
        logger.info("Epoch: {}\n{}".format(epoch, loss_acc_dict))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_target, train_pred = eval(device, train_loader)
            else:
                train_mae = 0
            val_mae, val_target, val_pred = eval(device, val_loader)
            test_mae, test_target, test_pred = eval(device, test_loader)

            train_mae_list.append(train_mae)
            val_mae_list.append(val_mae)
            test_mae_list.append(test_mae)
            logger.info(
                "train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
                    train_mae, val_mae, test_mae
                )
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_idx = len(train_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True, logger=logger)

                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        val_target=val_target,
                        val_pred=val_pred,
                        test_target=test_target,
                        test_pred=test_pred,
                    )
        logger.info("Took\t{}\n".format(time.time() - start_time))

    logger.info(
        "best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(
            train_mae_list[best_val_idx],
            val_mae_list[best_val_idx],
            test_mae_list[best_val_idx],
        )
    )

    save_model(save_best=False, logger=logger)
