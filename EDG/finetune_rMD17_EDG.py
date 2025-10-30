import sys

sys.path.append("./Geom3D")  # add Geom3D to envs
import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from config_distillation import args

from Geom3D.datasets import MoleculeDataset3DRadius
from Geom3D.datasets.dataset_rMD17_distillation import DatasetrMD17
from Geom3D.models import SchNet, EGNN, SphereNet, EquiformerEnergyForce

from EDG.distillation_utils import *
from EDG.logger import Logger
from EDG.data_utils import DualCollater

from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from torch.autograd import grad


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
            energy_and_force=True,
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
        model = EquiformerEnergyForce(
            irreps_in=args.Equiformer_irreps_in,
            max_radius=args.Equiformer_radius,
            node_class=node_class,
            number_of_basis=args.Equiformer_num_basis,
            irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
            irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
            fc_neurons=[64, 64], basis_type='exp',
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
        positions = batch.positions
        positions.requires_grad_()
        x = batch.x

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch, extract_representation=True)
        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=positions,
                edge_index=batch.radius_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        pred_force = - \
        grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True,
             retain_graph=True)[0]

        # distillation
        ED_loss = torch.zeros(1).to(device)
        if args.use_ED:
            molecule_3D_repr = alignMapper(molecule_3D_repr)  # 映射到和 image 相同的维度

            ED_graph = EDPredictor(molecule_3D_repr)
            ED_image = EDPredictor(batch.img_feat)
            ED_loss = get_loss(ED_graph, ED_image)

        actual_energy = batch.y
        actual_force = batch.force

        g_energy_loss = args.md17_energy_coeff * criterion(pred_energy, actual_energy)
        g_force_loss = args.md17_force_coeff * criterion(pred_force, actual_force)
        g_loss = g_energy_loss + g_force_loss

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


# @torch.no_grad()  # 不能用这一行，因为计算 force 时需要计算梯度。
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    pred_energy_list, actual_energy_list = [], []
    pred_force_list = torch.Tensor([]).to(device)
    actual_force_list = torch.Tensor([]).to(device)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    for batch in L:
        batch = batch.to(device)
        positions = batch.positions
        positions.requires_grad_()
        x = batch.x

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch, extract_representation=True)
        elif args.model_3d == "EGNN":
            x_one_hot = F.one_hot(x, num_classes=node_class)
            x = preprocess_input(
                x_one_hot,
                x,
                charge_power=args.EGNN_charge_power,
                charge_scale=node_class,
            )
            node_3D_repr = model(
                x=x,
                positions=positions,
                edge_index=batch.radius_edge_index,
                edge_attr=None,
            )
            molecule_3D_repr = global_mean_pool(node_3D_repr, batch.batch)
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        B = pred_energy.size()[0]

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        force = - \
        grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True,
             retain_graph=True)[0].detach_()

        if torch.sum(torch.isnan(force)) != 0:
            mask = torch.isnan(force)
            force = force[~mask].reshape((-1, 3))
            batch.force = batch.force[~mask].reshape((-1, 3))

        pred_energy_list.append(pred_energy.cpu().detach())
        actual_energy_list.append(batch.y.cpu())
        pred_force_list = torch.cat([pred_force_list, force], dim=0)
        actual_force_list = torch.cat([actual_force_list, batch.force], dim=0)

    pred_energy_list = torch.cat(pred_energy_list, dim=0)
    actual_energy_list = torch.cat(actual_energy_list, dim=0)
    energy_mae = torch.mean(torch.abs(pred_energy_list - actual_energy_list)).cpu().item()
    force_mae = torch.mean(torch.abs(pred_force_list - actual_force_list)).cpu().item()

    return energy_mae, force_mae, {
        "y_energy_pred": pred_energy_list,
        "y_energy_true": actual_energy_list,
        "y_force_pred": pred_force_list,
        "y_force_true": actual_force_list
    }


def load_model(model, graph_pred_linear, model_weight_file, load_latest=False, logger=None):
    log = print if logger is None else logger.info
    log("Loading from {}".format(model_weight_file))

    if load_latest:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    elif "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file)
        if "model_3D" in model_weight:
            model.load_state_dict(model_weight["model_3D"])
        else:
            model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

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

    if args.dataset == "rMD17":
        data_root = f"{args.dataroot}/rMD17"
        dataset = DatasetrMD17(data_root, task=args.task, img_feat_path=args.img_feat_path,
                               split_id=args.rMD17_split_id)
        split_idx = dataset.get_idx_split()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    logger.info("train: {} {}".format(len(split_idx["train"]), split_idx["train"][:5]))
    logger.info("valid: {} {}".format(len(split_idx["valid"]), split_idx["valid"][:5]))
    logger.info("test: {} {}".format(len(split_idx["test"]), split_idx["test"][:5]))

    if args.model_3d == "EGNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.EGNN_radius_cutoff}/{args.task}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.EGNN_radius_cutoff
        )
    train_dataset, val_dataset, test_dataset = \
        dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    # Remove energy mean.
    ENERGY_MEAN_TOTAL = 0
    FORCE_MEAN_TOTAL = 0
    NUM_ATOM = None
    for data in train_dataset:
        energy = data.y
        force = data.force
        NUM_ATOM = force.size()[0]
        energy_mean = energy / NUM_ATOM
        ENERGY_MEAN_TOTAL += energy_mean
        force_rms = torch.sqrt(torch.mean(force.square()))
        FORCE_MEAN_TOTAL += force_rms
    ENERGY_MEAN_TOTAL /= len(train_dataset)
    FORCE_MEAN_TOTAL /= len(train_dataset)
    ENERGY_MEAN_TOTAL = ENERGY_MEAN_TOTAL.to(device)
    FORCE_MEAN_TOTAL = FORCE_MEAN_TOTAL.to(device)

    DataLoaderClass = DataLoader
    collate_fn = DualCollater(follow_batch=[], multigpu=False)
    dataloader_kwargs = {"collate_fn": collate_fn}

    train_loader = DataLoaderClass(train_dataset, args.MD17_train_batch_size, shuffle=True,
                                   num_workers=args.num_workers, **dataloader_kwargs)
    val_loader = DataLoaderClass(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 **dataloader_kwargs)
    test_loader = DataLoaderClass(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  **dataloader_kwargs)

    node_class, num_tasks = 119, 1

    # set up model
    model, graph_pred_linear, hidden_dim = model_setup_for_kd()

    # load model
    image_dim = 512
    alignMapper = get_classifier("arch3", hidden_dim, image_dim)  # graph->image
    taskPredictor = get_classifier("arch3", image_dim, num_tasks)  # image->prediction

    # load from pre-trained model
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

    criterion = torch.nn.L1Loss()

    # set up optimizer
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

    train_energy_mae_list, train_force_mae_list = [], []
    val_energy_mae_list, val_force_mae_list = [], []
    test_energy_mae_list, test_force_mae_list = [], []
    best_val_force_mae = 1e10
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer, args)
        logger.info("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_energy_mae, train_force_mae, train_eval_dict = eval(device, train_loader)
            else:
                train_energy_mae = train_force_mae = 0
                train_eval_dict = None
            val_energy_mae, val_force_mae, val_eval_dict = eval(device, val_loader)
            test_energy_mae, test_force_mae, test_eval_dict = eval(device, test_loader)

            train_energy_mae_list.append(train_energy_mae)
            train_force_mae_list.append(train_force_mae)
            val_energy_mae_list.append(val_energy_mae)
            val_force_mae_list.append(val_force_mae)
            test_energy_mae_list.append(test_energy_mae)
            test_force_mae_list.append(test_force_mae)
            logger.info("Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae, val_energy_mae,
                                                                                  test_energy_mae))
            logger.info("Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae, val_force_mae,
                                                                                 test_force_mae))

            if val_force_mae < best_val_force_mae:
                best_val_force_mae = val_force_mae
                best_val_idx = len(train_energy_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)
                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        train_eval_dict=train_eval_dict,
                        val_eval_dict=val_eval_dict,
                        test_eval_dict=test_eval_dict,
                    )
        logger.info("Took\t{}\n".format(time.time() - start_time))

    save_model(save_best=False, logger=logger)

    optimal_test_energy, optimal_test_force = test_energy_mae_list[best_val_idx], test_force_mae_list[best_val_idx]
    logger.info("best Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae_list[best_val_idx],
                                                                               val_energy_mae_list[best_val_idx],
                                                                               optimal_test_energy))
    logger.info("best Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae_list[best_val_idx],
                                                                              val_force_mae_list[best_val_idx],
                                                                              optimal_test_force))
