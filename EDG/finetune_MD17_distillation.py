import sys
sys.path.append("./Geom3D")  # add Geom3D to envs

import time

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from distillation_utils import *

from config_distillation import args
from Geom3D.datasets import DatasetMD17, MoleculeDataset3DRadius
from Geom3D.datasets.dataset_rMD17_distillation import DatasetrMD17
from Geom3D.dataloaders import DataLoaderGemNet
from Geom3D.models import SchNet, DimeNet, DimeNetPlusPlus, EGNN, SphereNet, SEGNN, PaiNN, GemNet, EquiformerEnergyForce
from Geom3D.models.NequIP.model import model_from_config
from logger import Logger
from data_utils import DualCollater

from torch.autograd import grad


# debug: --verbose --model_3d EGNN --dataroot /data/xianghongxin/datasets/Geom3D --dataset rMD17 --task ethanol --split customized_01 --seed 42 --epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR --no_eval_train --print_every_epoch 1 --img_feat_path /data/xianghongxin/datasets/Geom3D/rMD17/ethanol/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz --num_workers 8 --pretrained_pth /data1/xianghongxin/work/IEMv2/pretrain/experiments/ED_Evaluator_from_10_epoch/video-200w-224x224/resnet18/seed42/ckpts/best_epoch=18_loss=0.30.pth --output_model_dir ./experiments/run_rMD17_distillation/EGNN/e1000_b128_lr5e-4_ed128_lsCosine/ED/ED0.001/rs42/ethanol --use_ED --weight_ED 0.001


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


def model_setup():
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

    elif args.model_3d == "DimeNetPlusPlus":
        model = DimeNetPlusPlus(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            num_blocks=args.DimeNetPlusPlus_num_blocks,
            int_emb_size=args.DimeNetPlusPlus_int_emb_size,
            basis_emb_size=args.DimeNetPlusPlus_basis_emb_size,
            out_emb_channels=args.DimeNetPlusPlus_out_emb_channels,
            num_spherical=args.DimeNetPlusPlus_num_spherical,
            num_radial=args.DimeNetPlusPlus_num_radial,
            cutoff=args.DimeNetPlusPlus_cutoff,
            envelope_exponent=args.DimeNetPlusPlus_envelope_exponent,
            num_before_skip=args.DimeNetPlusPlus_num_before_skip,
            num_after_skip=args.DimeNetPlusPlus_num_after_skip,
            num_output_layers=args.DimeNetPlusPlus_num_output_layers,
        )
        graph_pred_linear = None

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

    elif args.model_3d == "SEGNN":
        model = SEGNN(
            node_class,
            num_tasks,
            hidden_features=args.emb_dim,
            N=args.SEGNN_N,
            lmax_h=args.SEGNN_lmax_h,
            lmax_pos=args.SEGNN_lmax_pos,
            norm=args.SEGNN_norm,
            pool=args.SEGNN_pool,
            edge_inference=args.SEGNN_edge_inference
        )
        graph_pred_linear = None

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
        graph_pred_linear = None

    elif args.model_3d == "PaiNN":
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.PaiNN_n_interactions,
            n_rbf=args.PaiNN_n_rbf,
            cutoff=args.PaiNN_radius_cutoff,
            max_z=node_class,
            n_out=num_tasks,
            readout=args.PaiNN_readout,
        )
        graph_pred_linear = model.create_output_layers()

    elif args.model_3d == "GemNet":
        model = GemNet(
            # node_class=93,
            node_class=node_class,
            num_spherical=args.GemNet_num_spherical,
            num_radial=args.GemNet_num_radial,
            num_blocks=args.GemNet_num_blocks,
            emb_size_atom=args.emb_dim,
            emb_size_edge=args.emb_dim,
            emb_size_trip=args.GemNet_emb_size_trip,
            emb_size_quad=args.GemNet_emb_size_quad,
            emb_size_rbf=args.GemNet_emb_size_rbf,
            emb_size_cbf=args.GemNet_emb_size_cbf,
            emb_size_sbf=args.GemNet_emb_size_sbf,
            emb_size_bil_quad=args.GemNet_emb_size_bil_quad,
            emb_size_bil_trip=args.GemNet_emb_size_bil_trip,
            num_before_skip=args.GemNet_num_before_skip,
            num_after_skip=args.GemNet_num_after_skip,
            num_concat=args.GemNet_num_concat,
            num_atom=args.GemNet_num_atom,
            cutoff=args.GemNet_cutoff,
            int_cutoff=args.GemNet_int_cutoff,
            triplets_only=args.GemNet_triplets_only,
            direct_forces=args.GemNet_direct_forces,
            envelope_exponent=args.GemNet_envelope_exponent,
            extensive=args.GemNet_extensive,
            forces_coupled=args.GemNet_forces_coupled,
            output_init=args.GemNet_output_init,
            activation=args.GemNet_activation,
            scale_file=args.GemNet_scale_file,
            num_targets=num_tasks,
        )
        graph_pred_linear = None

    elif args.model_3d == "NequIP":
        # reference to https://github.com/mir-group/NequIP/discussions/131
        config = dict(
            model_builders=[
                "SimpleIrrepsConfig",
                "EnergyModel",
            ],
            dataset_statistics_stride=1,
            chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],
            r_max=args.NequIP_radius_cutoff,
            num_layers=5,
            chemical_embedding_irreps_out="64x0e",
            l_max=1,
            parity=True,
            num_features=64,
            nonlinearity_type="gate",
            nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
            nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
            resnet=False,
            num_basis=8,
            BesselBasis_trainable=True,
            PolynomialCutoff_p=6,
            invariant_layers=3,
            invariant_neurons=64,
            avg_num_neighbors=8,
            use_sc=True,
            compile_model=False,
        )
        model = model_from_config(config=config, initialize=True)
        graph_pred_linear = None

    elif args.model_3d == "Allegro":
        config = dict(
            model_builders=[
                "Geom3D.models.Allegro.model.Allegro",
            ],
            dataset_statistics_stride=1,
            chemical_symbols=["H", "C", "N", "O", "F", "Na", "Cl", "He", "P", "B"],
            default_dtype="float32",
            allow_tf32=False,
            model_debug_mode=False,
            equivariance_test=False,
            grad_anomaly_mode=False,
            _jit_bailout_depth=2,
            _jit_fusion_strategy=[("DYNAMIC", 3)],
            r_max=args.NequIP_radius_cutoff,
            num_layers=5,
            l_max=1,
            num_features=64,
            nonlinearity_type="gate",
            nonlinearity_scalars={'e': 'silu', 'o': 'tanh'},
            nonlinearity_gates={'e': 'silu', 'o': 'tanh'},
            num_basis=8,
            BesselBasis_trainable=True,
            PolynomialCutoff_p=6,
            invariant_layers=3,
            invariant_neurons=64,
            avg_num_neighbors=8,
            use_sc=True,
            
            parity="o3_full",
            mlp_latent_dimensions=[512],
        )
        model = model_from_config(config=config, initialize=True)
        graph_pred_linear = None

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

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))

    return model, graph_pred_linear


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

    elif args.model_3d == "DimeNet":
        model = DimeNet(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )
        graph_pred_linear = nn.Linear(args.emb_dim, num_tasks, bias=False)
        hidden_dim = args.emb_dim

    elif args.model_3d == "DimeNetPlusPlus":
        model = DimeNetPlusPlus(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=num_tasks,
            num_blocks=args.DimeNetPlusPlus_num_blocks,
            int_emb_size=args.DimeNetPlusPlus_int_emb_size,
            basis_emb_size=args.DimeNetPlusPlus_basis_emb_size,
            out_emb_channels=args.DimeNetPlusPlus_out_emb_channels,
            num_spherical=args.DimeNetPlusPlus_num_spherical,
            num_radial=args.DimeNetPlusPlus_num_radial,
            cutoff=args.DimeNetPlusPlus_cutoff,
            envelope_exponent=args.DimeNetPlusPlus_envelope_exponent,
            num_before_skip=args.DimeNetPlusPlus_num_before_skip,
            num_after_skip=args.DimeNetPlusPlus_num_after_skip,
            num_output_layers=args.DimeNetPlusPlus_num_output_layers,
        )
        graph_pred_linear = nn.Linear(args.DimeNetPlusPlus_out_emb_channels, num_tasks, bias=False)
        hidden_dim = args.DimeNetPlusPlus_out_emb_channels

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

    elif args.model_3d == "SEGNN":
        model = SEGNN(
            node_class,
            num_tasks,
            hidden_features=args.emb_dim,
            N=args.SEGNN_radius,
            lmax_h=args.SEGNN_N,
            lmax_pos=args.SEGNN_lmax_pos,
            norm=args.SEGNN_norm,
            pool=args.SEGNN_pool,
            edge_inference=args.SEGNN_edge_inference
        )
        graph_pred_linear = model.head_post_pool_layers
        model.head_post_pool_layers = nn.Identity()
        hidden_dim = args.emb_dim

    elif args.model_3d == "PaiNN":
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.PaiNN_n_interactions,
            n_rbf=args.PaiNN_n_rbf,
            cutoff=args.PaiNN_radius_cutoff,
            max_z=node_class,
            n_out=num_tasks,
            readout=args.PaiNN_readout,
        )
        graph_pred_linear = model.create_output_layers()
        hidden_dim = args.emb_dim

    elif args.model_3d == "GemNet":
        model = GemNet(
            # node_class=93,
            node_class=node_class,
            num_spherical=args.GemNet_num_spherical,
            num_radial=args.GemNet_num_radial,
            num_blocks=args.GemNet_num_blocks,
            emb_size_atom=args.emb_dim,
            emb_size_edge=args.emb_dim,
            emb_size_trip=args.GemNet_emb_size_trip,
            emb_size_quad=args.GemNet_emb_size_quad,
            emb_size_rbf=args.GemNet_emb_size_rbf,
            emb_size_cbf=args.GemNet_emb_size_cbf,
            emb_size_sbf=args.GemNet_emb_size_sbf,
            emb_size_bil_quad=args.GemNet_emb_size_bil_quad,
            emb_size_bil_trip=args.GemNet_emb_size_bil_trip,
            num_before_skip=args.GemNet_num_before_skip,
            num_after_skip=args.GemNet_num_after_skip,
            num_concat=args.GemNet_num_concat,
            num_atom=args.GemNet_num_atom,
            cutoff=args.GemNet_cutoff,
            int_cutoff=args.GemNet_int_cutoff,
            triplets_only=args.GemNet_triplets_only,
            direct_forces=args.GemNet_direct_forces,
            envelope_exponent=args.GemNet_envelope_exponent,
            extensive=args.GemNet_extensive,
            forces_coupled=args.GemNet_forces_coupled,
            output_init=args.GemNet_output_init,
            activation=args.GemNet_activation,
            scale_file=args.GemNet_scale_file,
            num_targets=num_tasks,
        )
        graph_pred_linear = None
        hidden_dim = args.emb_dim

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


def get_loss(y_graph, y_image, y_image_confidence=None, topn_ratio=1):
    assert y_graph.shape[0] == y_image.shape[0]
    assert topn_ratio >= 0 and topn_ratio <= 1  # topn 是个比例因为不同的 batch size，对应的 topn 是不一样的

    if y_image_confidence is not None:
        N = y_graph.shape[0]
        topn = int(N * topn_ratio)
        assert topn != 0
        topn_index = torch.argsort(y_image_confidence, dim=0, descending=True)[:topn].squeeze()  # 从大到小排序
        loss = F.smooth_l1_loss(y_graph[topn_index], y_image[topn_index], reduction='mean')
    else:
        loss = F.smooth_l1_loss(y_graph, y_image, reduction='mean')
    return loss


def get_adaptive_loss(y_graph, y_image, y_image_confidence_batch=None, y_image_confidence_all=None,
                      alpha_std_batch=0, alpha_std_all=0, beta_batch=0.5,
                      mean_all=None, std_all=None, threshold_all=None):
    """y_image_confidence_batch: 一个 batch 中样本的置信度，y_image_confidence_all: 所有样本的置信度"""
    assert y_graph.shape[0] == y_image.shape[0]
    assert beta_batch >= 0 and beta_batch <= 1

    if y_image_confidence_batch is None:  # 不进行置信度筛选
        loss = F.smooth_l1_loss(y_graph, y_image, reduction='mean')
        return loss
    else:  # 按照置信度进行数据的过滤
        mean_batch = torch.mean(y_image_confidence_batch).detach().cpu()
        if len(y_image_confidence_batch) > 1:
            std_batch = torch.std(y_image_confidence_batch).detach().cpu()
            threshold_batch = mean_batch + alpha_std_batch * std_batch  # 一条样本是计算不出来方差的，值为 nan
        else:
            threshold_batch = mean_batch

        threshold = threshold_batch
        if mean_all is not None and std_all is not None and threshold_all is not None:  # 表明已经预先计算了
            threshold = beta_batch * threshold_batch + (1 - beta_batch) * threshold_all
        elif y_image_confidence_all is not None:
            mean_all = torch.mean(y_image_confidence_all)
            std_all = torch.std(y_image_confidence_all)
            threshold_all = mean_all + alpha_std_all * std_all
            threshold = beta_batch * threshold_batch + (1 - beta_batch) * threshold_all

        flag = (y_image_confidence_batch >= threshold).flatten()
        if flag.sum() != 0:
            loss = F.smooth_l1_loss(y_graph[flag], y_image[flag], reduction='mean')
            return loss
        else:
            return torch.Tensor([0]).to(y_image.device)


def inference_y_image_confidence_all(device, loader, args):
    EDEvaluator.eval()
    EKEvaluator.eval()

    L = loader  # 禁用进度条，只在epoch结束时打印

    ED_image_confidence_all = []
    for step, batch in enumerate(L):
        batch = batch.to(device)
        ED_image_confidence = -1 * EDEvaluator(batch.img_feat)  # 乘以了-1，变为值越大越好
        ED_image_confidence_all.append(ED_image_confidence.detach().cpu())
    return torch.concat(ED_image_confidence_all)


def train(device, loader, y_image_confidence_all=None, mean_all=None, std_all=None, threshold_all=None):
    model.train()
    alignMapper.train()
    taskPredictor.train()

    EDPredictor.eval()
    EDEvaluator.eval()
    EKPredictor.eval()
    EKEvaluator.eval()

    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc_dict = {
        "total_loss": 0,
        "g_loss": 0,
        "kd_loss": 0,
        "ED_loss": 0,
        "EK_loss": 0,
    }
    num_iters = len(loader)

    L = loader  # 禁用进度条，只在epoch结束时打印

    for step, batch in enumerate(L):
        batch = batch.to(device)
        positions = batch.positions
        positions.requires_grad_()
        x = batch.x

        if args.model_3d == "SchNet":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch, extract_representation=True)
        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(x, positions, batch.radius_edge_index, batch.batch)
        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(x, positions, batch)
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
        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)
        elif args.model_3d in ["NequIP", "Allegro"]:
            data = {
                "atom_types": x,
                "pos": positions,
                "edge_index": batch.radius_edge_index,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"]
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:  # 进入这个函数里面的模型，都提取的是隐层特征
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:  # 没有 graph_pred_linear，就必须有 molecule_3D_y_pred
            pred_energy = molecule_3D_repr.squeeze(1)
        B = pred_energy.size()[0]

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        pred_force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0]

        # TODO: 蒸馏的代码加在这里
        ED_loss, EK_loss, kd_loss = torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device)
        if args.use_ED or args.use_EK:
            molecule_3D_repr = alignMapper(molecule_3D_repr)  # 映射到和 image 相同的维度

        if args.use_ED:
            ED_graph = EDPredictor(molecule_3D_repr)
            ED_image = EDPredictor(batch.img_feat)

            if args.use_evaluator:
                ED_image_confidence = -1 * EDEvaluator(batch.img_feat)  # 乘以了-1，变为值越大越好
                if args.evaluator_name == "topn_batch":
                    # top 只选择 topn 的知识进行迁移
                    ED_loss = get_loss(ED_graph, ED_image, y_image_confidence=ED_image_confidence, topn_ratio=args.topn_ratio)
                elif args.evaluator_name == "mean_std":
                    ED_loss = get_adaptive_loss(ED_graph, ED_image, y_image_confidence_batch=ED_image_confidence,
                                                y_image_confidence_all=y_image_confidence_all,
                                                alpha_std_batch=args.alpha_std_batch, alpha_std_all=args.alpha_std_all,
                                                beta_batch=args.beta_batch,
                                                mean_all=mean_all, std_all=std_all, threshold_all=threshold_all)
                else:
                    raise NotImplementedError
            else:
                ED_loss = get_loss(ED_graph, ED_image)

        if args.use_EK:
            EK_graph = EKPredictor(molecule_3D_repr)
            EK_image = EKPredictor(batch.img_feat)

            if args.use_evaluator:
                EK_image_confidence = -1 * EKEvaluator(batch.img_feat)  # 乘以了-1，变为值越大越好
                # top 只选择 topn 的知识进行迁移
                EK_loss = get_loss(EK_graph, EK_image, y_image_confidence=EK_image_confidence, topn_ratio=args.topn_ratio)
            else:
                EK_loss = get_loss(EK_graph, EK_image)

        actual_energy = batch.y
        actual_force = batch.force

        if args.use_kd:
            pred_image = taskPredictor(batch.img_feat).squeeze()
            kd_loss = get_loss(actual_energy, pred_image)

        g_energy_loss = args.md17_energy_coeff * criterion(pred_energy, actual_energy)
        g_force_loss = args.md17_force_coeff * criterion(pred_force, actual_force)
        g_loss = g_energy_loss + g_force_loss

        loss = g_loss + args.weight_kd * kd_loss + args.weight_ED * ED_loss + args.weight_EK * EK_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc_dict = {
            "total_loss": loss_acc_dict["total_loss"] + loss.cpu().detach().item(),
            "g_loss": loss_acc_dict["g_loss"] + g_loss.cpu().detach().item(),
            "kd_loss": loss_acc_dict["kd_loss"] + args.weight_kd * kd_loss.cpu().detach().item(),
            "ED_loss": loss_acc_dict["ED_loss"] + args.weight_ED * ED_loss.cpu().detach().item(),
            "EK_loss": loss_acc_dict["EK_loss"] + args.weight_EK * EK_loss.cpu().detach().item(),
        }

        if args.verbose and (step + 1) % 100 == 0:
            L.desc = (f"[train epoch {epoch}] "
                      f"total_loss: {loss_acc_dict['total_loss'] / (step + 1):.8f}; "
                      f"g_loss: {loss_acc_dict['g_loss'] / (step + 1):.8f}; "
                      f"kd_loss: {loss_acc_dict['kd_loss'] / (step + 1):.8f}; "
                      f"ED loss: {loss_acc_dict['ED_loss'] / (step + 1):.8f}; "
                      f"EK_loss: {loss_acc_dict['EK_loss'] / (step + 1):.8f}")

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
        elif args.model_3d == "DimeNetPlusPlus":
            molecule_3D_repr = model(x, positions, batch.batch)
        elif args.model_3d == "SphereNet":
            molecule_3D_repr = model(x, positions, batch.batch, extract_representation=True)
        elif args.model_3d == "PaiNN":
            molecule_3D_repr = model(x, positions, batch.radius_edge_index, batch.batch)
        elif args.model_3d == "GemNet":
            molecule_3D_repr = model(x, positions, batch)
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
        elif args.model_3d == "SEGNN":
            molecule_3D_repr = model(batch)
        elif args.model_3d in ["NequIP", "Allegro"]:
            data = {
                "atom_types": x,
                "pos": positions,
                "edge_index": batch.radius_edge_index,
                "batch": batch.batch,
            }
            out = model(data)
            molecule_3D_repr = out["total_energy"]
        elif args.model_3d == "Equiformer":
            molecule_3D_repr = model(node_atom=x, pos=positions, batch=batch.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        B = pred_energy.size()[0]

        if args.energy_force_with_normalization:
            pred_energy = pred_energy * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM

        force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0].detach_()

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
        model_weight = torch.load(model_weight_file, weights_only=False)
        model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    elif "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file, weights_only=False)
        if "model_3D" in model_weight:
            model.load_state_dict(model_weight["model_3D"])
        else:
            model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])

    else:
        model_weight = torch.load(model_weight_file, weights_only=False)
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

    if args.dataset == "MD17":
        data_root = f"{args.dataroot}/MD17"
        dataset = DatasetMD17(data_root, task=args.task)
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=args.seed)
    elif args.dataset == "rMD17":
        data_root = f"{args.dataroot}/rMD17"
        dataset = DatasetrMD17(data_root, task=args.task, img_feat_path=args.img_feat_path, split_id=args.rMD17_split_id)
        split_idx = dataset.get_idx_split()
    logger.info("train: {} {}".format(len(split_idx["train"]), split_idx["train"][:5]))
    logger.info("valid: {} {}".format(len(split_idx["valid"]), split_idx["valid"][:5]))
    logger.info("test: {} {}".format(len(split_idx["test"]), split_idx["test"][:5]))

    if args.model_3d == "PaiNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.PaiNN_radius_cutoff}/{args.task}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.PaiNN_radius_cutoff
        )
    elif args.model_3d in ["NequIP", "Allegro"]:
        # Will update this
        data_root = f"{args.dataroot}/{args.dataset}_{args.NequIP_radius_cutoff}/{args.task}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.NequIP_radius_cutoff
        )
    elif args.model_3d == "EGNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.EGNN_radius_cutoff}/{args.task}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.EGNN_radius_cutoff
        )
    elif args.model_3d == "SEGNN":
        data_root = f"{args.dataroot}/{args.dataset}_{args.SEGNN_radius}/{args.task}"
        dataset = MoleculeDataset3DRadius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.SEGNN_radius
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
    if args.model_3d == "GemNet":
        DataLoaderClass = DataLoaderGemNet
        dataloader_kwargs = {"cutoff": args.GemNet_cutoff, "int_cutoff": args.GemNet_int_cutoff, "triplets_only": args.GemNet_triplets_only}

    train_loader = DataLoaderClass(train_dataset, args.MD17_train_batch_size, shuffle=True, num_workers=args.num_workers, **dataloader_kwargs)
    val_loader = DataLoaderClass(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)
    test_loader = DataLoaderClass(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)

    node_class, num_tasks = 119, 1

    # set up model
    model, graph_pred_linear, hidden_dim = model_setup_for_kd()

    # 加载用于蒸馏模型
    image_dim = 512
    alignMapper = get_classifier("arch3", hidden_dim, image_dim)  # graph->image
    taskPredictor = get_classifier("arch3", image_dim, num_tasks)  # image->prediction

    # 从预训练模型中加载
    EDPredictor = Predictor(in_features=image_dim, out_features=768)
    EKPredictor = Predictor(in_features=image_dim, out_features=22)
    EDEvaluator = Predictor(in_features=image_dim, out_features=1)
    EKEvaluator = Predictor(in_features=image_dim, out_features=22)

    if args.pretrained_pth is not None:
        flag, resume_desc = load_checkpoint(args.pretrained_pth, EDPredictor, EKPredictor, EDEvaluator, EKEvaluator)
        assert flag
        logger.info(resume_desc)

    for my_model in [alignMapper, taskPredictor, EDPredictor, EKPredictor, EDEvaluator, EKEvaluator]:
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

    y_image_confidence_all, mean_all, std_all, threshold_all = None, None, None, None
    if args.use_evaluator and args.evaluator_name == "mean_std":
        y_image_confidence_all = inference_y_image_confidence_all(device, train_loader, args)
        mean_all = torch.mean(y_image_confidence_all)
        std_all = torch.std(y_image_confidence_all)
        threshold_all = mean_all + args.alpha_std_all * std_all
        n_dist = (y_image_confidence_all > threshold_all).sum()  # 这些样本会被蒸馏在训练过程中如果不考虑batch样本的话
        n_train = len(y_image_confidence_all)
        logger.info(f"从 global 角度，有 {n_dist}/{n_train} ({n_dist/n_train*100:.3f}%) 个样本会被蒸馏")

    train_energy_mae_list, train_force_mae_list = [], []
    val_energy_mae_list, val_force_mae_list = [], []
    test_energy_mae_list, test_force_mae_list = [], []
    best_val_force_mae = 1e10
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(device, train_loader, y_image_confidence_all, mean_all, std_all, threshold_all)
        logger.info("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_energy_mae, train_force_mae, train_eval_dict = eval(device, train_loader)
            else:
                train_energy_mae = train_force_mae = 0
                train_eval_dict = None
            val_energy_mae, val_force_mae, val_eval_dict = eval(device, val_loader)
            if args.eval_test:
                test_energy_mae, test_force_mae, test_eval_dict = eval(device, test_loader)
            else:
                test_energy_mae = test_force_mae = 0
                test_eval_dict = None

            train_energy_mae_list.append(train_energy_mae)
            train_force_mae_list.append(train_force_mae)
            val_energy_mae_list.append(val_energy_mae)
            val_force_mae_list.append(val_force_mae)
            test_energy_mae_list.append(test_energy_mae)
            test_force_mae_list.append(test_force_mae)
            logger.info("Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae, val_energy_mae, test_energy_mae))
            logger.info("Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae, val_force_mae, test_force_mae))

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

    if args.eval_test:
        optimal_test_energy, optimal_test_force = test_energy_mae_list[best_val_idx], test_force_mae_list[best_val_idx]
    else:
        optimal_model_weight = os.path.join(args.output_model_dir, "model.pth")
        load_model(model, graph_pred_linear, optimal_model_weight, load_latest=True)
        optimal_test_energy, optimal_test_force = eval(device, test_loader)

    logger.info("best Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae_list[best_val_idx], val_energy_mae_list[best_val_idx], optimal_test_energy))
    logger.info("best Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae_list[best_val_idx], val_force_mae_list[best_val_idx], optimal_test_force))
