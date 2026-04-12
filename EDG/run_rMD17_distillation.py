import shutil
import sys
sys.path.append("../")
import os
import argparse
from concurrent.futures import ThreadPoolExecutor


def executreCommand(command):
    # 执行命令
    os.system(command)


def construct_model_cmd(args):
    if args.model_3d in ["SchNet", "EGNN", "SphereNet"]:
        model_param_cmd = "--epochs 1000 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR"  #  --MD17_train_batch_size 128, 默认是 1
        model_param_dir = "e1000_b128_lr5e-4_ed128_lsCosine"
    elif args.model_3d in ["Equiformer"]:
        model_param_cmd = "--epochs 300 --batch_size 128 --lr 5e-4 --emb_dim 128 --lr_scheduler CosineAnnealingLR"
        model_param_dir = "e300_b128_lr5e-4_ed128_lsCosine"
    else:
        raise NotImplementedError
    return model_param_dir, model_param_cmd

def construct_distillation_cmd(args, weight_ED, weight_EK, weight_kd, topn_ratio, alpha_std_batch, alpha_std_all, beta_batch):
    dist_strategy_dir = ""
    dist_params = ""
    use_kd_cmd = ""
    use_ED_cmd = ""
    use_EK_cmd = ""
    use_evaluator_cmd = ""

    if args.use_kd:
        if dist_strategy_dir == "":
            dist_strategy_dir = "kd"
        else:
            dist_strategy_dir += "_kd"
        if dist_params == "":
            dist_params = f"kd{weight_kd}"
        else:
            dist_params += f"_kd{weight_kd}"
        use_kd_cmd = f"--use_kd --weight_kd {weight_kd}"

    if args.use_ED:
        if dist_strategy_dir == "":
            dist_strategy_dir = "ED"
        else:
            dist_strategy_dir += "_ED"
        if dist_params == "":
            dist_params = f"ED{weight_ED}"
        else:
            dist_params += f"_ED{weight_ED}"
        use_ED_cmd = f"--use_ED --weight_ED {weight_ED}"

    if args.use_EK:
        if dist_strategy_dir == "":
            dist_strategy_dir = "EK"
        else:
            dist_strategy_dir += "_EK"
        if dist_params == "":
            dist_params = f"EK{weight_EK}"
        else:
            dist_params += f"_EK{weight_EK}"
        use_EK_cmd = f"--use_EK --weight_EK {weight_EK}"

    if args.use_evaluator:
        if dist_strategy_dir == "":
            dist_strategy_dir = f"E@{args.evaluator_name}"
        else:
            dist_strategy_dir += f"_E@{args.evaluator_name}"
        if args.evaluator_name == "topn_batch":
            eval_params = f"E{topn_ratio}"
            use_evaluator_cmd_part = f"--topn_ratio {topn_ratio}"
        elif args.evaluator_name == "mean_std":
            eval_params = f"E@asb{alpha_std_batch}_asa{alpha_std_all}_bb{beta_batch}"
            use_evaluator_cmd_part = f"--alpha_std_batch {alpha_std_batch} --alpha_std_all {alpha_std_all} --beta_batch {beta_batch}"
        else:
            raise NotImplementedError
        if dist_params == "":
            dist_params = eval_params
        else:
            dist_params += f"_{eval_params}"
        use_evaluator_cmd = f"--use_evaluator --evaluator_name {args.evaluator_name} {use_evaluator_cmd_part}"

    if dist_strategy_dir == "" and dist_params == "":
        dist_dir = "no_dist"
    else:
        dist_dir = dist_strategy_dir + "/" + dist_params

    return dist_dir, use_kd_cmd, use_ED_cmd, use_EK_cmd, use_evaluator_cmd


parser = argparse.ArgumentParser()
parser.add_argument('--gpus', default="0,1,2,3", type=str, help='使用的gpu编号, e.g. --gpus 0,1,2,3')
parser.add_argument('--m', default=8, type=int, help='一个gpu跑多少个线程')

# data-related
parser.add_argument('--dataroot', default="/data/xianghongxin/datasets/Geom3D", type=str, help='')
parser.add_argument('--tasks', type=str, help='')

# model-related
parser.add_argument('--model_3d', default="SchNet", type=str, help='')

# distillation-related
parser.add_argument('--img_feat_root', type=str, default="/data/xianghongxin/datasets/Geom3D/rMD17", help='')
parser.add_argument('--img_feat_fn', type=str, default="ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz", help='')
parser.add_argument('--pretrained_pth', default=None, type=str, help='')
parser.add_argument('--use_kd', action='store_true', default=False, help='KD loss')
parser.add_argument('--use_ED', action='store_true', default=False, help='whether use electronic density')
parser.add_argument('--use_EK', action='store_true', default=False, help='electronics-related knowledge')
parser.add_argument('--use_evaluator', action='store_true', default=False, help='whether use evaluator')
parser.add_argument('--evaluator_name', type=str, default="topn_batch", choices=["topn_batch", "mean_std"], help='name of evaluator, topn 是选择一个 batch 中的 topn 来蒸馏。mean_std 根据是根据均值和方差来的')
# topn_batch
parser.add_argument("--topn_ratios", type=str, default="1", help="use_evaluator，选择top多少比例的样本来计算损失")
# mean_std
parser.add_argument("--alpha_std_batchs", type=str, default="0", help="一个 batch 中方差的偏移量")
parser.add_argument("--alpha_std_alls", type=str, default="0", help="所有样本中方差的偏移量")
parser.add_argument("--beta_batchs", type=str, default="0.5", help="")
# ED, EK, KD 的权重
parser.add_argument("--weight_EDs", type=str, default="1", help="计算损失时的权重")
parser.add_argument("--weight_EKs", type=str, default="1", help="计算损失时的权重")
parser.add_argument("--weight_kds", type=str, default="1", help="计算损失时的权重")
# train-related
parser.add_argument('--runseed', default="42", type=str, help='runseed')

# log-related
parser.add_argument('--output_model_dir', default="./experiments/run_rMD17_distillation", type=str, help='')


args = parser.parse_args()

gpus = [0, 1, 2, 3] if args.gpus is None else [int(item) for item in args.gpus.split(",") if not item.isspace()]

executor = []  # 多线程执行器
for gpu in gpus:
    executor.append(ThreadPoolExecutor(max_workers=args.m))

#################### run
# hyper-parameters selection
# batchs = [32] if args.batchs is None else [int(item) for item in args.batchs.split(",") if not item.isspace()]  # Check, [8, 16, 32, 64, 128]
# lrs = [0.001] if args.lrs is None else [float(item) for item in args.lrs.split(",") if not item.isspace()]
tasks = ['ethanol', 'azobenzene', 'naphthalene', 'salicylic', 'toluene', 'aspirin', 'uracil', 'paracetamol', 'malonaldehyde', 'benzene'] if args.tasks is None else [item for item in args.tasks.split(",") if not item.isspace()]  # Check,

alpha_std_batchs = ["0"] if args.alpha_std_batchs is None else [item for item in args.alpha_std_batchs.split(",") if not item.isspace()]  # Check
alpha_std_alls = ["0"] if args.alpha_std_alls is None else [item for item in args.alpha_std_alls.split(",") if not item.isspace()]  # Check
beta_batchs = ["0.5"] if args.beta_batchs is None else [item for item in args.beta_batchs.split(",") if not item.isspace()]  # Check

weight_EDs = ["1"] if args.weight_EDs is None else [item for item in args.weight_EDs.split(",") if not item.isspace()]  # Check
weight_EKs = ["1"] if args.weight_EKs is None else [item for item in args.weight_EKs.split(",") if not item.isspace()]  # Check
weight_kds = ["1"] if args.weight_kds is None else [item for item in args.weight_kds.split(",") if not item.isspace()]  # Check
topn_ratios = ["1"] if args.topn_ratios is None else [item for item in args.topn_ratios.split(",") if not item.isspace()]  # Check
runseeds = [42] if args.runseed is None else [int(item) for item in args.runseed.split(",") if not item.isspace()]

model_param_dir, model_param_cmd = construct_model_cmd(args)

# basic params
index = 0

for task in tasks:
    img_feat_path = f"{args.img_feat_root}/{task}/processed/teacher_feats/{args.img_feat_fn}"
    for weight_ED in weight_EDs:
        for weight_EK in weight_EKs:
            for weight_kd in weight_kds:
                for topn_ratio in topn_ratios:
                    for alpha_std_batch in alpha_std_batchs:
                        for alpha_std_all in alpha_std_alls:
                            for beta_batch in beta_batchs:
                                dist_dir, use_kd_cmd, use_ED_cmd, use_EK_cmd, use_evaluator_cmd = construct_distillation_cmd(
                                    args, weight_ED, weight_EK, weight_kd, topn_ratio, alpha_std_batch, alpha_std_all,
                                    beta_batch)
                                for runseed in runseeds:
                                    pretrained_pth = "" if args.pretrained_pth is None or args.pretrained_pth == "None" else f"--pretrained_pth {args.pretrained_pth}"
                                    output_model_dir = f"{args.output_model_dir}/{args.model_3d}/{model_param_dir}/{dist_dir}/rs{runseed}/{task}".replace("\\", "")  # 传参数的时候可能有的字段需要转义，这里给他处理一下

                                    log_path = f"{output_model_dir}/logs.log"
                                    if os.path.exists(log_path):
                                        with open(log_path, "r") as f:
                                            lines = f.readlines()
                                        if "best Force	train" in lines[-1]:  # 有这个信息表明跑完了
                                            continue
                                        else:
                                            shutil.rmtree(output_model_dir)  # 没跑完，把日志删了重新抛
                                            print(f"rmtree: {output_model_dir}")
                                            pass

                                    python_bin = os.environ.get("PYTHON_BIN", "python")
                                    command = (f"CUDA_VISIBLE_DEVICES={gpus[index % len(gpus)]} {python_bin} finetune_MD17_distillation.py "
                                               f"--verbose --model_3d {args.model_3d} --dataroot {args.dataroot} --dataset rMD17 "
                                               f"--task {task} --rMD17_split_id 01 --seed {args.runseed} {model_param_cmd} "
                                               f"--no_eval_train --print_every_epoch 1 --energy_force_with_normalization "
                                               f"--img_feat_path {img_feat_path} --num_workers 8 {pretrained_pth} "
                                               f"--output_model_dir {output_model_dir} "
                                               f"{use_kd_cmd} {use_ED_cmd} {use_EK_cmd} {use_evaluator_cmd}")
                                    # print(command)
                                    executor[index % len(gpus)].submit(executreCommand, command)
                                    index += 1
print(index)

