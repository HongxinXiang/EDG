# EDG

### Official PyTorch-based implementation of Paper "Electron Density-enhanced Molecular Geometry Learning"



## News!

**[2025/04/29]** 🔔️ Accepted in IJCAI 2025

**[2025/01/17]** Repository installation completed.

## Environments

#### 1. GPU environment

CUDA 11.6

Ubuntu 18.04


#### 2. create conda environment
```bash
# create conda env
conda create -n EDG python=3.9
conda activate EDG

# install environment
pip install rdkit
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install timm==0.6.12
pip install tensorboard
pip install scikit-learn
pip install setuptools==59.5.0
pip install pandas
pip install torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1%2Bcu116.html
pip install torch-geometric==1.6.0
pip install dgl-cu116
pip install ogb
```



## Stage 1: ED Representation Learning with ImageED

The pre-trained ImageED can be accessed in following table:

| Name                | Download link                | Description                                                             |
| ------------------- |------------------------------|-------------------------------------------------------------------------|
| Pre-trained ImageED | [ImageED with ViT-Base/16](https://1drv.ms/u/c/53030532e7d1aed6/EaAkIztKCqxAtZZx0dVXGgIB8OIJ-TI9VwRq3zvqMGwtuQ?e=S3gSeB) | You can download the ImageED for the feature extraction from ED images. |

run command to train ImageED:
```bash
log_dir=./experiments/pre-training/ImageED
batch_size=8
data_path=./pre-training/200w/cubes2pymol/6_views

python ImageED/pretrain_ImageED.py \ 
  --log_dir $log_dir \ 
  --output_dir $log_dir/checkpoints \
  --batch_size $batch_size \
  --model mae_vit_base_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.25 \
  --epochs 800 \
  --warmup_epochs 5 \
  --blr 0.00015 \
  --weight_decay 0.05 \
  --data_path $data_path \
  --world_size 1 \
  --local_rank 0 \
  --dist_url tcp://127.0.0.1:12345
```



## Stage 2: Pre-training of ED-aware Teacher

In order to improve the efficiency of pre-training ED-aware teacher, we provide 2 million ED features extracted by ImageED:

| Name        | Download link                                                | Description                                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ED features | [200w_ED_feats.pkl](https://1drv.ms/u/c/53030532e7d1aed6/EZclQc5cZYtMuoA_Dfy_N2wBkO9tKLETmRpP3f5NxqRwNw?e=0du0o7) | 2 million ED features extracted by ImageED. The pkl file is a dictionary: {"feats": ndarray, "ED_index_list": list} |



After downloading the `pkl` data, run the following command to train the ED-aware teacher:

```bash
dataroot=[your root]
dataset=[your dataset]
log_dir=./experiments/ED_teacher
ED_path=[path to 200w_ED_feats.pkl]

python ED_teacher/pretrain_ED_teachers.py \
	--model_name resnet18 \
	--lr 0.005 \
	--epochs 50 \
	--batch 128 \
	--dataroot $dataroot \
	--dataset $dataset \
	--log_dir $log_dir \
	--workers 16 \
	--validation-split 0.02 \
	--use_ED \
	--ED_path $ED_path
```



We provide the weight files of the pre-trained ED-aware teacher as follows:

| Name             | Download link                                                | Description                                                  |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ED-aware Teacher | [ED-aware Teacher](https://1drv.ms/u/c/53030532e7d1aed6/EdbT3drKAMlKv1QAHBw03WIBz03F_40TczOavhtqg4_FKg?e=RGzAWc) | The teacher trained for more than 280k steps on 2 million molecules: {"ED_teacher": `params`, "EDPredictor": `params`} |




## Stage 3: ED-enhanced Molecular Geometry Learning on Downstream Tasks

All downstream task data is publicly accessible:

| Benchmarks | #Datasets | #Task | Links                                                        |
| ---------- | --------- | ----- | ------------------------------------------------------------ |
| QM9        | 12        | 1     | [[OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/Et312b5E42JDp2OtY5ihSRYB4troL7EEaTSSDe4xCsxWlg?e=PL4HOl)] |
| rMD17      | 10        | 1     | [[OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/EktMbMh96j5GkEjSehIIsG0BPJfUVJk-v8-DoxYppf41rQ?e=hZjhhR)] |

`Note`: We provided the structural image features extracted by ED-aware Teacher for more efficient distillation. You can directly use this feature for subsequent features.



To use EDG to enhance the learning of a geometry model, use the following command:

```bash
model_3d=EGNN  # geometry models
dataroot=../datasets
dataset=QM9
task=alpha
img_feat_path=teacher_features.npz  # structural image features extracted by ED-aware Teacher
pretrained_pth=ED-aware-Teacher.pth  # path to checkpoint of ED-aware teacher
weight_ED=1.0

python EDG/finetune_QM9_EDG.py \
	--verbose \
	--model_3d $model_3d \
	--dataroot ../datasets \
	--dataset $dataset \
	--task $task \
	--split customized_01 \
	--seed 42 \
	--epochs 1000 \
	--batch_size 128 \
	--lr 5e-4 \
	--emb_dim 128 \
	--lr_scheduler CosineAnnealingLR \
	--no_eval_train \
	--print_every_epoch 1 \
	--img_feat_path $img_feat_path \
	--num_workers 8 \
	--pretrained_pth $pretrained_pth \
	--output_model_dir ./experiments/$dataset/$task \
	--use_ED \
	--weight_ED $weight_ED
```
