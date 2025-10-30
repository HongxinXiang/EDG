# 🐳 EDG: Electron Density-enhanced Molecular Geometry Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=HongxinXiang.EDG-X&left_color=gray&right_color=orange"></a>
[![Paper](https://img.shields.io/badge/Paper-IJCAI%202025-blue)](https://www.ijcai.org/proceedings/2025/872)

Official PyTorch implementation of "Electron Density-enhanced Molecular Geometry Learning" (IJCAI 2025)

- 🏠 [**Paper (Main Page)**](https://www.ijcai.org/proceedings/2025/872) – Main page of the EDG project.
- 📄 [**Paper (PDF)**](https://www.ijcai.org/proceedings/2025/0872.pdf) – Main paper.
- 📎 [**Appendix (PDF)**](https://github.com/HongxinXiang/EDG/blob/master/docs/EDG-Appendix.pdf) – Supplementary materials and technical details.

---


## 📑 Table of Contents
- [✨ News](#-news)
- [🧪 Summary](#-summary)
- [⚙️ Environments](#️-environments)
- [🚀 Pipeline](#-pipeline)
  - [Stage 1: ED Representation Learning with ImageED](#stage-1-ed-representation-learning-with-imageed)
  - [Stage 2: Pre-training of ED-aware Teacher](#stage-2-pre-training-of-ed-aware-teacher)
  - [Stage 3: ED-enhanced Molecular Geometry Learning](#stage-3-ed-enhanced-molecular-geometry-learning-on-downstream-tasks)
<!-- - [📊 Benchmarks](#-benchmarks)-->
- [📜 Citation](#-citation)

## ✨ News
- **[2025/09/19]** 🎉 Paper online!
- **[2025/04/29]** 🎉 Paper accepted to **IJCAI 2025**!
- **[2025/01/17]** 🛠️ Repository setup completed.


## 🧪 Summary
Electron density (ED), which describes the spatial distribution probability of electrons, plays a vital role in modeling energy and force distributions in molecular force fields (MFF). While existing machine learning force fields (MLFFs) typically enhance molecular geometry representations using atomic-level information, they often overlook the rich physical signals encoded in the electron cloud.

In this work, we introduce **EDG** — an efficient **E**lectron **D**ensity-enhanced molecular **G**eometry learning framework that leverages rendered ED images to enrich geometric representations for MLFFs.

We make three key contributions:
- **Image-based ED Representation**: We construct a large-scale dataset of 2 million 6-view RGB-D ED images.
- **ImageED**: A dedicated vision model trained to learn physical insights from ED images.
- **Cross-modal Knowledge Distillation**: An ED-aware teacher-student framework that transfers ED knowledge to geometry-based models.

Our method is model-agnostic and can be seamlessly integrated into existing MLFF architectures (e.g., SchNet, EGNN, SphereNet, ViSNet).  
Extensive experiments on QM9 and rMD17 demonstrate that EDG significantly improves geometry learning performance, with up to **33.7%** average gain.

<p align="center">
  <img src="/docs/images/overview.png" width="800">
</p>

## ⚙️ Environments

#### 1. GPU environment

- **CUDA**: 11.6
- **OS**: Ubuntu 18.04


#### 2. Conda Environment Setup
```bash
# Create conda environment
conda create -n EDG python=3.9
conda activate EDG

# Install dependencies
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

## 🚀 Pipeline

### Stage 1: ED Representation Learning with ImageED

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



### Stage 2: Pre-training of ED-aware Teacher

In order to improve the efficiency of pre-training ED-aware teacher, we provide 2 million ED features extracted by ImageED:

| Name        | Download link                                                | Description                                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ED features | [📥200w_ED_feats.pkl](https://1drv.ms/u/c/53030532e7d1aed6/EZclQc5cZYtMuoA_Dfy_N2wBkO9tKLETmRpP3f5NxqRwNw?e=0du0o7) | 2 million ED features extracted by ImageED. The pkl file is a dictionary: {"feats": ndarray, "ED_index_list": list} |

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

| Name             | Download link                                                                                              | Description                                                  |
| ---------------- |------------------------------------------------------------------------------------------------------------| ------------------------------------------------------------ |
| ED-aware Teacher | [📥Download](https://1drv.ms/u/c/53030532e7d1aed6/EdbT3drKAMlKv1QAHBw03WIBz03F_40TczOavhtqg4_FKg?e=RGzAWc) | The teacher trained for more than 280k steps on 2 million molecules: {"ED_teacher": `params`, "EDPredictor": `params`} |




### Stage 3: ED-enhanced Molecular Geometry Learning on Downstream Tasks

All downstream task data is publicly accessible:

| Benchmarks | #Datasets | #Task | Links                                                        |
| ---------- | --------- | ----- | ------------------------------------------------------------ |
| QM9        | 12        | 1     | [[OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/Et312b5E42JDp2OtY5ihSRYB4troL7EEaTSSDe4xCsxWlg?e=PL4HOl)] |
| rMD17      | 10        | 1     | [[OneDrive](https://1drv.ms/f/c/53030532e7d1aed6/EktMbMh96j5GkEjSehIIsG0BPJfUVJk-v8-DoxYppf41rQ?e=hZjhhR)] |

`Note`: We provided the structural image features extracted by ED-aware Teacher for more efficient distillation. You can directly use this feature for subsequent features.



To use EDG to enhance the learning of a geometry model, use the following command:

- QM9

```bash
model_3d=EGNN  # geometry models
dataroot=../datasets
dataset=QM9
task=alpha  # mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv
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

- rMD17

```bash
model_3d=SchNet  # geometry models
dataroot=../datasets
dataset=rMD17
task=ethanol  # ethanol,azobenzene,naphthalene,salicylic,toluene,aspirin,uracil,paracetamol,malonaldehyde,benzene
img_feat_path=teacher_features.npz  # structural image features extracted by ED-aware Teacher
pretrained_pth=ED-aware-Teacher.pth  # path to checkpoint of ED-aware teacher
weight_ED=1.0

python EDG/finetune_rMD17_EDG.py \
	--verbose \
	--model_3d $model_3d \
	--dataroot ../datasets \
	--dataset $dataset \
	--task $task \
	--rMD17_split_id 01 \
	--seed 42 \
	--epochs 1000 \
	--batch_size 128 \
	--lr 5e-4 \
	--emb_dim 128 \
	--lr_scheduler CosineAnnealingLR \
	--no_eval_train \
	--print_every_epoch 1 \
	--energy_force_with_normalization \
	--img_feat_path $img_feat_path \
	--num_workers 8 \
	--pretrained_pth $pretrained_pth \
    --output_model_dir ./experiments/$dataset/$task \
	--use_ED \
	--weight_ED $weight_ED
```





## 📜 Citation

If our paper or code is helpful to you, please do not hesitate to point a star for our repository and cite the following content.

```bib
@inproceedings{ijcai2025p872,
  title     = {Electron Density-enhanced Molecular Geometry Learning},
  author    = {Xiang, Hongxin and Xia, Jun and Jin, Xin and Du, Wenjie and Zeng, Li and Zeng, Xiangxiang},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {7840--7848},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/872},
  url       = {https://doi.org/10.24963/ijcai.2025/872},
}
```

