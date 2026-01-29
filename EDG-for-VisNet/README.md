
## Notes

- Please treat - Please treat this directory as an **independent project**.
- The environment setup should be **consistent with VisNet**.
- You may adjust `--weight_ED` according to the descriptions in the paper to **reproduce the reported results**.
- If you encounter any issues during installation or execution, please **open an issue** in this repository.

Quick Start

Example command:

```bash
dataroot=./datasets/rMD17
dataset=ethanol
logdir=./experiments/run_train_visnet_with_EDG
img_feat_path=teacher_features.npz
pretrained_pth=ED-aware-Teacher.pth
weight_ED=0.005

CUDA_VISIBLE_DEVICES=0 python train_with_EDG.py \
    --conf examples/ViSNet-rMD17.yml \
    --dataset-arg $dataset \
    --dataset-root $dataroot \
    --log-dir $logdir \
    --lr 0.0002 \
    --cutoff 5.0 \
    --seed 42 \
    --split_id 01 \
    --img_feat_path $img_feat_path \
    --pretrained_pth $pretrained_pth \
    --use_ED \
    --weight_ED $weight_ED
```


