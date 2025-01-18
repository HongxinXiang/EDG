import os.path
import pickle
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torchvision import transforms
from tqdm import tqdm


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def transforms_no_aug(resize=224, mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def transforms_for_train(resize=224, mean_std=None, p=0.2, rotation=False):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.Resize(resize), RandomApply(transforms.CenterCrop(resize), p=p),
                                         RandomApply(transforms.RandomRotation(degrees=360), p=p),
                                         RandomApply(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), p=p),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def check_num_view_of_structural_image(structural_image_path_list, n_frame):
    """
    Check whether the frame number in the structural_image is complete.
    :param structural_image_path_list: e.g. [
                                    ["./structural_image1/1.png", "./structural_image1/2.png", ..., "./structural_image1/n.png"],  # structural_image1
                                    ["./structural_image2/1.png", "./structural_image2/2.png", ..., "./structural_image2/n.png"],  # structural_image2
                                ]
    :param n_frame: e.g. 60
    :return:
    """
    for idx, frame_path_list in enumerate(structural_image_path_list):
        assert len(frame_path_list) == n_frame, \
            "The frame number of structural_image {} is {}, not equal to the expected {}"\
                .format(idx, len(frame_path_list), n_frame)


def load_structural_image_data_list(dataroot, dataset, structural_image_type="processed", label_column_name=None, structural_image_dir_name="structural_image",
                         csv_suffix="", is_cache=False, logger=None):
    log = print if logger is None else logger.info
    if is_cache:
        cache_path = f"{dataroot}/{dataset}/{structural_image_type}/cache_{dataset}_load_structural_image_data_list.pkl"
        if os.path.exists(cache_path):
            log(f"load from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data["structural_image_index_list"], data["structural_image_path_list"], data["structural_image_label_list"]

    csv_file_path = os.path.join(dataroot, dataset, structural_image_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns
    structural_image_root = f"{dataroot}/{dataset}/{structural_image_type}/{structural_image_dir_name}"

    if label_column_name is not None:
        assert label_column_name in columns
        structural_image_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    else:
        structural_image_label_list = [-1] * len(df)

    structural_image_index_list = df["index"].tolist()
    structural_image_path_list = []

    for structural_image_index in tqdm(structural_image_index_list, desc="load_structural_image_data_list"):
        structural_image_path = []
        for idx in range(60):
            structural_image_path.append(f"{structural_image_root}/{structural_image_index}/{idx}.png")
        structural_image_path_list.append(structural_image_path)

    if is_cache:
        log(f"save to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "structural_image_index_list": structural_image_index_list,
                "structural_image_path_list": structural_image_path_list,
                "structural_image_label_list": structural_image_label_list},
                f)
    return structural_image_index_list, structural_image_path_list, structural_image_label_list


def read_image(image_path, img_type="RGB"):
    """从 image_path 从读取图片
    如果 img_type="RGB"，则直接读取；
    如果 img_type="BGR"，则将 BGR 转换为 RGB
    """
    if img_type == "RGB":
        return Image.open(image_path).convert('RGB')
    elif img_type == "BGR":
        img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img
    else:
        raise NotImplementedError

