import os.path
import pickle
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
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


def simple_transforms_no_aug(resize=224, mean_std=None):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms


def simple_transforms_for_train_1(resize=224, mean_std=None, p=0.2, rotation=False):
    if mean_std is None:
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    else:
        mean, std = mean_std[0], mean_std[1]
    img_transforms = transforms.Compose([transforms.Resize(resize), RandomApply(transforms.CenterCrop(resize), p=p),
                                         RandomApply(transforms.RandomRotation(degrees=360), p=p),
                                         RandomApply(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), p=p),
                                         transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return img_transforms



def transforms_for_train(config, is_training=True):
    """

    :param config: e.g. {'input_size': (3, 224, 224), 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'crop_pct': 0.9}
    :return: e.g. Compose(
                        RandomResizedCropAndInterpolation(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic)
                        RandomHorizontalFlip(p=0.5)
                        ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=None)
                        ToTensor()
                        Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
                    )
    """
    return create_transform(**config, is_training=is_training)


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


def load_video_data_list(dataroot, dataset, video_type="processed", label_column_name=None, video_dir_name="video",
                         csv_suffix="", is_cache=False, logger=None):
    log = print if logger is None else logger.info
    if is_cache:
        cache_path = f"{dataroot}/{dataset}/{video_type}/cache_{dataset}_load_video_data_list.pkl"
        if os.path.exists(cache_path):
            log(f"load from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data["video_index_list"], data["video_path_list"], data["video_label_list"]

    csv_file_path = os.path.join(dataroot, dataset, video_type, "{}{}.csv".format(dataset, csv_suffix))
    df = pd.read_csv(csv_file_path)
    columns = df.columns.tolist()
    assert "index" in columns
    video_root = f"{dataroot}/{dataset}/{video_type}/{video_dir_name}"

    if label_column_name is not None:
        assert label_column_name in columns
        video_label_list = df[label_column_name].apply(lambda x: str(x).split(' ')).tolist()
    else:
        video_label_list = [-1] * len(df)

    video_index_list = df["index"].tolist()
    video_path_list = []

    for video_index in tqdm(video_index_list, desc="load_video_data_list"):
        video_path = []
        for idx in range(60):
            video_path.append(f"{video_root}/{video_index}/{idx}.png")
        video_path_list.append(video_path)
        # for filename in os.listdir(f"{video_root}/{video_index}"):
        #     video_path.append(f"{video_root}/{video_index}/{filename}")
        # sort video_path (0.png, 1.png, 2.png, ...)
        # video_path.sort(key=lambda x: int(os.path.split(x)[1].split('.')[0]))
        # video_path_list.append(video_path)

    if is_cache:
        log(f"save to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "video_index_list": video_index_list,
                "video_path_list": video_path_list,
                "video_label_list": video_label_list},
                f)
        # np.savez(cache_path, video_index_list=video_index_list, video_path_list=video_path_list, video_label_list=video_label_list)
    return video_index_list, video_path_list, video_label_list


def check_num_frame_of_video(video_path_list, n_frame):
    """
    Check whether the frame number in the video is complete.
    :param video_path_list: e.g. [
                                    ["./video1/1.png", "./video1/2.png", ..., "./video1/n.png"],  # video1
                                    ["./video2/1.png", "./video2/2.png", ..., "./video2/n.png"],  # video2
                                ]
    :param n_frame: e.g. 60
    :return:
    """
    for idx, frame_path_list in enumerate(video_path_list):
        assert len(frame_path_list) == n_frame, \
            "The frame number of video {} is {}, not equal to the expected {}"\
                .format(idx, len(frame_path_list), n_frame)

