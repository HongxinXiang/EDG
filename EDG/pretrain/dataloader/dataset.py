import collections.abc as container_abcs
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch, InMemoryDataset
from torchvision import transforms
import time
from dataloader.data_utils import load_video_data_list, check_num_frame_of_video
from utils.splitter import *
from dataloader.data_utils import read_image
string_classes, int_classes = str, int


class DualCollater(object):
    def __init__(self, follow_batch, multigpu=False):
        self.follow_batch = follow_batch
        self.multigpu = multigpu

    def collate(self, batch):

        elem = batch[0]
        if isinstance(elem, Data):
            if self.multigpu:
                return batch
            else:
                return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, np.ndarray):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class PretrainQCDataset(Dataset):
    def __init__(self, dataroot, dataset, use_ED, ED_path, use_EK, EK_path, split="all", transforms=None, ret_index=False, args=None,
                 idx_frame_list=[0, 12, 32, 52], logger=None, img_type="RGB"):
        """
        取 [0, 12, 32, 52] 的意思是：0是原始图片，12、32和 52 表示在x、y、z轴上旋转 180°
        :param video_path_list: e.g. [
                                        ["./video1/1.png", "./video1/2.png", ..., "./video1/n.png"],  # video1
                                        ["./video2/1.png", "./video2/2.png", ..., "./video2/n.png"],  # video2
                                        ...
                                    ]
        :param video_labels: video label
        :param video_indexs: video index
        :param transforms:
        :param n_frame: number of frame for each video
        :param ret_index:
        :param args:
        """
        self.logger = logger
        self.log = print if logger is None else logger.info
        self.use_ED, self.ED_path, self.use_EK, self.EK_path = use_ED, ED_path, use_EK, EK_path
        self.img_type = img_type
        self.split = split
        assert split in ["all", "train", "valid"]
        # load data
        n_frame = len(idx_frame_list)
        index_list, video_path_list, ED, EK, EK_knowledge, label_list = self.load_data(dataroot, dataset, idx_frame_list)
        self.args = args
        self.indexs = index_list
        self.video_path_list = video_path_list
        self.ED = ED
        self.EK = EK
        self.EK_knowledge = EK_knowledge
        self.label_list = label_list

        self.total_video = len(self.video_path_list)
        self.total_frame = len(self.video_path_list) * n_frame
        self.transforms = transforms
        self.n_frame = n_frame
        self.ret_index = ret_index

    def load_data(self, dataroot, dataset, idx_frame_list):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        suffix = "_".join([str(item) for item in idx_frame_list])
        cache_data_path = f"./cache/PretrainQCDataset@{dataset}@{self.split}@{suffix}@ED{self.use_ED}@EK{self.use_EK}.npz"
        if os.path.exists(cache_data_path):
            self.log(f"load cache from {cache_data_path}")
            data = np.load(cache_data_path, allow_pickle=True, mmap_mode='r')  # mmap_mode='r' 可以提速
            return data["index_list"], data["video_path_list"], data["ED"].item(), data["EK"].item(), data["EK_knowledge"], data["label_list"]
        else:
            # load multi-view data
            index_list, video_path_list, label_list = load_video_data_list(dataroot, dataset, label_column_name=None,
                                                                           is_cache=True, logger=self.logger)
            n_total = len(index_list)
            if self.split == "train":
                index_list, video_path_list, label_list = index_list[:int(n_total * 0.95)], video_path_list[:int(
                    n_total * 0.95)], label_list[:int(n_total * 0.95)]
            elif self.split == "valid":
                index_list, video_path_list, label_list = index_list[int(n_total*0.95):], video_path_list[int(
                    n_total*0.95):], label_list[int(n_total*0.95):]
            else:
                raise ValueError

            tmp_video_path_list = []
            for tmp_list in tqdm(video_path_list, desc=f"sample {idx_frame_list}"):
                tmp_video_path_list.append(np.array(tmp_list)[idx_frame_list].tolist())
            video_path_list = tmp_video_path_list
            n_frame = len(idx_frame_list)
            check_num_frame_of_video(video_path_list, n_frame)

            # load 1-D electronic density (768 features)
            if self.use_ED:
                with open(self.ED_path, "rb") as f:
                    ED_data = pickle.load(f)
                    print(f"ED data from {ED_data['resume']}")
                    ED = dict(zip(ED_data["video_index_list"], ED_data["feats"].tolist()))
            else:
                ED = None

            # load 1-D electronic knowledge (22 features, 有 nan)
            if self.use_EK:
                df_EK = pd.read_csv(self.EK_path)
                df_EK_index, df_EK_knowledge = df_EK["index"].astype(str), df_EK[list(set(df_EK.columns) - set(["index"]))]
                EK_knowledge = df_EK_knowledge.columns
                EK = dict(zip(df_EK_index.tolist(), df_EK_knowledge.to_numpy().tolist()))
            else:
                EK_knowledge = None
                EK = None

            self.log(f"save cache to {cache_data_path}")
            np.savez(cache_data_path, index_list=index_list, video_path_list=video_path_list,
                     label_list=label_list, ED=ED, EK=EK, EK_knowledge=EK_knowledge)

            return index_list, video_path_list, ED, EK, EK_knowledge, label_list

    def get_EK(self, index):
        video_index = str(self.indexs[index])
        return torch.from_numpy(np.array(self.EK[video_index]))

    def get_video(self, index):
        frame_path_list = self.video_path_list[index]
        video = [read_image(frame_path, self.img_type) for frame_path in frame_path_list]
        if self.transforms is not None:
            video = list(map(lambda img: self.transforms(img).unsqueeze(0), video))
            video = torch.cat(video)
        return video

    def get_ED(self, index):
        video_index = str(self.indexs[index])
        try:
            mask = torch.Tensor([1])
            return torch.from_numpy(np.array(self.ED[video_index])), mask
        except:
            # 有可能找不到，因为只有 1.9+ million 的分子
            mask = torch.Tensor([0])
            rubbish_data = torch.rand(size=(768, )).double()
            return rubbish_data, mask


    def __getitem__(self, index):
        # start = time.perf_counter()  # Python3.8不支持clock了，使用timer.perf_counter()
        video = self.get_video(index)
        # end = time.perf_counter()
        # self.log(f'get_video time: %s Seconds' % (end - start))
        if self.use_EK:
            EK_feat = self.get_EK(index)  # 有的 knowledge 没有，用 nan 表示
        else:
            EK_feat = torch.Tensor([np.nan])  # 占位符

        if self.use_ED:
            # start = time.perf_counter()  # Python3.8不支持clock了，使用timer.perf_counter()
            ED_feat, ED_mask = self.get_ED(index)
            # end = time.perf_counter()
            # self.log(f'get_ED time: %s Seconds' % (end - start))
        else:
            ED_feat, ED_mask = torch.Tensor([np.nan]), torch.Tensor([0])  # 占位符

        if self.ret_index:
            return video, EK_feat, ED_feat, ED_mask, self.indexs[index]
        else:
            return video, EK_feat, ED_feat, ED_mask

    def __len__(self):
        return self.total_video

