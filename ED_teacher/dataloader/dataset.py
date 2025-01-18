import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from ED_teacher.dataloader.data_utils import load_structural_image_data_list, check_num_view_of_structural_image
from ED_teacher.dataloader.data_utils import read_image

string_classes, int_classes = str, int


class PretrainQCDataset(Dataset):
    def __init__(self, dataroot, dataset, use_ED, ED_path, split="all", transforms=None, ret_index=False, args=None,
                 idx_frame_list=[0, 12, 32, 52], logger=None, img_type="RGB"):
        self.logger = logger
        self.log = print if logger is None else logger.info
        self.use_ED, self.ED_path = use_ED, ED_path
        self.img_type = img_type
        self.split = split
        assert split in ["all", "train", "valid"]
        # load data
        n_frame = len(idx_frame_list)
        index_list, structural_image_path_list, ED, label_list = self.load_data(dataroot, dataset, idx_frame_list)
        self.args = args
        self.indexs = index_list
        self.structural_image_path_list = structural_image_path_list
        self.ED = ED
        self.label_list = label_list

        self.total_structural_image = len(self.structural_image_path_list)
        self.total_frame = len(self.structural_image_path_list) * n_frame
        self.transforms = transforms
        self.n_frame = n_frame
        self.ret_index = ret_index

    def load_data(self, dataroot, dataset, idx_frame_list):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        suffix = "_".join([str(item) for item in idx_frame_list])
        cache_data_path = f"./cache/PretrainQCDataset@{dataset}@{self.split}@{suffix}@ED{self.use_ED}.npz"
        if os.path.exists(cache_data_path):
            self.log(f"load cache from {cache_data_path}")
            data = np.load(cache_data_path, allow_pickle=True, mmap_mode='r')  # mmap_mode='r' 可以提速
            return data["index_list"], data["structural_image_path_list"], data["ED"].item(), data["label_list"]
        else:
            # load multi-view data
            index_list, structural_image_path_list, label_list = load_structural_image_data_list(dataroot, dataset, label_column_name=None,
                                                                           is_cache=True, logger=self.logger)
            n_total = len(index_list)
            if self.split == "train":
                index_list, structural_image_path_list, label_list = index_list[:int(n_total * 0.95)], structural_image_path_list[:int(
                    n_total * 0.95)], label_list[:int(n_total * 0.95)]
            elif self.split == "valid":
                index_list, structural_image_path_list, label_list = index_list[int(n_total*0.95):], structural_image_path_list[int(
                    n_total*0.95):], label_list[int(n_total*0.95):]
            else:
                raise ValueError

            tmp_structural_image_path_list = []
            for tmp_list in tqdm(structural_image_path_list, desc=f"sample {idx_frame_list}"):
                tmp_structural_image_path_list.append(np.array(tmp_list)[idx_frame_list].tolist())
            structural_image_path_list = tmp_structural_image_path_list
            n_frame = len(idx_frame_list)
            check_num_view_of_structural_image(structural_image_path_list, n_frame)

            # load 1-D electronic density (768 features)
            if self.use_ED:
                with open(self.ED_path, "rb") as f:
                    ED_data = pickle.load(f)
                    print(f"ED data from {ED_data['resume']}")
                    ED = dict(zip(ED_data["structural_image_index_list"], ED_data["feats"].tolist()))
            else:
                ED = None

            self.log(f"save cache to {cache_data_path}")
            np.savez(cache_data_path, index_list=index_list, structural_image_path_list=structural_image_path_list,
                     label_list=label_list, ED=ED)

            return index_list, structural_image_path_list, ED, label_list

    def get_structural_image(self, index):
        frame_path_list = self.structural_image_path_list[index]
        structural_image = [read_image(frame_path, self.img_type) for frame_path in frame_path_list]
        if self.transforms is not None:
            structural_image = list(map(lambda img: self.transforms(img).unsqueeze(0), structural_image))
            structural_image = torch.cat(structural_image)
        return structural_image

    def get_ED(self, index):
        structural_image_index = str(self.indexs[index])
        try:
            mask = torch.Tensor([1])
            return torch.from_numpy(np.array(self.ED[structural_image_index])), mask
        except:
            # 有可能找不到
            mask = torch.Tensor([0])
            rubbish_data = torch.rand(size=(768, )).double()
            return rubbish_data, mask


    def __getitem__(self, index):
        structural_image = self.get_structural_image(index)

        if self.use_ED:
            # start = time.perf_counter()  # Python3.8不支持clock了，使用timer.perf_counter()
            ED_feat, ED_mask = self.get_ED(index)
            # end = time.perf_counter()
            # self.log(f'get_ED time: %s Seconds' % (end - start))
        else:
            ED_feat, ED_mask = torch.Tensor([np.nan]), torch.Tensor([0])  # 占位符

        if self.ret_index:
            return structural_image, ED_feat, ED_mask, self.indexs[index]
        else:
            return structural_image, ED_feat, ED_mask

    def __len__(self):
        return self.total_structural_image

