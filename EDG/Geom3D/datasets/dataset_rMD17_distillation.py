import copy
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate
from tqdm import tqdm
from Geom3D.datasets.image_utils import read_image


class DatasetrMD17(InMemoryDataset):
    def __init__(self, root, task, img_feat_path, transform=None, pre_transform=None, pre_filter=None, split_id="01"):
        self.root = root
        self.dataset = "rMD17"
        self.task = task
        self.split_id = split_id  # This is from {01, 02, 03, 04, 05}, which have been provided in the original file.

        self.img_feat_path = img_feat_path
        if img_feat_path is not None and img_feat_path != "None" and img_feat_path != "":
            npz_data = np.load(img_feat_path, allow_pickle=True)
            self.img_feat = dict(zip(npz_data['drug_id'], npz_data['feats']))  # dict

        super(DatasetrMD17, self).__init__(self.root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return "rmd17_{}.npz".format(self.task)

    @property
    def raw_dir(self):
        return osp.join(self.root, "npz_data")

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    @property
    def processed_dir(self):
        return osp.join(self.root, self.task, "processed")

    def process(self):
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        E = data['energies']
        F = data['forces']
        R = data['coords']
        z = data['nuclear_charges']

        data_list = []
        for i in tqdm(range(len(E))):
            R_i = torch.tensor(R[i], dtype=torch.float32)
            atomic_number_list = torch.tensor(z, dtype=torch.int64)
            E_i = torch.tensor(E[i], dtype=torch.float32)
            F_i = torch.tensor(F[i], dtype=torch.float32)

            X_i = []
            for atomic_number in atomic_number_list:
                atom_features = atomic_number - 1
                X_i.append(atom_features)
            X_i = torch.tensor(X_i, dtype=torch.int64)

            data = Data(x=X_i, positions=R_i, y=E_i, force=F_i)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        return

    def get_idx_split(self):
        train_file = osp.join(self.root, "splits/index_train_{}.csv".format(self.split_id))
        test_file = osp.join(self.root, "splits/index_test_{}.csv".format(self.split_id))

        train_csv = pd.read_csv(train_file, header=None)
        train_idx_list = train_csv.values.squeeze()
        assert len(train_idx_list) == 1000
        
        test_csv = pd.read_csv(test_file, header=None)
        test_idx_list = test_csv.values.squeeze()
        assert len(test_idx_list) == 1000

        train_idx = torch.tensor(train_idx_list[:950])
        val_idx = torch.tensor(train_idx_list[950:])
        test_idx = torch.tensor(test_idx_list)

        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            data = copy.copy(self._data_list[idx])
            data.img_feat = self.img_feat[idx]
            return data

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)
        data.img_feat = self.img_feat[idx]

        return data
