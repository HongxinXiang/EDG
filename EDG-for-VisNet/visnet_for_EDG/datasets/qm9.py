import copy

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.transforms import Compose


class QM9(QM9_geometric):
    def __init__(self, root, img_feat_path, transform=None, pre_transform=None, pre_filter=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        super(QM9, self).__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        self.img_feat_path = img_feat_path
        if img_feat_path is not None and img_feat_path != "None" and img_feat_path != "":
            npz_data = np.load(img_feat_path, allow_pickle=True)
            self.img_feat = dict(zip(npz_data['drug_id'], npz_data['feats']))  # dict

        assert len(self.img_feat) == len(self) and len(set(npz_data['drug_id']) - set(self.data.name)) == 0

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def get(self, idx: int) -> Data:
        key = self.data.name[idx]
        img_feat = self.img_feat[key]

        if self.len() == 1:
            self.data.img_feat = img_feat
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            data = copy.copy(self._data_list[idx])
            data.img_feat = img_feat
            return data

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)
        data.img_feat = img_feat

        return data
