import collections.abc as container_abcs

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Batch

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
                batch = Batch.from_data_list(batch, self.follow_batch)
                if "img_feat" in batch:
                    # assert len(batch.img_feat.shape) == 1  # img_feat 被拉平了
                    # n = batch.num_graphs
                    batch.img_feat = torch.from_numpy(np.stack(batch.img_feat))
                    # batch.img_feat = batch.img_feat.resize(n, 512)
                return batch
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
