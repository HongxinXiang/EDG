from os.path import join

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from visnet_for_EDG.datasets import *
from visnet_for_EDG.utils import MissingLabelException, make_splits
import os.path as osp
import pandas as pd
import numpy as np


class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(hparams, "__dict__") else self.hparams.update(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        
        assert hasattr(self, f"_prepare_{self.hparams['dataset']}_dataset"), f"Dataset {self.hparams['dataset']} not defined"

        if self.hparams['dataset'] == "QM9":
            dataset_factory = lambda t: getattr(self, f"_prepare_{t}_dataset")(split=self.hparams["split_qm9"])  # load dataset
        else:
            dataset_factory = lambda t: getattr(self, f"_prepare_{t}_dataset")(split=self.hparams["split_id"])  # load dataset
        self.idx_train, self.idx_val, self.idx_test = dataset_factory(self.hparams["dataset"])
            
        print(f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}")
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        delta = 1 if self.hparams['reload'] == 1 else 2
        if (
            len(self.test_dataset) > 0
            and (self.trainer.current_epoch + delta) % self.hparams["test_interval"] == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (store_dataloader and not self.hparams["reload"])
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl
    
    @rank_zero_only
    def _standardize(self):
        def get_label(batch, atomref):
            if batch.y is None:
                raise MissingLabelException()

            if atomref is None:
                return batch.y.clone()

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False), 
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            ys = torch.cat([get_label(batch, atomref) for batch in data])
        except MissingLabelException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return None

        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
    
    # def _prepare_Chignolin_dataset(self):
    #
    #     self.dataset = Chignolin(root=self.hparams["dataset_root"])
    #     train_size = self.hparams["train_size"]
    #     val_size = self.hparams["val_size"]
    #
    #     idx_train, idx_val, idx_test = make_splits(
    #         len(self.dataset),
    #         train_size,
    #         val_size,
    #         None,
    #         self.hparams["seed"],
    #         join(self.hparams["log_dir"], "splits.npz"),
    #         self.hparams["splits"],
    #     )
    #
    #     return idx_train, idx_val, idx_test
    #
    # def _prepare_MD17_dataset(self):
    #
    #     self.dataset = MD17(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
    #     train_size = self.hparams["train_size"]
    #     val_size = self.hparams["val_size"]
    #
    #     idx_train, idx_val, idx_test = make_splits(
    #         len(self.dataset),
    #         train_size,
    #         val_size,
    #         None,
    #         self.hparams["seed"],
    #         join(self.hparams["log_dir"], "splits.npz"),
    #         self.hparams["splits"],
    #     )
    #
    #     return idx_train, idx_val, idx_test
    #
    # def _prepare_MD22_dataset(self):
    #
    #     self.dataset = MD22(root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"])
    #     train_val_size = self.dataset.molecule_splits[self.hparams["dataset_arg"]]
    #     train_size = round(train_val_size * 0.95)
    #     val_size = train_val_size - train_size
    #
    #     idx_train, idx_val, idx_test = make_splits(
    #         len(self.dataset),
    #         train_size,
    #         val_size,
    #         None,
    #         self.hparams["seed"],
    #         join(self.hparams["log_dir"], "splits.npz"),
    #         self.hparams["splits"],
    #     )
    #
    #     return idx_train, idx_val, idx_test
    #
    # def _prepare_Molecule3D_dataset(self):
    #
    #     self.dataset = Molecule3D(root=self.hparams["dataset_root"])
    #     split_dict = self.dataset.get_idx_split(self.hparams['split_mode'])
    #     idx_train = split_dict['train']
    #     idx_val = split_dict['valid']
    #     idx_test = split_dict['test']
    #
    #     return idx_train, idx_val, idx_test

    def split_qm9_random_customized_01(self, dataset, task_idx=None, null_value=0, seed=0, smiles_list=None, logger=None):
        log = print if logger is None else logger.info
        if task_idx is not None:
            # filter based on null values in task_idx
            # get task array
            y_task = np.array([data.y[task_idx].item() for data in dataset])
            non_null = (
                    y_task != null_value
            )  # boolean array that correspond to non null values
            idx_array = np.where(non_null)[0]
            dataset = dataset[torch.tensor(idx_array)]  # examples containing non
            # null labels in the specified task_idx
        else:
            pass

        num_mols = len(dataset)
        np.random.seed(seed)
        all_idx = np.random.permutation(num_mols)

        Nmols = 133885 - 3054
        Ntrain = 110000
        Nvalid = 10000
        Ntest = Nmols - (Ntrain + Nvalid)

        train_idx = all_idx[:Ntrain]
        valid_idx = all_idx[Ntrain: Ntrain + Nvalid]
        test_idx = all_idx[Ntrain + Nvalid:]

        log(f"train_idx: {train_idx}")
        log(f"valid_idx: {valid_idx}")
        log(f"test_idx: {test_idx}")
        # np.savez("customized_01", train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(valid_idx).intersection(set(test_idx))) == 0
        assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

        return train_idx, valid_idx, test_idx
    def _prepare_QM9_dataset(self, split):
        assert split in ["customized_01", "default"]
        self.dataset = QM9(root=self.hparams["dataset_root"], img_feat_path=self.hparams["img_feat_path"], dataset_arg=self.hparams["dataset_arg"])

        if split == "customized_01":  # follow Geom3D
            idx_train, idx_val, idx_test = self.split_qm9_random_customized_01(self.dataset, task_idx=None, null_value=0, seed=self.hparams['seed'], logger=None)
        elif split == "default":
            train_size = self.hparams["train_size"]
            val_size = self.hparams["val_size"]

            idx_train, idx_val, idx_test = make_splits(
                len(self.dataset),
                train_size,
                val_size,
                None,
                self.hparams["seed"],
                join(self.hparams["log_dir"], "splits.npz"),
                self.hparams["splits"],
            )
        else:
            raise ValueError

        return idx_train, idx_val, idx_test
    
    def _prepare_rMD17_dataset(self, split="01"):
        assert split in ["01", "02", "03", "04", "05", "all"]

        self.dataset = rMD17(root=self.hparams["dataset_root"], img_feat_path=self.hparams["img_feat_path"], dataset_arg=self.hparams["dataset_arg"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]

        if split == "all":
            idx_train, idx_val, idx_test = make_splits(
                len(self.dataset),
                train_size,
                val_size,
                None,
                self.hparams["seed"],
                join(self.hparams["log_dir"], "splits.npz"),
                self.hparams["splits"],
            )
        else:
            dataset_root = self.hparams['dataset_root']
            dataset = self.hparams['dataset_arg']
            split = self.hparams["split_id"]

            train_file = osp.join(f"{dataset_root}/{dataset}/raw/rmd17", f"splits/index_train_{split}.csv")
            test_file = osp.join(f"{dataset_root}/{dataset}/raw/rmd17", f"splits/index_test_{split}.csv")

            train_csv = pd.read_csv(train_file, header=None)
            train_idx_list = train_csv.values.squeeze()
            assert len(train_idx_list) == 1000

            test_csv = pd.read_csv(test_file, header=None)
            test_idx_list = test_csv.values.squeeze()
            assert len(test_idx_list) == 1000

            idx_train = torch.tensor(train_idx_list[:950])
            idx_val = torch.tensor(train_idx_list[950:])
            idx_test = torch.tensor(test_idx_list)

        return idx_train, idx_val, idx_test
