import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.Baselineloder import (get_Baseline, Baseline, vizBaseline)
from dataloaders.MMGNloder import (get_MMGN, MMGN, vizMMGN)


class DataModule(pl.LightningDataModule):
    def __init__(self, 
        data_dir: str = "data/gst_data.npz",
        model_name: str = "ResMLP",
        pre_method: list = ["zscore", "zscore"],
        reduce_dim: list = [],
        task: str = "task1",
        sampling_rate: list = [0., 0.],
        n_train_val: list = [128, 32],
        b_train_val: list = [16, 4],
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.model_name = model_name
        self.pre_method = pre_method
        self.reduce_dim = reduce_dim
        self.task = task
        self.sampling_rate = sampling_rate
        self.n_train_val = n_train_val
        self.n_train, self.n_val = n_train_val
        self.b_train, self.b_val = b_train_val
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if self.model_name in {"ResMLP", "SIREN", "FFN_P", "FFN_G"}:
            grid_lat, grid_lon, time_t, temprature, indices, normalizer = get_Baseline(self.data_dir, self.n_train, self.pre_method, self.reduce_dim, self.task, self.sampling_rate)
            self.train_data = Baseline(temprature, grid_lat, grid_lon, time_t, indices)
            self.val_data = vizBaseline(temprature[:self.n_val,:,:], grid_lat, grid_lon, time_t[:self.n_val])
            self.test_data = vizBaseline(temprature, grid_lat, grid_lon, time_t)
        elif self.model_name == "MMGN":
            grid_lat, grid_lon, temprature, indices, normalizer = get_MMGN(self.data_dir, self.n_train, self.pre_method, self.reduce_dim, self.task, self.sampling_rate)
            self.train_data = MMGN(temprature, grid_lat, grid_lon, indices)
            self.val_data = vizMMGN(temprature[:self.n_val,:,:], grid_lat, grid_lon)
            self.test_data = vizMMGN(temprature, grid_lat, grid_lon)
        del grid_lat, grid_lon, temprature, indices
        return normalizer

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.b_val, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.b_train, num_workers=self.num_workers, pin_memory=True, shuffle=False)