import os
import torch
import lightning.pytorch as pl
from typing import List, Optional, Sequence, Union, Any, Callable
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np


class LatticeDiffraction(Dataset):
    def __init__(self, data_root):
        self.data = []
        for sample in os.listdir(data_root):
            sample_folder = os.path.join(data_root, sample)
            input_img = os.path.join(sample_folder, "Input.npy")
            output_img = os.path.join(sample_folder, "Output.npy")
            self.data.append((input_img, output_img))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_img = np.clip(np.load(self.data[idx][0]), 0.0, 1.0)
        output_img = np.clip(np.load(self.data[idx][1]), 0.0, 1.0)

        return [
            torch.from_numpy(output_img).float().clone().detach(),
            torch.from_numpy(input_img).float().clone().detach(),
        ]


class FelixDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = LatticeDiffraction(self.data_dir)
        self.train_dataset, self.val_dataset = random_split(dataset, [0.9, 0.1])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
        )
