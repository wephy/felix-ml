from typing import Any, Dict, Optional, Tuple

import os
import torch

# from skimage import io
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.transforms import transforms
import numpy as np


class FDP(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.ICSD_codes = os.listdir(self.data_dir)

    def __len__(self):
        return len(self.ICSD_codes)

    def __getitem__(self, idx):
        ICSD_code = idx

        structure = np.load(
            os.path.join(self.data_dir, ICSD_code, ICSD_code + "_structure.npy")
        ).astype(np.float32)
        pattern = np.clip(
            np.load(os.path.join(self.data_dir, ICSD_code, ICSD_code + "_+0+0+0.npy")),
            0.0,
            1.0,
        ).astype(np.float32)

        if self.transform:
            structure = self.transform(structure)
            pattern = self.transform(pattern)

        return (pattern, structure)


class FDPDataModule(LightningDataModule):
    """Example of LightningDataModule for CVAE dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir,
        split,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.data_dir = data_dir
        self.split = split

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = FDP(self.data_dir, transform=self.transforms)
            with open(os.path.join(self.split, "train")) as file:
                self.data_train = Subset(dataset, file.read().split("\n")[:-1])
            with open(os.path.join(self.split, "val")) as file:
                self.data_val = Subset(dataset, file.read().split("\n")[:-1])
            with open(os.path.join(self.split, "test")) as file:
                self.data_test = Subset(dataset, file.read().split("\n")[:-1])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    pass
