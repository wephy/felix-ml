import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FLAGDataset(Dataset):
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
        # accessing data at particular indexes
        input_img = np.clip(np.load(self.data[idx][0]), 0.0, 1.0)
        output_img = np.clip(np.load(self.data[idx][1]), 0.0, 1.0)

        # pytorch tensors should be returned as tensor is the primary data type pytorch works with
        return [torch.from_numpy(output_img).float().clone().detach(),
                torch.from_numpy(input_img).float().clone().detach()]


if __name__ == "__main__":
    """
    This dataset has the following structure.

    .
    |-- 1/
    |   |-- input.npy
    |   `-- output.npy
    |-- 2/
    |   |-- input.npy
    |   `-- output.npy
    ...
    """
    flag_location = "//storage/disqs/felix-ML/ProjectSpace/VAE_000/Data"

    flag_dataset = FLAGDataset(flag_location)
    flag_dataloader = DataLoader(
        dataset=flag_dataset,  # the dataset instance
        batch_size=64,  # automatic batching
        drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
        shuffle=True,  # shuffles the dataset before every epoch
    )

    print(flag_dataset)
    print(len(flag_dataset))
    print(flag_dataset[3])
