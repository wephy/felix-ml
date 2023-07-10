import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import logging

import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error
from model import Model, CVAE

# ============= Hyperparams ==============
batch_size = 64
latent_size = 20
epochs = 10
image_size = 128 * 128
seed = 1
num_workers = 2

# ================ Setup =================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
kwargs = {"num_workers": num_workers, "pin_memory": True}
torch.manual_seed(seed)

# ============= Load dataset =============
flag_location = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "flag"))
flag_dataset = FLAGDataset(flag_location)
flag_12k = torch.utils.data.Subset(flag_dataset, torch.arange(1200))
train_dataset, test_dataset = random_split(flag_12k, [1050, 150])
train_loader = DataLoader(
    dataset=train_dataset,  # the dataset instance
    batch_size=batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    **kwargs,
)
test_loader = DataLoader(
    dataset=test_dataset,  # the dataset instance
    batch_size=batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    **kwargs,
)

# =========== Load CVAE model ============
model = Model(train_loader, test_loader, latent_size=latent_size, device=device)

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        model.train_model()
        model.test_model()

        with torch.no_grad():
            for i, (data, lattice) in enumerate(test_loader):
                if i == 1:
                    break
                data, lattice = data.to(device), lattice.to(device)

                lattice = lattice[0]
                save_image(
                    lattice.view(1, 1, 128, 128),
                    "latent_space_results/lattice_" + str(model.epoch) + ".png",
                )

                felix_pattern = data[0]
                save_image(
                    data.view(1, 1, 128, 128),
                    "latent_space_results/felix_" + str(model.epoch) + ".png",
                )

                prediction_patterns = torch.cat(
                    [
                        model.decode(torch.randn(1, latent_size).to(device), lattice).cpu()
                        for _ in range(64)
                    ]
                )
                save_image(
                    prediction_patterns.view(64, 1, 128, 128),
                    "latent_space_results/predictions_" + str(model.epoch) + ".png",
                )
