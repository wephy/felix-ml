import os
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
# import matplotlib.pyplot as plt
# import logging

import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error
from model import ResNet, SimpleNet, Bottleneck

# ============= Hyperparams ==============
batch_size = 64
epochs = 10
image_size = 128 * 128
seed = 1
num_workers = 1

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
flag_12k = torch.utils.data.Subset(flag_dataset, torch.arange(12000))
train_dataset, test_dataset = random_split(flag_12k, [10000, 2000])
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
# model = ResNet(Bottleneck, [1, 1, 1, 1], train_loader, test_loader, device).to(device)
model = SimpleNet(train_loader, test_loader, device).to(device)
model.optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        model.train_model()
        model.test_model()

        with torch.no_grad():
            for i, (felix_patterns, lattices) in enumerate(test_loader):
                felix_patterns, lattices = felix_patterns.to(device), lattices.to(device)
                felix_patterns, lattices, = (felix_patterns[:8], lattices[:8])  # Batch size >= 8
                prediction_patterns = torch.cat(
                    [model.decode(torch.randn(1, 2).to(device), l).to(device)
                        for l in lattices])
                comparison = torch.cat(
                    [transforms.functional.invert(lattices.view(-1, 1, 128, 128)[:8]).to(device),
                     felix_patterns.view(-1, 1, 128, 128)[:8].to(device),
                     prediction_patterns.view(-1, 1, 128, 128)[:8].to(device),])
                save_image(comparison.to(device), "results/SimpleNet_" + str(model.epoch) + ".png")
                break
