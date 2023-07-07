from __future__ import print_function

import argparse
import itertools
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error
from vae.bin.basic_vae import VAE

# Arguments
parser = argparse.ArgumentParser(description="VAE FLAG Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 8)",
)
parser.add_argument(
    "--latent-size",
    type=int,
    default=16,
    metavar="N",
    help="input batch size for training (default: 8)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=2,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_built()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

# Load dataset
flag_location = r"D:\\felix-ml\datasets\\flag"
flag_dataset = FLAGDataset(flag_location)
flag_shrunk = torch.utils.data.Subset(flag_dataset, torch.arange(3000))

dataset_len = len(flag_shrunk)
train_dataset, test_dataset = torch.utils.data.random_split(flag_shrunk, [2500, 500])

train_loader = DataLoader(
    dataset=train_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=4,
)
test_loader = DataLoader(
    dataset=test_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=4,
)

model = VAE(args.batch_size, 16384, 16384, args.latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x.view(-1, 16384).clamp(0, 1),
        x.view(-1, 16384).clamp(0, 1),
        reduction="sum",
    )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        lattice = data["input"]
        pattern = data["output"]
        lattice = lattice.to(device)
        pattern = pattern.to(device)

        recon_batch, mu, logvar = model(pattern, lattice)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, pattern, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * args.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item() / len(data),
                )
            )

    print(
        "====> Epoch: {} Average loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        )
    )


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            lattice = data["input"]
            pattern = data["output"]
            lattice = lattice.to(device)
            pattern = pattern.to(device)

            recon_batch, mu, logvar = model(pattern, lattice)
            test_loss += (
                loss_function(recon_batch, pattern, mu, logvar).detach().cpu().numpy()
            )

            # save the first batch input and output of every epoch
            if i == 0:
                num_rows = 8
                comparison = torch.cat(
                    (
                        pattern.view(args.batch_size, 1, 128, 128)[:8],
                        recon_batch.view(args.batch_size, 1, 128, 128)[:8],
                    )
                )
                save_image(
                    comparison.cpu(),
                    f"vae/results/reconstruction_{epoch}.png",
                    nrow=num_rows,
                )

    test_loss /= len(test_loader.dataset)
    print("====> Test set loss: {:.4f}".format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, args.features).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(
        #         sample.view(64, 1, 128, 128),
        #         "vae/results/sample_" + str(epoch) + ".png",
        #     )
