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
from model import CVAE

# ============= Hyperparams ==============
batch_size = 64
latent_size = 20
epochs = 10
image_size = 128 * 128
seed = 1
num_workers = 1

# ================ Setup =================
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
kwargs = {'num_workers': num_workers, 'pin_memory': True} 
torch.manual_seed(seed)

# ============= Load dataset =============
flag_location = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "flag"))
flag_dataset = FLAGDataset(flag_location)
flag_12k = torch.utils.data.Subset(flag_dataset, torch.arange(12000))
train_dataset, test_dataset = random_split(flag_12k, [10500, 1500])
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
model = CVAE(image_size, latent_size, image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, image_size), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, lattice) in enumerate(train_loader):
        data, lattice = data.to(device), lattice.to(device)
        recon_batch, mu, logvar = model(data, lattice)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, lattice) in enumerate(test_loader):
            data, lattice = data.to(device), lattice.to(device)
            recon_batch, mu, logvar = model(data, lattice)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
            # if i == 0:
            #     n = min(data.size(0), 5)
            #     comparison = torch.cat([data.view(-1, 1, 128, 128)[:n],
            #                           recon_batch.view(-1, 1, 128, 128)[:n]])
            #     save_image(comparison.cpu(),
    #                      'vae/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    logging.basicConfig(filename='results.log')

    for epoch in range(1, epochs + 1):
            train(epoch)
            test(epoch)
            with torch.no_grad():
                pass
            # for i, (data, lattice) in enumerate(test_loader):
            #     if i == 1:
            #         break

            #     data, lattice = data.to(device), lattice.to(device)
            #     lattices = lattice[:8]
            #     felix_patterns = data[:8]

            #     sample = torch.randn(1, 20).to(device)
            #     prediction_patterns = torch.cat([model.decode(sample, l).to(device) for l in lattices])

            #     comparison = torch.cat([lattices.view(-1, 1, 128, 128)[:8],
            #                             felix_patterns.view(-1, 1, 128, 128)[:8],
            #                             prediction_patterns.view(-1, 1, 128, 128)[:8]])

            #     save_image(comparison.cpu(),
            #             'vae/results/sample_' + str(epoch) + '.png')
