from __future__ import print_function

import argparse
import itertools
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error

# Arguments
parser = argparse.ArgumentParser(description='VAE FLAG Example')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Load dataset
flag_location = "//home/physics/phusnq/felix-ml/datasets/flag"
flag_dataset = FLAGDataset(flag_location)
flag_shrunk = torch.utils.data.Subset(flag_dataset, torch.arange(800))

dataset_len = len(flag_shrunk)
train_dataset, test_dataset= torch.utils.data.random_split(flag_shrunk, [400, 100])

train_loader = DataLoader(
    dataset=train_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=0,
)
test_loader = DataLoader(
    dataset=test_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=0,
)

print(len(train_loader))
print(len(test_loader))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encode
        # self.conv1 = nn.Conv2d(1, 20, kernel_size=(6, 6), stride=2, padding=1)
        # self.conv2 = nn.Conv2d(20, 20, kernel_size=(6, 6), stride=2, padding=1)
        # self.fc1 = nn.Linear(20480, 25)
        self.main_encode = nn.Sequential(
            nn.Unflatten(1, (128, 128)),
            nn.Conv2d(args.batch_size, 20, kernel_size=(6, 6), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(2, 2), stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(1, 2),
            nn.Flatten(0, 1),
            nn.Linear(20480, 25),
        )
        self.fc21 = nn.Linear(25, 25)
        self.fc22 = nn.Linear(25, 25)

        # decode
        self.main_decode = nn.Sequential(
            nn.Linear(25, 40960),
            nn.Unflatten(-1, (40, 32, 32)),
            nn.ConvTranspose2d(40, 80, kernel_size=(6, 6), stride=2, padding=1),
            nn.ReLU(),
            # nn.ConvTranspose2d(80, 80, kernel_size=(8, 8), stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(80, args.batch_size, kernel_size=(2, 2), stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(1, 2),
        )
        # self.fc3 = nn.Linear(128, 8192)
        # self.fc4 = nn.Linear(16384, 16384)

    def encode(self, x):
        h1 = self.main_encode(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.main_decode(z)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 16384))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 16384), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data["output"]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
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
        for i, data in enumerate(test_loader):
            data = data["output"]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(args.batch_size, 1, 128, 128),
                                      recon_batch.view(args.batch_size, 1, 128, 128)[:n]])
                save_image(comparison.cpu(),
                         'vae/results/reconstruction_' + str(epoch) + "_" + str(i) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(8, 25).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 128, 128),
                       'vae/results/sample_' + str(epoch) + '.png')
