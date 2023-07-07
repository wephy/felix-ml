import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error

# etup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} 

# hyper params
batch_size = 64
latent_size = 20
epochs = 10

flag_location = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "flag"))
flag_dataset = FLAGDataset(flag_location)
flag_shrunk = torch.utils.data.Subset(flag_dataset, torch.arange(5000))

train_dataset, test_dataset = random_split(flag_shrunk, [4000, 1000])
train_loader = DataLoader(
    dataset=train_dataset,  # the dataset instance
    batch_size=batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=1,
)
test_loader = DataLoader(
    dataset=test_dataset,  # the dataset instance
    batch_size=batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=1,
)


# def one_hot(lattice, class_size):
#     targets = torch.zeros(lattice.size(0), class_size)
#     for i, label in enumerate(lattice):
#         targets[i, label] = 1
#     return targets.to(device)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c.view(-1, 128*128)], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 128*128), c.view(-1, 128*128))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# create a CVAE model
model = CVAE(128*128, latent_size, 128*128).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 128*128), reduction='sum')
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
            if i == 0:
                n = min(data.size(0), 5)
                comparison = torch.cat([data.view(-1, 1, 128, 128)[:n],
                                      recon_batch.view(-1, 1, 128, 128)[:n]])
                save_image(comparison.cpu(),
                         'vae/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            for i, (data, lattice) in enumerate(test_loader):
                if i == 1:
                    break

                data, lattice = data.to(device), lattice.to(device)
                lattices = lattice[:8]
                felix_patterns = data[:8]

                sample = torch.randn(1, 20).to(device)
                prediction_patterns = torch.cat([model.decode(sample, l).to(device) for l in lattices])

                comparison = torch.cat([lattices.view(-1, 1, 128, 128)[:8],
                                        felix_patterns.view(-1, 1, 128, 128)[:8],
                                        prediction_patterns.view(-1, 1, 128, 128)[:8]])

                save_image(comparison.cpu(),
                        'vae/results/sample_' + str(epoch) + '.png')
