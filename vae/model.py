import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms


class CVAE(nn.Module):
    def __init__(self, latent_size):
        super(CVAE, self).__init__()
        self.image_size = 128 * 128
        self.feature_size = self.image_size
        self.class_size = self.image_size  # For now using the lattice image (consider using Cif file in the future)
        self.latent_size = latent_size

        # encode
        self.fc1  = nn.Linear(self.feature_size + self.class_size, 400)
        self.fc21 = nn.Linear(400, self.latent_size)
        self.fc22 = nn.Linear(400, self.latent_size)

        # decode
        self.fc3 = nn.Linear(self.latent_size + self.class_size, 400)
        self.fc4 = nn.Linear(400, self.feature_size)

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
        inputs = torch.cat([z, c.view(-1, self.image_size)], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.image_size), c.view(-1, self.image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class Model(CVAE):
    def __init__(self, train_loader, test_loader, latent_size=20, device=torch.device("cuda")):
        super(Model, self).__init__(latent_size=latent_size)
        self.device=device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.image_size), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def test_model(self):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, lattice) in enumerate(self.test_loader):
                data, lattice = data.to(self.device), lattice.to(self.device)
                recon_batch, mu, logvar = self(data, lattice)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()
        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train_model(self):
        self.epoch += 1
        self.train()
        train_loss = 0
        for batch_idx, (data, lattice) in enumerate(self.train_loader):
            data, lattice = data.to(self.device), lattice.to(self.device)
            recon_batch, mu, logvar = self(data, lattice)
            self.optimizer.zero_grad()
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))
