import torch
from torch import nn
import torch.nn.functional as F


class Convolutional(nn.Module):
    def __init__(self, latent_dims=25):
        super(Convolutional, self).__init__()

        self.latent_dims = latent_dims

        self.input_encoder = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        self.condition_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(2 * 16 * 8 * 8, 256)
        )
        self.fc_mu = nn.Linear(256, latent_dims)
        self.fc_var = nn.Linear(256, latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(16 * 8 * 8 + latent_dims, 16 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (16, 8, 8)),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        # inputs = torch.cat([x, c], 2)
        x = self.input_encoder(x).view(x.size(0), -1)
        c = self.condition_encoder(c).view(x.size(0), -1)
        h = self.encoder(torch.cat([x, c], 1))
        z_mu = self.fc_mu(h)
        z_var = self.fc_var(h)
        return z_mu, z_var

    def decode(self, z, c):
        # inputs = torch.cat([z, c], 1)
        c = self.condition_encoder(c).view(c.size(0), -1)
        return self.decoder(torch.cat([z, c], 1)).view(c.size(0), -1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c):
        batch_size, width, height = x.size()

        x = x.view(batch_size, 1, 128, 128)
        c = c.view(batch_size, 1, 128, 128)

        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        h = self.decode(z, c)
        return h, mu, logvar
