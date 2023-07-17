import torch
from torch import nn
import torch.nn.functional as F


class Convolutional(nn.Module):
    def __init__(self, latent_dims=25):
        super(Convolutional, self).__init__()

        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Linear(128 * 128 + 29 * 29 * 7, 256),
            nn.ELU(),
        )
        self.fc_mu = nn.Linear(256, latent_dims)
        self.fc_var = nn.Linear(256, latent_dims)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dims + 29 * 29 * 7, 256),
            nn.ELU(),
            nn.Linear(256, 128 * 128),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        inputs = self.encoder(inputs)
        z_mu = self.fc_mu(inputs)
        z_var = self.fc_var(inputs)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        inputs = self.decoder(inputs)
        return inputs

    def forward(self, x, c):
        batch_size, _ = x.size()
        x = x.view(batch_size, -1)
        c = c.view(batch_size, -1)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
