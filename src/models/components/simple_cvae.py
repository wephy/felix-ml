import torch
from torch import nn


class SimpleCVAE(nn.Module):
    def __init__(
        self,
        input_size: int = 16384,
        condition_size: int = 16384,
        latent_size: int = 10,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        lin4_size: int = 256,
        lin5_size: int = 256,
        lin6_size: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size + condition_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
        )
        self.fc31 = nn.Linear(lin3_size, latent_size)
        self.fc32 = nn.Linear(lin3_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_size, lin4_size),
            nn.BatchNorm1d(lin4_size),
            nn.ReLU(),
            nn.Linear(lin4_size, lin5_size),
            nn.BatchNorm1d(lin5_size),
            nn.ReLU(),
            nn.Linear(lin5_size, lin6_size),
            nn.BatchNorm1d(lin6_size),
            nn.ReLU(),
            nn.Linear(lin6_size, input_size)
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h3 = self.encoder(inputs)
        z_mu = self.fc31(h3)
        z_var = self.fc32(h3)
        return z_mu, z_var

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h6 = self.decoder(inputs)
        return self.sigmoid(h6)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c):
        batch_size, channels, width, height = x.size()

        x = x.view(batch_size, -1)
        c = c.view(batch_size, -1)

        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
