import torch
from torch import nn


class SimpleCVAE(nn.Module):
    def __init__(
        self,
        input_size: int = 16384,
        condition_size: int = 16384,
        latent_size: int = 10,
        encoder_lin1_size: int = 512,
        decoder_lin1_size: int = 512,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size + condition_size, encoder_lin1_size),
            nn.ELU(),
        )
        self.fc_mu = nn.Linear(encoder_lin1_size, latent_size)
        self.fc_var = nn.Linear(encoder_lin1_size, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_size, decoder_lin1_size),
            nn.ELU(),
            nn.Linear(decoder_lin1_size, input_size),
            nn.Sigmoid(),
        )

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h2 = self.encoder(inputs)
        z_mu = self.fc_mu(h2)
        z_var = self.fc_var(h2)
        return z_mu, z_var

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        return self.decoder(inputs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c):
        batch_size, width, height = x.size()

        x = x.view(batch_size, -1)
        c = c.view(batch_size, -1)

        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
