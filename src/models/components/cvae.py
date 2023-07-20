import torch
from torch import nn
import torch.nn.functional as F


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3):
        super(ResDown, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_out // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)

        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 2, kernel_size // 2)

        self.act_fnc = nn.ELU()

    def forward(self, x):
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return self.act_fnc(self.bn2(x + skip))


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2):
        super(ResUp, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_in // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channel_in // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channel_out, eps=1e-4)
        self.conv3 = nn.Conv2d(channel_in, channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.up_nn(x)
        skip = self.conv3(x)
        x = self.act_fnc(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return self.act_fnc(self.bn2(x + skip))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        ch = 16
        latent_channels = 32
        condition_size = 29 * 29 * 7
        self.latent_dims = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 7, 1, 3),
            ResDown(ch, 2 * ch),
            ResDown(2 * ch, 4 * ch),
            ResDown(4 * ch, 8 * ch),
            ResDown(8 * ch, 16 * ch),
            nn.Conv2d(16 * ch, 16 * ch, 4),
            nn.Conv2d(16 * ch, 16 * ch, 5),
        )
        self.encoder_combine = nn.Sequential(
            nn.Linear(256 + condition_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(256, latent_channels),
        )
        self.fc_var = nn.Sequential(
            nn.Linear(256, latent_channels),
        )
        self.decoder_combine = nn.Sequential(
            nn.Linear(latent_channels + condition_size, 256),
            nn.ELU(),
            nn.Linear(256, latent_channels),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, 5),
            nn.ConvTranspose2d(latent_channels, ch * 16, 4, 1),
            ResUp(ch * 16, ch * 8),
            ResUp(ch * 8, ch * 4),
            ResUp(ch * 4, ch * 2),
            ResUp(ch * 2, ch),
            nn.Conv2d(ch, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, trues, conditions):
        batch_size = trues.size(0)

        x = self.encoder(trues)
        x = self.encoder_combine(torch.cat([x.view(batch_size, -1), conditions], 1))
        mu, var = self.fc_mu(x), self.fc_var(x)
        return mu, var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, conditions):
        batch_size = z.size(0)
        z = self.decoder_combine(torch.cat([z.view(batch_size, -1), conditions], 1))
        x = self.decoder(z.view(batch_size, -1, 1, 1))
        return x

    def forward(self, trues, conditions):
        mu, logvar = self.encode(trues, conditions)
        x = self.decode(self.reparameterize(mu, logvar), conditions)#.view(batch_size, 128, 128)
        return x, mu, logvar
