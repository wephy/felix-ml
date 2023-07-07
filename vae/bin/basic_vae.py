import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, batch_size, feature_size, class_size, latent_size):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.class_size = class_size

        # encoder
        self.fc1 = nn.Linear(in_features=feature_size + class_size, out_features=512)
        # self.fc2 = nn.Linear(in_features=4096, out_features=512)
        self.fc31 = nn.Linear(in_features=512, out_features=latent_size)
        self.fc32 = nn.Linear(in_features=512, out_features=latent_size)

        # decoder
        self.fc4 = nn.Linear(in_features=latent_size + class_size, out_features=512)
        # self.fc5 = nn.Linear(in_features=512, out_features=4096)
        self.fc6 = nn.Linear(in_features=512, out_features=feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):  # Q(z|x, c)
        """
        x: (bs, feature_size)
        c: (bs, class_size)
        """
        inputs = torch.cat([x, c], 1)  # (bs, feature_size+class_size)
        # h1 = self.elu(self.fc2(self.elu(self.fc1(inputs.view(-1, self.feature_size + self.class_size)))))
        h1 = self.fc1(inputs.view(-1, self.feature_size + self.class_size))
        z_mu = self.fc31(h1)
        z_var = self.fc32(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        """
        z: (bs, latent_size)
        c: (bs, class_size)
        """
        inputs = torch.cat(
            [z.view(-1, self.latent_size), c.view(-1, self.class_size)], 1
        )  # (bs, latent_size+class_size)
        # h3 = self.elu(self.fc5(self.elu(self.fc4(inputs))))
        h3 = self.elu(self.fc4(inputs))
        return self.sigmoid(self.fc6(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 128, 128), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
