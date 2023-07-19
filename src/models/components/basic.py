import torch
from torch import nn
import torch.nn.functional as F


class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()

        self.image_size = 128 * 128
        self.latent_dims = 20
        # self.class_size = self.image_size
        self.class_size = 29 * 29 * 7
        
        # encode
        self.fc1  = nn.Linear(self.image_size + self.class_size, 400)
        self.fc2  = nn.Linear(400, 400)
        self.fc21 = nn.Linear(400, self.latent_dims)
        self.fc22 = nn.Linear(400, self.latent_dims)

        # decode
        self.fc3 = nn.Linear(self.latent_dims + self.class_size, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, self.image_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc2(self.elu(self.fc1(inputs))))
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
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc4(self.elu(self.fc3(inputs))))
        return self.sigmoid(self.fc5(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar







    #     self.encoder = nn.Sequential(
    #         nn.Linear(128 * 128 + 29 * 29 * 7, 512),
    #         nn.ELU(),
    #     )
    #     self.fc_mu = nn.Linear(512, latent_dims)
    #     self.fc_var = nn.Linear(512, latent_dims)

    #     self.decoder = nn.Sequential(
    #         nn.Linear(latent_dims + 29 * 29 * 7, 512),
    #         nn.ELU(),
    #         nn.Linear(512, 128 * 128),
    #         nn.Sigmoid()
    #     )

    # def encode(self, x, c):
    #     inputs = torch.cat([x, c], 1)
    #     inputs = self.encoder(inputs)
    #     z_mu = self.fc_mu(inputs)
    #     z_var = self.fc_var(inputs)
    #     return z_mu, z_var

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps*std

    # def decode(self, z, c):
    #     inputs = torch.cat([z, c], 1)
    #     inputs = self.decoder(inputs)
    #     return inputs

    # def forward(self, x, c):
    #     # batch_size, _ = x.size()
    #     # x = x.view(batch_size, -1)
    #     # c = c.view(batch_size, -1)
    #     mu, logvar = self.encode(x, c)
    #     z = self.reparameterize(mu, logvar)
    #     return self.decode(z, c), mu, logvar
