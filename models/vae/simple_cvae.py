import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F
from ..types_ import *


class SimpleCVAE(pl.LightningModule):
    def __init__(self,
                 conditional_channels: int,
                 latent_dim: int,
                 img_size: int = 128,
                 **kwargs) -> None:
        super(SimpleCVAE, self).__init__()
        
        self.features = img_size * img_size
        self.latent_dim = latent_dim

        self.embed_class = nn.Linear(conditional_channels, img_size * img_size)
        self.embed_data = nn.Conv2d(self.features, self.features, kernel_size=1)

        # encode
        self.fc1  = nn.Linear(self.features + conditional_channels, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)

        # decode
        self.fc3 = nn.Linear(latent_dim + conditional_channels, 512)
        self.fc4 = nn.Linear(512, self.features)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, input: Tensor) -> List[Tensor]:
        h1 = self.elu(self.fc1(input))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return [z_mu, z_var]

    def decode(self, z: Tensor) -> Tensor:
        h3 = self.elu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, y: Tensor) -> List[Tensor]:
        embedded_condition = self.embed_class(y)
        embedded_condition = embedded_condition.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        
        x = torch.cat([embedded_input, embedded_condition], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim = 1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        BCE = F.binary_cross_entropy(recons, input.view(-1, self.image_size), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def sample(self,
               num_samples:int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, y)[0]
