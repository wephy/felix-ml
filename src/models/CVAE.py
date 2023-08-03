import os
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
import torchvision


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Encoder_ResNet_VAE(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.input_dim = (1, 128, 128)
        self.latent_dims = latent_dims
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, latent_dims)
        self.log_var = nn.Linear(128 * 4 * 4, latent_dims)

    def forward(self, c, output_layer_levels=None):
        max_depth = self.depth

        # out = torch.cat([x, c], 1)
        out = c

        for i in range(max_depth):
            out = self.layers[i](out)

        return (
            self.embedding(out.reshape(c.shape[0], -1)),
            self.log_var(out.reshape(c.shape[0], -1)),
        )


class Decoder_ResNet_AE(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.input_dim = (1, 128, 128)
        self.latent_dims = latent_dims
        self.n_channels = 1

        # self.condition_resize = torchvision.transforms.Resize((16, 16), antialias=True)

        layers = nn.ModuleList()

        layers.append(nn.Linear(self.latent_dims, 128 * 4 * 4))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
                # ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 128, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z):
        # bs = c.size(0)
        # c = self.condition_resize(c)
        # out = torch.cat([z, c.view(bs, -1)], 1)
        out = z

        for i in range(self.depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)

        return out


class CVAELitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.learning_rate = config.learning_rate
        self.loss_function = config.loss_function
        self.latent_dims = config.latent_dims

        self.encoder = Encoder_ResNet_VAE(self.latent_dims)
        self.decoder = Decoder_ResNet_AE(self.latent_dims)

        self.epoch = 0

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="binary")
        # self.val_acc = Accuracy(task="binary")
        # self.test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def model_step(self, real, condition):
        # Pre-training using real images
        z = self.encoder(condition)[0]
        fake = self.decoder(z)
        loss, recon_loss, embedding_loss = self.loss_function(fake, real, z)

        return loss, fake

    def on_train_start(self):
        self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()

    def training_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        self.epoch += 1

    def validation_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self):
        # torch.save(self.net.state_dict(), "saved_models/PixelCNNAE.pt")
        pass

    def configure_optimizers(self):
        params = []
        for m in [self.encoder, self.decoder]:
            params.extend(m.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
        )
        return {"optimizer": optimizer}
