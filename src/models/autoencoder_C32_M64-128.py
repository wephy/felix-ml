import os
import torch
import torchvision.models as models
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.autograd import Variable
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from .components import GDN, MS_SSIM_Loss


class Encoder(nn.Sequential):
    def __init__(self, C=32, M=64):
        super().__init__(
            nn.Conv2d(
                in_channels=1,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            GDN(M),
            nn.Conv2d(
                in_channels=M,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            GDN(M),
            nn.Conv2d(
                in_channels=M,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            GDN(M),
            nn.Conv2d(
                in_channels=M,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            GDN(M),
            nn.Conv2d(
                in_channels=M,
                out_channels=C,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
        )


class Decoder(nn.Sequential):
    def __init__(self, C=32, M=128):
        super().__init__(
            nn.ConvTranspose2d(
                in_channels=C,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(
                in_channels=M,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(
                in_channels=M,
                out_channels=M,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            GDN(M, inverse=True),
            nn.ConvTranspose2d(
                in_channels=M,
                out_channels=1,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        # Encoding components
        self.encoder = Encoder()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, self.embed_dim)
        self.fc_bn2 = nn.BatchNorm1d(self.embed_dim)

        # Decoding components
        self.fc3 = nn.Linear(self.embed_dim, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 32 * 4 * 4)
        self.fc_bn4 = nn.BatchNorm1d(32 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (32, 4, 4))
        self.decoder = Decoder()

        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)  # flatten output of conv

        # Encoding FC layers
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x)))
        return x

    def decode(self, z):
        # Decoding FC layers
        x = self.relu(self.fc_bn3(self.fc3(z)))
        x = self.relu(self.fc_bn4(self.fc4(x)))

        # Decoder
        x = self.unflatten(x)
        x = self.decoder(x)
        x = torch.cat([x, torch.flip(x, [2])], 2)
        x = torch.cat([x, torch.flip(x, [3])], 3)
        return x

    def forward(self, x):
        coded = self.encode(x)
        x_reconst = self.decode(coded)
        return x_reconst


class AELitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.learning_rate = config.learning_rate
        self.embed_dim = config.embed_dim

        self.model = Autoencoder(embed_dim=config.embed_dim)
        self.loss_function = MS_SSIM_Loss(data_range=1.0, channel=1, win_size=5)
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCELoss()

        # for averaging loss across batches
        self.MS_SSIM_train_loss = MeanMetric()
        self.MS_SSIM_val_loss = MeanMetric()
        self.MS_SSIM_test_loss = MeanMetric()

        self.MSE_train_loss = MeanMetric()
        self.MSE_val_loss = MeanMetric()
        self.MSE_test_loss = MeanMetric()

        self.BCE_train_loss = MeanMetric()
        self.BCE_val_loss = MeanMetric()
        self.BCE_test_loss = MeanMetric()

    def model_step(self, real, condition):
        # Pre-training using real images
        fake = self.model(condition)
        loss1 = self.loss_function(fake, real)
        loss2 = self.MSE(fake, real)
        loss3 = self.BCE(fake, real)

        return loss1, loss2, loss3, fake

    def on_train_start(self):
        self.MS_SSIM_val_loss.reset()
        self.MSE_val_loss.reset()
        self.BCE_val_loss.reset()

    def training_step(self, batch, batch_idx):
        real, condition = batch
        loss1, loss2, loss3, fake = self.model_step(real, condition)

        # update and log metrics
        self.MS_SSIM_train_loss(loss1)
        self.MSE_train_loss(loss2)
        self.BCE_train_loss(loss3)
        self.log(
            "train/loss_MS_SSIM", self.MS_SSIM_train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/loss_MSE", self.MSE_train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/loss_BCE", self.BCE_train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss1

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        real, condition = batch
        loss1, loss2, loss3, fake = self.model_step(real, condition)

        # update and log metrics
        self.MS_SSIM_val_loss(loss1)
        self.MSE_val_loss(loss2)
        self.BCE_val_loss(loss3)
        self.log(
            "val/loss_MS_SSIM", self.MS_SSIM_val_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/loss_MSE", self.MSE_val_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/loss_BCE", self.BCE_val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        real, condition = batch
        loss1, loss2, loss3, fake = self.model_step(real, condition)

        # update and log metrics
        self.MS_SSIM_test_loss(loss1)
        self.MSE_test_loss(loss2)
        self.BCE_test_loss(loss3)
        self.log(
            "test/loss_MS_SSIM", self.MS_SSIM_test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/loss_MSE", self.MSE_test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/loss_BCE", self.BCE_test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.learning_rate,
        )
        return {"optimizer": optimizer}
