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
    def __init__(self, C=32, M=128):
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
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.embed_dim)
        self.fc_bn3 = nn.BatchNorm1d(self.embed_dim)

        # Decoding components
        self.fc4 = nn.Linear(self.embed_dim, 512)
        self.fc_bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 512)
        self.fc_bn5 = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 32 * 4 * 4)
        self.fc_bn6 = nn.BatchNorm1d(32 * 4 * 4)
        self.relu = nn.ReLU(inplace=True) 
        self.unflatten = nn.Unflatten(1, (32, 4, 4))
        self.decoder = Decoder()

    def encode(self, x):
        # Encoder
        x = self.encoder(x)
        x = self.flatten(x)  # flatten output of conv

        # Encoding FC layers
        x = self.relu(self.fc_bn1(self.fc1(x)))
        x = self.relu(self.fc_bn2(self.fc2(x)))
        x = self.relu(self.fc_bn3(self.fc3(x)))
        return x

    def decode(self, z):
        # Decoding FC layers
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x)))
        x = self.relu(self.fc_bn6(self.fc6(x)))

        # Decoder
        x = self.unflatten(x)
        x = self.decoder(x)
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
        self.loss_function = MS_SSIM_Loss(data_range=1.0, channel=1, win_size=7)

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
        fake = self.model(condition)
        loss = self.loss_function(fake, real)

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
        pass

    def validation_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.learning_rate,
        )
        return {"optimizer": optimizer}