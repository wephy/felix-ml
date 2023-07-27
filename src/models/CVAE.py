import os
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from skimage.metrics import structural_similarity


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
    def __init__(self, latent_dim):
        super().__init__()

        self.input_dim = (2, 128, 128)
        self.latent_dim = latent_dim
        self.n_channels = 2

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x, c, output_layer_levels=None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        # output = ModelOutput()

        max_depth = self.depth

        # if output_layer_levels is not None:

        #     assert all(
        #         self.depth >= levels > 0 or levels == -1
        #         for levels in output_layer_levels
        #     ), (
        #         f"Cannot output layer deeper than depth ({self.depth})."
        #         f"Got ({output_layer_levels})."
        #     )

        #     if -1 in output_layer_levels:
        #         max_depth = self.depth
        #     else:
        #         max_depth = max(output_layer_levels)

        out = torch.cat([x, c], 1)

        for i in range(max_depth):
            out = self.layers[i](out)

        return (
            self.embedding(out.reshape(x.shape[0], -1)),
            self.log_var(out.reshape(x.shape[0], -1)),
        )


class Decoder_ResNet_AE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.input_dim = (1, 128, 128)
        self.latent_dim = latent_dim + (128 * 128)
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(self.latent_dim, 64 * 4 * 4))

        layers.append(nn.ConvTranspose2d(64, 128, 3, 2, padding=1, output_padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
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

    def forward(self, z, c, output_layer_levels=None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        # output = ModelOutput()

        max_depth = self.depth

        # if output_layer_levels is not None:

        #     assert all(
        #         self.depth >= levels > 0 or levels == -1
        #         for levels in output_layer_levels
        #     ), (
        #         f"Cannot output layer deeper than depth ({self.depth})."
        #         f"Got ({output_layer_levels})"
        #     )

        #     if -1 in output_layer_levels:
        #         max_depth = self.depth
        #     else:
        #         max_depth = max(output_layer_levels)

        bs = c.size(0)
        out = torch.cat([z, c.view(bs, -1)], 1)

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 64, 4, 4)

            # if output_layer_levels is not None:
            #     if i + 1 in output_layer_levels:
            #         output[f"reconstruction_layer_{i+1}"] = out

            # if i + 1 == self.depth:
            #     output["reconstruction"] = out

        return out


class CVAELitModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.learning_rate = model.learning_rate
        self.loss_function = model.loss_function
        self.latent_dim = model.latent_dim

        self.encoder = Encoder_ResNet_VAE(self.latent_dim)
        self.decoder = Decoder_ResNet_AE(self.latent_dim)

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="binary")
        # self.val_acc = Accuracy(task="binary")
        # self.test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def model_step(self, real, condition):
        z = self.encoder(real, condition)[0]
        fake = self.decoder(z, condition)
        loss, recon_loss, embedding_loss = self.loss_function(fake, real, z)

        return loss, fake

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(
        #     [
        #         structural_similarity(
        #             real[i].view(128, 128).cpu().numpy(),
        #             fake[i].view(128, 128).cpu().numpy(),
        #             data_range=1
        #         )
        #         for i in range(real.size(0))
        #     ],
        #     [1 for i in range(real.size(0))]
        # )
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        # )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(
        #     [
        #         structural_similarity(
        #             real[i].view(128, 128).cpu().numpy(),
        #             fake[i].view(128, 128).cpu().numpy(),
        #             data_range=1
        #         )
        #         for i in range(real.size(0))
        #     ],
        #     [1 for i in range(real.size(0))]
        # )
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        real, condition = batch
        loss, fake = self.model_step(real, condition)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(
        #     [
        #         structural_similarity(
        #             real[i].view(128, 128).cpu().numpy(),
        #             fake[i].view(128, 128).cpu().numpy(),
        #             data_range=1
        #         )
        #         for i in range(real.size(0))
        #     ],
        #     [1 for i in range(real.size(0))]
        # )

        self.log(
            "test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        # self.log("test/acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

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
