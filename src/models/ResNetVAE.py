import os
import torch
import torchvision.models as models
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.autograd import Variable
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
#                                stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, 512)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [1, 1, 1, 1])


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.groupnorm_1 = nn.BatchNorm2d(in_channels)
#         self.conv_1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)

#         self.groupnorm_2 = nn.BatchNorm2d(out_channels)
#         self.conv_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)

#         if in_channels == out_channels:
#             self.residual_layer = nn.Identity()
#         else:
#             self.residual_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, padding=0)

#     def forward(self, x):
#         residue = x

#         x = self.groupnorm_1(x)
#         x = F.silu(x)
#         x = self.conv_1(x)

#         x = self.groupnorm_2(x)
#         x = F.silu(x)
#         x = self.conv_2(x)

#         return x + self.residual_layer(residue)


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(1, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),  # b, 64, 64, 64
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),  # b, 64, 32, 32
            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),  # b, 128, 16, 16
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),  # b, 256, 8, 8
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),  # b, 512, 4, 4
        )


class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Unflatten(1, (128, 4, 4)),
            nn.Upsample(scale_factor=2, mode='nearest'),  # b, 512, 8, 8
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # b, 512, 8, 8
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # b, 512, 16, 16
            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # b, 256, 16, 16
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # b, 256, 32, 32
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # b, 128, 32, 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # b, 128, 64, 64
            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # b, 64, 64, 64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # b, 64, 128, 128
            nn.Conv2d(16, 1, 3, stride=1, padding=1),  # b, 1, 128, 128
            nn.Sigmoid()
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


class ResNet_VAE(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        # encoding components
        self.encoder = Encoder()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.embed_dim)
        self.bn2 = nn.BatchNorm1d(self.embed_dim)
        
        # Sampling vector
        self.fc4 = nn.Linear(self.embed_dim, 512)
        self.fc_bn4 = nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, 128 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(128 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.decoder = Decoder()

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # flatten output of conv
        
        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p)
        # mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x)))#.view(-1, 4, 16, 16)
        x = self.decoder(x)
        # x = F.interpolate(x, size=(128, 128), mode="bilinear")
        return x

    def forward(self, x):
        coded = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(coded)

        return x_reconst


class VAELitModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.learning_rate = config.learning_rate
        self.loss_function = config.loss_function
        self.embed_dim = config.embed_dim

        self.model = ResNet_VAE(
            embed_dim=config.embed_dim
        )

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
        self.epoch += 1

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
