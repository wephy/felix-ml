import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, train_loader, test_loader, device):
        super().__init__()
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_size = 128 * 128
        self.feature_size = self.image_size
        self.class_size = self.image_size
        self.latent_size = 1
        self.epoch = 0
        self.optimizer = None
        self.to(device)

        # ResNet
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc11 = nn.Linear(512*ResBlock.expansion, self.latent_size)
        self.fc12 = nn.Linear(512*ResBlock.expansion, self.latent_size)

        # decode
        self.fc2 = nn.Linear(self.latent_size + self.class_size, 512)
        self.fc3 = nn.Linear(512, 4096)
        self.fc4 = nn.Linear(4096, self.feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)

    def encode(self, x, c):
        x = torch.cat([x, c], 2)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        z_mu = self.fc11(x)
        z_var = self.fc12(x)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c.view(-1, self.image_size)], 1)
        h3 = self.elu(self.fc3(self.elu(self.fc2(inputs))))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 1, 128, 128), c.view(-1, 1, 128, 128))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.image_size).to(self.device), reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def test_model(self):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for (data, lattice) in self.test_loader:
                data, lattice = data.to(self.device), lattice.to(self.device)
                recon_batch, mu, logvar = self(data, lattice)
                test_loss += (
                    self.loss_function(recon_batch, data, mu, logvar)
                )
        test_loss /= len(self.test_loader.dataset)
        print("====> Test set loss: {:.4f}".format(test_loss))

    def train_model(self):
        self.epoch += 1
        self.train()
        train_loss = 0
        for batch_idx, (data, lattice) in enumerate(self.train_loader):
            recon_batch, mu, logvar = self(data.to(self.device), lattice.to(self.device))
            self.optimizer.zero_grad()
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item() / len(data),
                    )
                )


class SimpleNet(nn.Module):
    def __init__(self, train_loader, test_loader, device):
        super().__init__()
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.image_size = 128 * 128
        self.feature_size = self.image_size
        self.class_size = self.image_size
        self.latent_size = 5
        self.epoch = 0
        self.optimizer = None
        self.to(device)

        # encode
        self.fc1  = nn.Linear(self.feature_size + self.class_size, 512)
        self.fc21 = nn.Linear(512, self.latent_size)
        self.fc22 = nn.Linear(512, self.latent_size)

        # decode
        self.fc3 = nn.Linear(self.latent_size + self.class_size, 512)
        self.fc4 = nn.Linear(512, self.feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        inputs = torch.cat([z, c.view(-1, self.image_size)], 1)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.image_size), c.view(-1, self.image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.image_size).to(self.device), reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def test_model(self):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for (data, lattice) in self.test_loader:
                data, lattice = data.to(self.device), lattice.to(self.device)
                recon_batch, mu, logvar = self(data, lattice)
                test_loss += (
                    self.loss_function(recon_batch, data, mu, logvar)
                )
        test_loss /= len(self.test_loader.dataset)
        print("====> Test set loss: {:.4f}".format(test_loss))

    def train_model(self):
        self.epoch += 1
        self.train()
        train_loss = 0
        for batch_idx, (data, lattice) in enumerate(self.train_loader):
            recon_batch, mu, logvar = self(data.to(self.device), lattice.to(self.device))
            self.optimizer.zero_grad()
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.detach().cpu().numpy()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item() / len(data),
                    )
                )
