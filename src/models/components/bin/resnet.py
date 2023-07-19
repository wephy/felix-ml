import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deconv=None):
        super(BasicBlock, self).__init__()
        
        self.deconv = deconv
        if deconv:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
    
        if deconv:
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        else:
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )

    def forward(self, x):
        """
        No batch normalization for deconv.
        """
        if self.deconv:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        else: #self.batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1, deconv=False):
#         super(Bottleneck, self).__init__()

#         self.deconv = deconv
#         if deconv:
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.bn2 = nn.BatchNorm2d(planes)
#             self.bn3 = nn.BatchNorm2d(self.expansion * planes)
#             self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=1, bias=False)
#             self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#             self.conv3 = nn.ConvTranspose2d(planes, self.expansion*planes, kernel_size=1, bias=False)

#         else:
#             self.bn1 = nn.BatchNorm2d(planes)
#             self.bn2 = nn.BatchNorm2d(planes)
#             self.bn3 = nn.BatchNorm2d(self.expansion * planes)
#             self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#             self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#             self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

#         self.shortcut = nn.Sequential()

#         if deconv:
#             if stride != 1 or in_planes != self.expansion * planes:
#                 self.shortcut = nn.Sequential(
#                     nn.ConvTranspose2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(self.expansion * planes)
#                 )
#         else:
#             if stride != 1 or in_planes != self.expansion * planes:
#                 self.shortcut = nn.Sequential(
#                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                     nn.BatchNorm2d(self.expansion * planes)
#                 )

#     def forward(self, x):
#         """
#         No batch normalization for deconv.
#         """
#         if self.deconv:
#             out = F.relu((self.conv1(x)))
#             out = F.relu((self.conv2(out)))
#             out = self.conv3(out)
#             out += self.shortcut(x)
#             out = F.relu(out)
#             return out
#         else:
#             out = F.relu(self.bn1(self.conv1(x)))
#             out = F.relu(self.bn2(self.conv2(out)))
#             out = self.bn3(self.conv3(out))
#             out += self.shortcut(x)
#             out = F.relu(out)
#             return out


class ResNet(nn.Module):
    def __init__(self, layers, latent_dims=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        block = BasicBlock

        self.fc_mu = nn.Linear(16384, latent_dims)
        self.fc_var = nn.Linear(16384, latent_dims)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, layers[0], stride=1, deconv=False),
            self._make_layer(block, 128, layers[1], stride=2, deconv=False),
            self._make_layer(block, 256, layers[2], stride=2, deconv=False),
            self._make_layer(block, 512, layers[3], stride=2, deconv=False),
            # nn.MaxPool2d(4, 2),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            self._make_layer(block, 64, layers[0], stride=1, deconv=True),
            self._make_layer(block, 128, layers[1], stride=2, deconv=True),
            self._make_layer(block, 256, layers[2], stride=2, deconv=True),
            self._make_layer(block, 512, layers[3], stride=2, deconv=True),
            nn.MaxUnpool2d(4, 2),
            nn.Linear(512*block.expansion, 512),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dims + 512, 512),
            nn.ELU(),
            nn.Linear(512, 16384),
            nn.Sigmoid(),
        )

    def _make_layer(self, block, planes, num_blocks, stride, deconv):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deconv))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, c):
        inputs = torch.cat([x, c], 2)

        inputs = self.encoder(inputs)
        inputs =  nn.functional.avg_pool2d(inputs, 4)
        batch_size = inputs.size(0)

        z_mu = self.fc_mu(inputs.view(batch_size, -1))
        z_var = self.fc_var(inputs.view(batch_size, -1))
        return z_mu, z_var

    def decode(self, z, c):
        # inputs = torch.cat([z, c], 2)

        c = self.decoder1(c)
        inputs = torch.cat([z, c], 2)

        return self.decoder2(inputs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, c):
        batch_size, width, height = x.size()

        x = x.view(batch_size, 1, 128, 128)
        c = c.view(batch_size, 1, 128, 128)

        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
