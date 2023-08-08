import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class bce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")

        return recon_loss


class zncc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x):
        x_bar = torch.mean(x)
        y_bar = torch.mean(recon_x)
        u = x - x_bar
        v = recon_x - y_bar
        top = torch.dot(u.flatten(), v.flatten())
        bottom = torch.norm(u) * torch.norm(v)
        zncc = top / bottom

        return (1 - zncc)


class mse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        return recon_loss