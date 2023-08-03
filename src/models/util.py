import torch
from torch import nn
import torch.nn.functional as F


class bce(nn.Module):
    def __init__(self, embedding_weight=0.1):
        super().__init__()
        self.embedding_weight = embedding_weight

    def forward(self, recon_x, x, z):
        recon_loss = F.binary_cross_entropy(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2

        return (
            (recon_loss + self.embedding_weight * embedding_loss).mean(dim=0),
            (recon_loss).mean(dim=0),
            (embedding_loss).mean(dim=0),
        )

class zncc(nn.Module):
    def __init__(self, embedding_weight=0.1):
        super().__init__()
        self.embedding_weight = embedding_weight

    def forward(self, recon_x, x, z):
        x_bar = torch.mean(x)
        y_bar = torch.mean(recon_x)
        u = x - x_bar
        v = recon_x - y_bar
        top = torch.dot(u.flatten(), v.flatten())
        bottom = torch.norm(u) * torch.norm(v)
        zncc = top / bottom

        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2

        return (
            ((1 - zncc) + self.embedding_weight * embedding_loss).mean(dim=0),
            (zncc).mean(dim=0),
            (embedding_loss).mean(dim=0),
        )

class mse(nn.Module):
    def __init__(self, embedding_weight=0.1):
        super().__init__()
        self.embedding_weight = embedding_weight

    def forward(self, recon_x, x, z):
        recon_loss = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)

        embedding_loss = 0.5 * torch.linalg.norm(z, dim=-1) ** 2

        return (
            (recon_loss + self.embedding_weight * embedding_loss).mean(dim=0),
            (recon_loss).mean(dim=0),
            (embedding_loss).mean(dim=0),
        )
