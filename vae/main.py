import torch
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import os, time, tqdm
from model import loss, cVAE
from utils import EarlyStop
import argparse
import importer  # noqa # pylint: disable=unused-import
from datasets import FLAGDataset  # noqa # pylint: disable=import-error


# Arguments
parser = argparse.ArgumentParser(description="VAE FLAG Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    metavar="N",
    help="input batch size for training (default: 8)",
)
parser.add_argument(
    "--latent-size",
    type=int,
    default=16,
    metavar="N",
    help="input batch size for training (default: 8)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=2,
    metavar="N",
    help="how many batches to wait before logging training status",
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_built()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}

############## loading data ###################

flag_location = r"D:\\felix-ml\datasets\\flag"
flag_dataset = FLAGDataset(flag_location)
flag_shrunk = torch.utils.data.Subset(flag_dataset, torch.arange(3000))
dataset_len = len(flag_shrunk)
train_dataset, test_dataset = torch.utils.data.random_split(flag_shrunk, [2500, 500])
train_loader = DataLoader(
    dataset=train_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=4,
)
test_loader = DataLoader(
    dataset=test_dataset,  # the dataset instance
    batch_size=args.batch_size,  # automatic batching
    drop_last=True,  # drops the last incomplete batch in case the dataset size is not divisible by 64
    shuffle=True,  # shuffles the dataset before every epoch
    num_workers=4,
)

############## loading models ###################

net = cVAE((1, 128, 128), (1, 128, 128), nhid=2, ncond=args.latent_size)
net.to(device)
print(net)
save_name = "cVAE.pt"

############### training #########################

lr = 0.01
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0001
)


def adjust_lr(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_rate


retrain = True
if os.path.exists(save_name):
    print("Model parameters have already been trained before. Retrain ? (y/n)")
    ans = input()
    if not (ans == "y"):
        checkpoint = torch.load(save_name, map_location=device)
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = lr

max_epochs = 1000
early_stop = EarlyStop(patience=20, save_name=save_name)
net = net.to(device)

print("training on ", device)
for epoch in range(max_epochs):

    train_loss, n, start = 0.0, 0, time.time()
    for X, y in tqdm.tqdm(train_loader, ncols=50):
        X = X.to(device)
        y = y.to(device)
        X_hat, mean, logvar = net(X, y)

        l = loss(X, X_hat, mean, logvar).to(device)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.cpu().item()
        n += X.shape[0]

    train_loss /= n
    print(
        "epoch %d, train loss %.4f , time %.1f sec"
        % (epoch, train_loss, time.time() - start)
    )

    adjust_lr(optimizer)

    if early_stop(train_loss, net, optimizer):
        break

checkpoint = torch.load(early_stop.save_name)
net.load_state_dict(checkpoint["net"])
