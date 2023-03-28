import torch
import pytest
from pytorch_loss_functions import DiceLoss, DiceBCELoss
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader
from pytorch_loss_functions import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss

def train(model, device, train_loader, optimizer, epoch, loss):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch['image']
        target = batch['mask']
        data, target = data.to(device), target.to(device)
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        if loss == "dice":
            loss_fn = DiceLoss() #smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss == "dicebce":
            loss_fn = DiceBCELoss()
        elif loss == "IoU":
            loss_fn = IoULoss()
        elif loss == "focal":
            loss_fn = FocalLoss()
        elif loss == "tversky":
            loss_fn = TverskyLoss()
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))

def dry_run(loss):
    
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    root = "./"
    #SimpleOxfordPetDataset.download(root)
    model = smp.Unet('mobilenet_v2', in_channels=3, classes=1).to(device)
    train_dataset = SimpleOxfordPetDataset(root, "train")
    valid_dataset = SimpleOxfordPetDataset(root, "valid")
    test_dataset = SimpleOxfordPetDataset(root, "test")

    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, device, train_dataloader, optimizer, 0, loss)

class TestSegmentationLosses():
    def test_dice_loss(self):
        dry_run("dice")
    def test_dicebce_loss(self):
        dry_run("dicebce")
    def test_iou_loss(self):
        dry_run("IoU")
    def test_focal_loss(self):
        dry_run("focal")
    def test_tversky_loss(self):
        dry_run("tversky")
