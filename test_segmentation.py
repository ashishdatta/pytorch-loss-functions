from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
import os
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from pprint import pprint
from torch.utils.data import DataLoader
from pytorch_loss_functions import DiceLoss


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch['image']
        target = batch['mask']
        data, target = data.to(device), target.to(device)
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        loss_fn = DiceLoss() #smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))

def main():
    
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

    """
    sample = train_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = valid_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()

    sample = test_dataset[0]
    plt.subplot(1,2,1)
    plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
    plt.subplot(1,2,2)
    plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
    plt.show()
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, device, train_dataloader, optimizer, 0)
if __name__ == '__main__':
    main()
