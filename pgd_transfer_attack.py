import torch
import argparse
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset 
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchattacks import PGD
import torchvision.transforms as transform

def create_attack():

    # dataset
    test_set = CIFAR10('.', train=False, download=True, transform=transform.ToTensor())
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # load pre-trained model
    model = ptcv_get_model('resnet20_cifar10', pretrained=True)  # 5.97 top-1 error
    model = model.eval().cuda()

    # prepare attack
    atck = PGD(model, eps=8/255, alpha=2/255, steps=10) # same params as in hydra
    atck.set_return_type('int') # Save as integer.
    atck.save(data_loader=test_loader, save_path="./cifar10_pgd.pt", verbose=True)

if __name__ == '__main__':
    create_attack()