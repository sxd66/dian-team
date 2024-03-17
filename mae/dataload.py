from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import numpy as np
def data_load():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.2),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_val = transforms.Compose([

        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root="../data/fashion-mnist", train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root="../data/fashion-mnist", train=False, transform=transform_val, download=True)

    train_loader=DataLoader(train_dataset,32,True,drop_last=True)
    val_loader=DataLoader(test_dataset,32,True,drop_last=True)
    return train_loader,val_loader

