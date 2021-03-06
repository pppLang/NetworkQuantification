import os
import time
import torch
from torch import nn
from torch import optim
from torch.autograd import Function as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleNet_Bin, SimpleNet
from loaddata import dataset
from utils import SignFunction, BinConv2d, updataConvWei
from train import train_epoch, test
from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



if __name__ == "__main__":

    """ train_dataset = datasets.MNIST(root='data', download=True, train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False) """

    train_loader = torch.utils.data.DataLoader(dataset('train'), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset('test'), batch_size=64)
    # net = SimpleNet_Bin(10)
    net = SimpleNet(10)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-6, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_test = nn.CrossEntropyLoss(reduction='sum').cuda()

    log_path = 'logs/bin'
    writer = SummaryWriter(log_dir=log_path)

    epoch_num = 20
    lr0 = 1e-4
    for epoch in range(epoch_num):
        current_lr = lr0 / 2**int(epoch/4)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        # train_epoch(net, optimizer, train_loader, criterion, epoch, writer, current_lr=current_lr, mode='Bin')
        train_epoch(net, optimizer, train_loader, criterion, epoch, writer, current_lr=current_lr, mode='normal')
        test(net, test_loader, criterion_test, epoch, writer)
