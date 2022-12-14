import sys,os,math,random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchsummary import summary
import pdb


img_to_tensor = transforms.ToTensor()   # img -> tensor
tensor_to_pil = transforms.ToPILImage() # tensor -> img


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)         # 32x32 -> 28x28    maxpool -> 14x14
        self.conv2 = nn.Conv2d(6, 16, 5)        # 14x14 -> 10x10    maxpool -> 5x5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_one_epoch(model: nn.Module,
                    criterion: nn.Module,
                    data_loader_train: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    data_loader_val: Iterable):
    model.train()
    criterion.train()
    for batch, data in enumerate(data_loader_train):
        samples, labels = data
        samples = samples.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, labels)
        acc = (torch.max(outputs, 1)[1] == labels).sum()/len(samples)
        acc_val = evaluate(model, data_loader_val, device)

        loss.backward()
        optimizer.step()

    return acc.item(), loss.item(), acc_val.item()


def evaluate(model, data_loader_val, device):
    correct = 0
    total = 0
    for batch, data in enumerate(data_loader_val):
        samples, labels = data
        outputs = model(samples.to(device))
        labels = labels.to(device)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return correct/total


def main(args):
    # 选择cuda或者cpu，表示tensor分配到的设备
    torch.cuda.set_device(args.gpu)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建model
    model = Net()
    model.to(device)
    print(summary(model, (3, 32, 32), batch_size=args.batchsize))
    
    # 构建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 模型参数数量的统计
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of params in model: {n_parameters}')

    # 定义参数优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 对数据的预处理
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # 训练集和测试集划分
    trainset = CIFAR10(root='/data1/zjw/Dataset/', train=True, download=False, transform=transform)
    valset = CIFAR10(root='/data1/zjw/Dataset/', train=False, download=False, transform=transform)
    data_loader_train = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    data_loader_val = DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    classes = ('airplane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    # 一些测试环节和可视化
    (image, label) = trainset[56]           # 采一张图片出图
    print(f'The {56+1}th picture in train set: {classes[label]}')
    # (image+1)/2是为了还原被归一化的数据 (-1,1) -> (0,1)
    # tensor_to_pil((image+1)/2).show()    # 这个出图好慢啊

    dataiter = iter(data_loader_train)
    images, labels = dataiter.next()        # 取了一个batch
    print(images.shape, labels.shape)
    # tensor_to_pil(tv.utils.make_grid((images+1)/2)).show() # 输出这个batch的所有图像

    # 测试
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        
    if args.eval:
        model.eval()
        # 部分结果
        dataiter = iter(data_loader_val)
        images, labels = dataiter.next()
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        print("label:", ''.join('%08s' % classes[labels[j]] for j in range(4)))
        print('predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

        # 输出测试准确率
        acc_val = evaluate(model, data_loader_val, device)
        print(f'10000张测试集上的准确率为: {100*acc_val.item()}')
        
        return
    
    # 训练网络
    accdim = []
    lossdim = []
    accvdim = []
    for epoch in range(args.epochs):
        acc,loss,acc_val = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, data_loader_val)
        lr_scheduler.step()
        accdim.append(acc)
        lossdim.append(loss)
        accvdim.append(acc_val)
        print(f'Epoch:{epoch:3}, Loss:{loss:9.6f}, Accuracys:{acc:9.6f}, Val accuracys:{acc_val:9.6f}')
    
    print('Finishing Training')
    if args.output_dir:
        torch.save(model.state_dict(), args.output_dir+'checkpoint.pth')

    plt.figure(figsize=(10, 8), dpi=200)
    plt.subplot(211)
    plt.plot(accdim, label='Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.subplot(212)
    plt.plot(lossdim, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann-hw5/cifar/1.png')
    
    plt.figure(figsize=(10, 8), dpi=200)
    plt.plot(accdim, label='Accuracy')
    plt.plot(accvdim, label='Test')
    plt.xlabel('Step')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann-hw5/cifar/2.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CIFAR10-Classification', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batchsize', default=1000, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', default=5)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    print(args)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)


