import os, random, math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from torchvision import transforms
from torchsummary import summary
import pdb

img_to_tensor = transforms.ToTensor()   # img -> tensor
tensor_to_pil = transforms.ToPILImage() # tensor -> img


# 读取数据集
class FAMNIST(Dataset):
    def __init__(self, root_dir):
        super(FAMNIST, self).__init__()
        self.notation = []
        self.super_classes = os.listdir(root_dir)
        self.afname = {'cat': 0, 'cow': 1, 'dog': 2, 'horse': 3, 'pig': 4,
                       'apple': 5, 'banana': 6, 'durian': 7, 'grape': 8, 'orange': 9}

        for item in self.super_classes:
            self.sub_classes = os.listdir(os.path.join(root_dir, item))
            for sub_item in self.sub_classes:
                images = os.listdir(os.path.join(root_dir, item, sub_item))
                assert False not in ['.png' in path for path in images]
                for fname in images:
                    self.notation.append([item, sub_item, os.path.join(root_dir, item, sub_item, fname)])

    def __getitem__(self, item):
        img = cv2.imread(self.notation[item][2])
        img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = self.afname[self.notation[item][1]]
        return img, label

    def __len__(self):
        return len(self.notation)


# 定义LeNet
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Linear1 = nn.Linear(16 * 13 * 13, 120)
        self.Linear2 = nn.Linear(120, 86)
        self.Linear3 = nn.Linear(86, 10)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Linear3(x)
        return x


def main():
    # 参数设置
    gpu_id = '5'
    seed = 1
    epoch_num = 30
    lr = 0.0001
    batch_size = 100

    torch.cuda.set_device(int(gpu_id))
    device = torch.device('cuda')

    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型
    model = LeNet()
    model.to(device)
    print(summary(model, (3, 64, 64), batch_size=batch_size))

    # 构建构建损失函数 + 参数优化器
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 读取数据，训练集划分
    _dataset = FAMNIST('/data1/zjw/Dataset/FruitAnimal')
    data_loader_train = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 一些测试环节和可视化
    (image, label) = _dataset[7]           # 采一张图片出图
    print(f'The {7+1}th picture in train set: {list(_dataset.afname.keys())[label]}')
    # tensor_to_pil(image.permute(2,0,1)).show()

    dataiter = iter(data_loader_train)
    images, labels = dataiter.next()        # 取一个batch
    print(images.shape, labels.shape)
    # tensor_to_pil(tv.utils.make_grid(images.permute(0,3,1,2))).show() # 输出这个batch的所有图像
    
    # 设置参数
    if_crop = False
    if_noise = False
    if_rotate = False
    if_jitter = False

    # 随机加噪声
    def AddNoise(input):
        rands = torch.rand(input.size()).cuda()
        res = torch.rand(input.size()).cuda()
        res.copy_(input).cuda()
        res[rands > 0.8] = 0
        return res

    # 变为64*64
    resize = transforms.Resize(size=[64, 64])
    # 随机旋转
    rotate = transforms.RandomRotation(degrees=(0, 180), expand=False)
    # 随机抖动亮度&饱和度
    jitter = transforms.ColorJitter(brightness=0.3,
                                    contrast=0.1,
                                    saturation=0.1,
                                    hue=0.5)
    # 随机裁剪长宽比&尺度
    crop = transforms.RandomResizedCrop(size=[64, 64])


    # # 数据增强可视化
    # (image, label) = _dataset[521]
    # img = (image/255.).permute(2, 0, 1)
    # plt.figure(0)
    # plt.imshow(img.permute(1, 2, 0))
    # plt.savefig('/data1/zjw/homework/ann-hw5/FA/11.png')

    # noise = AddNoise(img)
    # plt.figure(1)
    # plt.imshow(noise.permute(1, 2, 0))
    # plt.savefig('/data1/zjw/homework/ann-hw5/FA/12.png')

    # rot = rotate(img)
    # plt.figure(2)
    # plt.imshow(rot.permute(1, 2, 0))
    # plt.savefig('/data1/zjw/homework/ann-hw5/FA/13.png')

    # jit = jitter(img)
    # plt.figure(3)
    # plt.imshow(jit.permute(1, 2, 0))
    # plt.savefig('/data1/zjw/homework/ann-hw5/FA/14.png')

    # croped = crop(img)
    # plt.figure(4)
    # plt.imshow(croped.permute(1, 2, 0))
    # plt.savefig('/data1/zjw/homework/ann-hw5/FA/15.png')


    # 训练网咯
    losses = []
    accs = []
    for epoch in range(epoch_num):
        for batchid, data in enumerate(data_loader_train):
            samples, labels = data
            samples = (samples.permute(0, 3, 1, 2).float()).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(resize(samples))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if if_crop:
                optimizer.zero_grad()
                outputs = model(crop(samples))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if if_noise:
                optimizer.zero_grad()
                outputs = model(resize(AddNoise(samples)))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if if_jitter:
                optimizer.zero_grad()
                outputs = model(resize(jitter(samples)))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if if_rotate:
                optimizer.zero_grad()
                outputs = model(resize(rotate(samples)))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if batchid%100==0:
                with torch.no_grad():
                    outputs = model(resize(samples))
                    loss = criterion(outputs, labels)
                    acc = (torch.max(outputs, 1)[1]== labels).sum()/len(samples)
                    losses.append(loss.item())
                    accs.append(acc.item())
                    print(f"Epoch:{epoch:3}, Loss:{loss.item():9.6f}, Accuracy:{acc.item():9.6f}")

    def plot_twin(_y1, _y2, _yl1, _yl2):
        fig = plt.figure(figsize=(10, 8), dpi=200)
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Epoch', fontsize=15)
        ax1.set_ylabel(_yl1, fontsize=15)
        ax1.plot(_y1, label=_yl1, color='tab:blue')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left', fontsize=15)

        ax2 = ax1.twinx()           # 创建共用x轴的第二个y轴
        ax2.set_ylabel(_yl2, fontsize=15)
        ax2.plot(_y2, label=_yl2, color='tab:red')
        ax2.tick_params(axis='y')
        ax2.set_ylim([0, 1.1])
        ax2.legend(loc='upper right', fontsize=15)

        fig.tight_layout()
        plt.savefig('/data1/zjw/homework/ann-hw5/FA/1.png')

    plot_twin(losses, accs, 'Loss', 'Accuracy')


if __name__ == "__main__":
    main()