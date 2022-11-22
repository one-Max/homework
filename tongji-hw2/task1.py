import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
# 加载数据
train = datasets.MNIST(root="./data/",
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)
test = datasets.MNIST(root='./data/',
                      train=False,
                      transform=transforms.ToTensor(),
                      download=True)

train_data = np.array(train.data).reshape(len(train), 784)
train_label = np.array(train.train_labels.data)
test_data = np.array(test.data).reshape(len(test), 784)
test_label = np.array(test.test_labels.data)

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)