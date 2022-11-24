import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn import tree
from sklearn.preprocessing import normalize, scale
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pdb


# visualize the dataset
def visualize(x, y):
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(y[i])
    
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/tongji-hw2/digit.png')
    plt.show()


# ---------------------------------------------------------
# load train/test data
trainSet = datasets.MNIST(root="./data/",
                    train=True,
                    download=False)
testSet = datasets.MNIST(root='./data/',
                    train=False,
                    download=False)
pdb.set_trace()

x_train = np.array(trainSet.data).reshape(len(trainSet), 784)
y_train = np.array(trainSet.targets.data)
x_test = np.array(testSet.data).reshape(len(testSet), 784)
y_test = np.array(testSet.targets.data)

# ---------------------------------------------------------
# visualize(x_train, y_train)

# ---------------------------------------------------------
# data preprocessing
x_train = normalize(x_train, norm='l2', axis=1)
x_test = normalize(x_test, norm='l2', axis=1)

pdb.set_trace()
# 统计每类数据的2维分布
plt.figure(figsize=(10, 5))
for i in range(10):
    distribute = np.sum(np.sign(x_train[y_train == i]),axis=0)/len(x_train[y_train == i])
    plt.plot(range(len(distribute)), distribute, label=str(i), alpha=0.8)

plt.xlabel('pixel', fontsize=15)
plt.ylabel('fre', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('/data1/zjw/homework/tongji-hw2/3.png')
plt.show()

