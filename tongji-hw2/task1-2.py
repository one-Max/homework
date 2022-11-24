import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# load train/test data
trainSet = datasets.MNIST(root="./data/",
                    train=True,
                    download=False)
testSet = datasets.MNIST(root='./data/',
                    train=False,
                    download=False)

x_train = np.array(trainSet.data).reshape(len(trainSet), 784)
y_train = np.array(trainSet.targets.data)
x_test = np.array(testSet.data).reshape(len(testSet), 784)
y_test = np.array(testSet.targets.data)

# ---------------------------------------------------------
# data preprocessing
x_train = normalize(x_train, norm='l2', axis=1)
x_test = normalize(x_test, norm='l2', axis=1)

# ---------------------------------------------------------
# PCA projects the dimension into 2
pca = PCA(n_components=2)
pca.fit(x_test)
x_projected = pca.transform(x_test)

# ---------------------------------------------------------
# kmeans
kmeans = KMeans(n_clusters=10, random_state=0)
y_pred = kmeans.fit_predict(x_projected)

# ---------------------------------------------------------
# visualize cluster result
col=['#0072BD','#D95319','#D95319','#7E2F8E','#4298b5','#77AC30','#EEDD82','#0000FF','#4DBEEE','#009F4D']
plt.figure(figsize=(10, 8))
handles=[]
labels=[]

for i in range(10):
    markers = dict(c=col[i], linewidths=1, edgecolors='w')
    handle = plt.scatter(x_projected[y_pred==i, 0], x_projected[y_pred==i, 1], **markers)
    handles.append(handle)
    labels.append(str(i))

plt.xlabel('First Principal Component', fontsize=15)
plt.ylabel('Second Principal Component', fontsize=15)
plt.legend(handles=handles,labels=labels)
plt.grid(True)
plt.tight_layout()
plt.savefig('/data1/zjw/homework/tongji-hw2/kmeans.png')
plt.show()

