import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn import tree
from sklearn.preprocessing import normalize, scale
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


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

x_train = np.array(trainSet.data).reshape(len(trainSet), 784)
y_train = np.array(trainSet.targets.data)
x_test = np.array(testSet.data).reshape(len(testSet), 784)
y_test = np.array(testSet.targets.data)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')

# ---------------------------------------------------------
# visualize(x_train, y_train)

# ---------------------------------------------------------
# data preprocessing
x_train = normalize(x_train, norm='l2', axis=1)
x_test = normalize(x_test, norm='l2', axis=1)

# x_train = scale(x_train, axis=1)
# x_test = scale(x_test, axis=1)

# ---------------------------------------------------------
# decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                    splitter='best',
                                    max_depth=13,
                                    min_samples_leaf=6,
                                    random_state=0)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accu = accuracy_score(y_test, y_pred)
print(accu)

# ---------------------------------------------------------
# # plot the whole tree
# plt.figure()
# plot_tree(clf, filled=True)
# plt.title("Decision tree on train set")
# plt.show()

# ---------------------------------------------------------
# find the best parameters setting
# scores = []
# for i in range(15):
#     clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                       splitter='best',
#                                       max_depth=i+1)
#     clf = clf.fit(x_train, y_train)
#     result = clf.score(x_test,y_test)
#     scores.append(result)
#     print(result)
    
# plt.figure()
# plt.plot(scores)
# plt.xlabel('max depth')
# plt.ylabel('score')
# plt.grid()
# plt.tight_layout()
# plt.savefig('/data1/zjw/homework/tongji-hw2/1.png')
# plt.show()


# # ---------------------------------------------------------
# # PCA projects the dimension into 2
# pca = PCA(n_components=2)
# pca.fit(x_test)
# x_projected = pca.transform(x_test)


# # ---------------------------------------------------------
# # visualize cluster result
# col=['#0072BD','#D95319','#D95319','#7E2F8E','#4298b5','#77AC30','#EEDD82','#0000FF','#4DBEEE','#009F4D']
# plt.figure(figsize=(10, 8))
# handles=[]
# labels=[]

# for i in range(10):
#     markers = dict(c=col[i], linewidths=1, edgecolors='w')
#     handle = plt.scatter(x_projected[y_pred==i, 0], x_projected[y_pred==i, 1], **markers)
#     handles.append(handle)
#     labels.append(str(i))

# plt.xlabel('First Principal Component', fontsize=15)
# plt.ylabel('Second Principal Component', fontsize=15)
# plt.legend(handles=handles,labels=labels)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('/data1/zjw/homework/tongji-hw2/decision-tree.png')
# plt.show()
