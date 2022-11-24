import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from xgboost import plot_importance


def tsne_visualize(x, y, n_components):
    # ---------------------------------------------------------
    # tSNE projects the dimension into 2
    tsne = TSNE(n_components=n_components, learning_rate='auto')
    x_projected = tsne.fit_transform(x)

    # ---------------------------------------------------------
    # visualize cluster result
    col=['#0072BD','#D95319','#D95319','#7E2F8E','#4298b5','#77AC30','#EEDD82','#0000FF','#4DBEEE','#009F4D']
    plt.figure(figsize=(10, 8))
    handles=[]
    labels=[]

    for i in range(10):
        markers = dict(c=col[i], linewidths=1, edgecolors='w')
        handle = plt.scatter(x_projected[y==i, 0], x_projected[y==i, 1], **markers)
        handles.append(handle)
        labels.append(str(i))

    plt.xlabel('First Principal Component', fontsize=15)
    plt.ylabel('Second Principal Component', fontsize=15)
    plt.legend(handles=handles,labels=labels)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/tongji-hw2/tsne-gt.png')
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

# ---------------------------------------------------------
# data preprocessing
x_train = normalize(x_train, norm='l2', axis=1)
x_test = normalize(x_test, norm='l2', axis=1)

# tsne_visualize(x_train, y_train, 2)

model = XGBClassifier(learning_rate=0.1,
                      n_estimators=20,
                      max_depth=10,
                      min_child_weight = 1,
                      subsample = 0.8,
                      colsample_btree = 0.8,
                      objective = 'multi:softmax',
                      scale_pos_weight = 1,
                      random_state= 0)
model.fit(x_train,
          y_train,
          eval_set=[(x_test,y_test)],
          eval_metric = 'mlogloss',
          early_stopping_rounds = 10,
          verbose = True)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"accuarcy:{accuracy}")
 
 
 
 
 