import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
# 数据生成：获得全部数据 -> 6x26个
def data_loader(num_pattern, root_path):
    ascii_data = []
    label_data = []

    for i in range(num_pattern):
        with open(root_path+'ascii'+str(i+1)+'.txt', 'r', encoding='UTF-8') as f:
            group = []
            label = []
            for index, line in enumerate(f.readlines()):
                temp = []
                for j in line.strip():
                    temp.append(int(j))
                group.append(temp)
                label.append(index+1)

            ascii_data.append(group)
            label_data.append(label)
    
    return np.array(ascii_data), np.array(label_data)

# ======================================================
# 胜者为王：输出层神经元竞争得到内积最大的id
def WTA(x, W):
    # innerdot = W @ x
    # return np.where(innerdot == np.max(innerdot))[0][0]
    innerdot = np.array([(x-ww).dot(x-ww) for ww in W])
    return np.where(innerdot == np.min(innerdot))[0][0]

# ======================================================
# 竞争策略：无拓扑结构
def compete(X, W, lr=0.1):
    for x in X:
        win_id = WTA(x, W)
        W[win_id] = W[win_id] + lr * (x-W[win_id])

    return W

# ======================================================
# 竞争策略：一维拓扑结构
def compete1(X, W, lr):
    for xx in X:
        id = WTA(xx, W)
        W[id] = W[id] + lr * (xx - W[id])

        if id > 0:
            W[id-1] = W[id-1] + lr*(xx-W[id-1])
        if id+1 < W.shape[0]:
            W[id+1] = W[id+1] + lr*(xx-W[id+1])

    return W

# ======================================================
# 竞争策略：二维拓扑结构
def compete2(X, W, lr):
    for xx in X:
        id = WTA(xx, W)
        nb_id = fu_neighbor(id, 5, 5)

        for i in nb_id:
            W[i] = W[i] + lr * (xx - W[i])

    return W

# ======================================================
# 领域函数：只包括上下左右
def fu_neighbor(center_id, row, col):
    nb_id = [center_id]
    rown = center_id // col
    coln = center_id % col

    if coln > 0:
        nb_id.append(center_id-1)
    if coln < col-1:
        nb_id.append(center_id+1)
    if rown > 0:
        nb_id.append(center_id-col)
    if rown < row-1:
        nb_id.append(center_id+col)
    
    return nb_id

# ======================================================
# 出图：25个竞争神经元的可视化
def Wplot(W, title):
    plt.clf()
    for index, w in enumerate(W):
        cols, rows = np.where(w.reshape(7, 5) != nan)
        plt.subplot(5, 5, index+1)
        ax = plt.gca()                              # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')          # 将X坐标轴移到上面
        ax.invert_yaxis()                           # 翻转y轴
        plt.margins(0.2)
        plt.scatter(rows, cols, s=w*5, c='b', alpha=0.5)
        plt.xticks(())
        plt.yticks(())

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.002)

# ======================================================
# 评估聚类结果
def performance(X, W, X_label, CharStr):

    out = []
    for i in range(W.shape[1]):
        out.append('')

    for index, x in enumerate(X):
        id_ = WTA(x, W)                     # 当前x，25个神经元哪个获胜
        char = CharStr[X_label[index]-1]    # 当前x，对应的真实类别字母
        out[id_] += char
    
    for i in range(5):
        for j in range(5):
            print(f'{(i*5+j)+1:2}: {out[i*5+j]:15}', end = ' ')
        print('\n')


# ======================================================
def main():

    np.random.seed(1)

    # -------------------------------------------------------------------
    # 读取6种字体文件
    num_p = 6
    data_root = '/data1/zjw/homework/ann-hw3/data/'
    ascii_data, label_data = data_loader(num_p, data_root)

    # -------------------------------------------------------------------
    # 字母列表初始化，选择 ZHUOQING 8个字母
    CharStr = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    target = np.array(['G', 'N', 'I', 'Q', 'O', 'U', 'H', 'Z'])
    target_2 = np.array(['Z', 'H', 'O', 'U', 'J', 'I', 'A', 'W', 'E'])

    X_train = []        # 48x35
    X_label = []        # 48

    for index, pattern in enumerate(label_data):
        for letter in target:
            id_ = np.where(CharStr == letter)[0][0]
            X_train.append(ascii_data[index][id_])
            X_label.append(pattern[id_])
    
    # -------------------------------------------------------------------
    # training set prepared
    X_train = np.array(X_train).astype('float32')

    # -------------------------------------------------------------------
    # initialization
    num_epochs = 1000
    learning_rate = 0.6
    d = 25
    W = np.random.rand(d, X_train.shape[1])     # 随机初始化

    # -------------------------------------------------------------------
    # 训练
    plt.rcParams['figure.figsize'] = (4, 10)
    plt.draw()
    plt.pause(0.2)
    
    for epoch in range(num_epochs):
        lr = learning_rate*(1-epoch/(num_epochs-1)) + 0.01*epoch/(num_epochs-1)

        W = compete2(X_train, W, lr)
        # Wplot(W, 'Step:%d, lr:%4.2f' % (epoch+1, lr))

    Wplot(W, 'Step:%d, lr:%4.2f' % (epoch+1, lr))
    plt.savefig('homework/ann-hw3/picture/jieguo2.png', dpi=1000)
    plt.show()

    # -------------------------------------------------------------------
    # 评价结果
    performance(X_train, W, X_label, CharStr)


if __name__ == '__main__':
    main()

