import random
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from numpy import nan
from typing import Optional


# ------------------------------------------------------------
# 数据生成：获得全部数据
def data_loader(path):
    with open(path, 'r', encoding='UTF-8') as f:
        data = []
        for line in f:
            d = [int(j) for j in line.strip()]
            data.append(d)

    return np.array(data)


# ------------------------------------------------------------
# 可视化
def visualize(W, path=None):
    plt.clf()
    for i, w in enumerate(W):
        cols, rows = np.where(w.reshape(7, 5) != nan)
        plt.subplot(2, 4, i+1)
        ax = plt.gca()                              # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')          # 将X坐标轴移到上面
        ax.invert_yaxis()                           # 翻转y轴
        plt.margins(0.1)
        plt.scatter(rows, cols, s=w*100)
        plt.grid(linestyle='--')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.002)

    if path is not None:
        plt.savefig(path)


# ------------------------------------------------------------
# X状态更新
def dhnn(W, x):
    net = x @ W
    net[net>0] = 1
    net[net<=0] = -1            
    return net


# ------------------------------------------------------------
# 按一定比例增加噪声
def addnoise(c, noise_ratio = 0.1):
    noisenum = int(len(c) * noise_ratio)
    noisepos = [1]*len(c)
    noisepos[:noisenum] = [-1]*noisenum
    random.shuffle(noisepos)

    cc = np.array([x*y for x,y in zip(c, noisepos)])
    return cc


#------------------------------------------------------------
def main():

    np.random.seed(1)
    plt.rcParams['figure.figsize'] = (12, 8)

    # 读取8个字母
    data_path = '/data1/zjw/homework/ann-hw4/data/char.txt'
    number_data = data_loader(data_path)

    # 字母列表初始化，选择 ZHUOQING 8个字母
    CharStr = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
    target = np.array(['G', 'N', 'I', 'Q', 'O', 'U', 'H', 'Z'])
    # target = np.array(['A', 'M', 'I'])      # 字母本身特征数少，一旦结构相似，网络就分不出来了

    # 生成训练数据和标签
    X_train = []        # 8x35
    X_label = []        # 8x3

    for index, letter in enumerate(target):
        id_ = np.where(CharStr == letter)[0][0]
        X_train.append(number_data[id_])

        bi = bin(index).replace('0b','')
        la = np.array([int(i) for i in f'{bi:0>3}'])
        la[la == 0] = -1
        X_label.append(la)

    # 数据预处理: 0/1 -> 1/-1
    X_train = np.array(X_train).astype('float32') * 2 -1
    # visualize(X_train)

    # 链接权系数矩阵W
    W = X_train.T @ X_train
    for i in range(X_train.shape[1]):
        W[(i, i)] = 0
    # plt.imshow(W)
    # plt.show()

    # 更新X状态
    timeNum = 1000
    plt.draw()
    plt.pause(0.2)

    for noi in range(1):                 # 4种加噪声结果
        C = copy.deepcopy(X_train)
        for i,x in enumerate(X_train):
            C[i] = addnoise(x, 0)
        
        # visualize(c)
        for _ in range(timeNum):      # 迭代次数
            C = dhnn(W, C)
        visualize(C)

    plt.show()


if __name__ == '__main__':
    main()
