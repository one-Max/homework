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
    # plt.draw()
    # plt.pause(0.002)

    if path is not None:
        plt.savefig(path)


# ------------------------------------------------------------
# X状态更新
def bam(W, x, y_last):
    """
    input : W: connnection weight;   x: flattrend letter vector,  
    output: y: predection for x; x_new: updated x; ee: energy
    """
    y = np.sign(x @ W)          # 1x3
    y[np.where(y==0)] = y_last[np.where(y==0)]

    x_new = np.sign(y @ W.T)    # 1x35
    x_new[np.where(x_new==0)] = x[np.where(x_new==0)]

    return x_new, y


# ------------------------------------------------------------
# 计算当前的能量函数值
def cal_energy(W, x, y):
    print(x)
    print(y)
    return -((x @ W) @ y)


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


    # 生成训练数据和标签
    X = []        # 8x35
    Y = []        # 8x3

    for index, letter in enumerate(target):
        id_ = np.where(CharStr == letter)[0][0]
        X.append(number_data[id_])

        bi = bin(index).replace('0b','')
        la = np.array([int(i) for i in f'{bi:0>3}'])
        la[la == 0] = -1
        Y.append(la)

    # 数据预处理: 0/1 -> 1/-1
    X = np.array(X).astype('float32') * 2 -1

    # 链接权系数矩阵W
    W = X.T @ Y
    # plt.imshow(W)
    # plt.show()

    # 计算能量函数
    energy = np.zeros(8)
    for i in range(8):
        cal_energy(W, X[i], Y[i])
    print(f'Energy: {energy}')


    # 从噪声数据中还原到对应的标签
    timeNum = 10

    for noi in range(50):                   # 50种加噪声的结果
        X_copy = copy.deepcopy(X)

        for i,x in enumerate(X):
            X_copy[i] = addnoise(x, 0.1)    # 加噪声后的结果
        
        # visualize(X_copy)
        # plt.show()
        
        for j,x in enumerate(X_copy):       # 由于更新的是状态，权值不变，那么<对所有样本迭代一遍再重复time次>和<逐个对单个样本迭代time次> 一样
            y = np.sign(x @ W)
            for _ in range(timeNum):        # 迭代次数
                ee = cal_energy(W, x, y)
                x, y = bam(W, x, y)
            print(y, Y[j])
        
        # visualize(X_copy)
        # plt.show()
        break


if __name__ == '__main__':
    main()
