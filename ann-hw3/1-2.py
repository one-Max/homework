import sys
import os
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
def WTA(x, W):
    innerdot = W @ x
    return np.where(innerdot == np.max(innerdot))[0][0]

# ======================================================
def compete(X, W, lr=0.1):
    for x in X:
        win_id = WTA(x, W)
        W[win_id] = W[win_id] + lr * (x-W[win_id])

    return W

# ======================================================
def cal_precision(X, W):
    match_pred = X @ W.T
    win_id = [np.where(mark == np.max(mark))[0][0] for mark in match_pred]
    return win_id

# ======================================================
def Wplot(W, title):
    plt.clf()
    for index, w in enumerate(W):
        cols, rows = np.where(w.reshape(5,5)!=nan)
        plt.subplot(1, len(W), index+1)
        ax = plt.gca()                              # 获取到当前坐标轴信息
        ax.xaxis.set_ticks_position('top')          # 将X坐标轴移到上面
        ax.invert_yaxis()                           # 翻转y轴
        plt.margins(0.2)
        plt.scatter(rows, cols, s=w*400, c='b', alpha=0.5)
        plt.grid(True, linestyle='-')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.002)

def main():

    np.random.seed(1)

    # ----------------------------------------------------------
    # original data
    C = [0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0]
    H = [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1]
    L = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1]

    # ----------------------------------------------------------
    # training set prepared
    X_origin = np.array([C, H, L]).astype('float32')

    # ----------------------------------------------------------
    # 载入含噪声字母数据
    oneNoise = np.load('homework/ann-hw3/data/data_onenoise.npy')
    twoNoise = np.load('homework/ann-hw3/data/data_twonoise.npy')

    # ----------------------------------------------------------
    # 合并原始字母和onenoise数据集，创建测试数据集
    id = np.random.randint(0, len(oneNoise)-1, 5)
    
    X_noise = oneNoise[id]
    X_train = np.concatenate((X_origin, X_noise), axis=0)

    X_test = [[], [], []]
    for i in range(3):
        id_r = [i for i in range(i*25, (i+1)*25) if i not in id]
        X_test[i] = oneNoise[id_r]

    # id_reverse = [i for i in range(len(oneNoise)-1) if i not in id]
    # X_test = oneNoise[id_reverse]

    # ----------------------------------------------------------
    # initialization
    num_epochs = 100
    learning_rate = 0.6
    d = 3

    # ----------------------------------------------------------
    # W的不同初始化方式
    W = np.random.rand(d, len(C))               # 随机初始化
    # W = X_origin + np.mean(X_noise, axis=0)     # 原始字符初始化
    # Wplot(W, 'Step:%d, eta:%4.2f' % (0, 0.1))

    # ----------------------------------------------------------
    # 训练
    plt.rcParams['figure.figsize'] = (8, 4)
    plt.draw()
    plt.pause(0.2)
    
    for epoch in range(num_epochs):
        # 学习率规则
        # lr = learning_rate
        lr = learning_rate * (1 - epoch/(num_epochs-1))

        W = compete(X_train, W, lr)
        # Wplot(W, 'Step:%d, lr:%4.2f' % (epoch+1, lr))

    Wplot(W, 'Step:%d, lr:%4.2f' % (epoch+1, lr))
    plt.savefig('homework/ann-hw3/picture/jieguo.png')
    plt.show()

    # win_id = cal_precision(X_test, W)
    # print(win_id)

    for i in range(3):
        win_id = cal_precision(X_test[i], W)
        print(win_id)


if __name__ == '__main__':
    main()
