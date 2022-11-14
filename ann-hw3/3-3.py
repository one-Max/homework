import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
# 胜者为王：输出层神经元竞争得到内积最大的id
def WTA(x, W):
    innerdot = np.array([(x-ww).dot(x-ww) for ww in W])
    return np.where(innerdot == np.min(innerdot))[0][0]


# ======================================================
# 领域函数
def fu_neighbor(center_id, length, r):
    nb_list = []
    for i in range(center_id-r, center_id+r+1):
        if i >= 0 and i <= length-1:
            nb_list.append(i)
        elif i < 0:
            nb_list.append(i+length)
        else:
            nb_list.append(i-length)

    return nb_list


# ======================================================
# 竞争策略：一维拓扑结构
def compete1(X, W, lr, r):
    for xx in X:
        id = WTA(xx, W)

        nb_list = fu_neighbor(id, W.shape[0], r)

        for iid in nb_list:
            W[iid] = W[iid] + lr*(xx-W[iid])

        for id in range(W.shape[0]):
            W[id] = W[id] + (np.random.random(2) - 0.5) * lr**2

    return W


# ======================================================
# 出图
def show_data(data, lineflag=0, title=''):
    if lineflag == 0:
        plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='View Site')
    else:
        plt.scatter(data[:, 0], data[:, 1], s=35,
                    c='red', label='Neural Position', alpha=0.5)
        plt.plot(data[:, 0], data[:, 1], 'y--', linewidth=1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis([-0.05, 1.05, -0.05, 1.05])

    if len(title) > 0:
        plt.title(title, fontsize=10)

    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper right')


# ======================================================
def main():

    np.random.seed(9)

    # -------------------------------------------------------------------
    # 生成数据
    X_train = [[236, 53], [408, 79], [909, 89], [115, 264], [396, 335],
              [185, 456], [699, 252], [963, 317], [922, 389], [649, 515]]


    X_train = np.array([[xy[0]/1000, xy[1]/600] for xy in X_train])


    # -------------------------------------------------------------------
    # initialization
    num_epochs = 200
    d = 10
    lr_start = 0.3;  lr_end = 0.02
    r_start = 2;     r_end = 0

    W = np.random.rand(d, X_train.shape[1])

    # -------------------------------------------------------------------
    # 训练
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.draw()
    plt.pause(0.2)

    for epoch in range(num_epochs):
        lr = lr_start*(1-epoch/(num_epochs-1)) + lr_end*epoch/(num_epochs-1)
        r = int(r_start*(1-(epoch/(num_epochs-1))) +
                r_end*epoch/(num_epochs-1))

        W = compete1(X_train, W, lr, r)

        
        # plt.clf()
        # show_data(X_train, 0)
        # show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %
        #       (epoch+1, num_epochs, r, lr))
        # plt.draw() 
        # plt.pause(0.001)
    
    show_data(X_train, 0)
    show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %
              (epoch+1, num_epochs, r, lr))
    plt.savefig('homework/ann-hw3/picture/jieguo3-3.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
