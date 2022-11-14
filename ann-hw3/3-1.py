import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
# 数据生成：num个三角区域内的样本点
def data_generateor(num=100):
    X = np.random.random(5000)
    Y = np.random.random(5000)
    point_group = []

    for x, y in zip(X, Y):
        if y <= (3**0.5)*x and y <= 3**0.5-(3**0.5)*x and y >= 0:
            point_group.append([x, y])

    point_group = np.array(point_group).astype('float32')
    id_list = np.random.randint(0, len(point_group)-1, num)
    sample_data = point_group[id_list]

    return sample_data


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

    return nb_list


# ======================================================
# 竞争策略：一维拓扑结构
def compete1(X, W, lr, r):
    for xx in X:
        id = WTA(xx, W)
        
        nb_list = fu_neighbor(id, W.shape[0], r)

        for iid in nb_list:
            W[iid] = W[iid] + lr*(xx-W[iid])

    return W


# ======================================================
# 出图
def show_data(data, lineflag=0, title=''):

    if lineflag == 0:
        plt.scatter(data[:, 0], data[:, 1], s=10, c='blue')
    else:
        plt.scatter(data[:, 0], data[:, 1], s=35, c='red')
        plt.plot(data[:, 0], data[:, 1], 'y-', linewidth=1)

    board_x = np.linspace(0, 1, 200)
    board_y = [np.sqrt(3)*(0.5-abs(x-0.5)) for x in board_x]
    plt.plot(board_x, board_y, 'c--', linewidth=1)
    plt.plot(board_x, np.zeros(len(board_x)), 'c--', linewidth=1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis([-0.05, 1.05, -0.05, .9])

    if len(title) > 0:
        plt.title(title, fontsize=10)

    plt.grid(True)
    plt.tight_layout()


# ======================================================
def main():

    np.random.seed(0)

    # -------------------------------------------------------------------
    # 生成数据
    sample_num = 400
    X_train = data_generateor(sample_num)

    # -------------------------------------------------------------------
    # initialization
    num_epochs = 200
    d = 100; 
    lr_start = 0.3;  lr_end = 0.01
    r_start = 10;    r_end = 0

    W = np.random.rand(d, X_train.shape[1])
    W[:,1] = W[:,1] * (3**0.5)/2

    # -------------------------------------------------------------------
    # 训练
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.draw()
    plt.pause(0.2)
    # show_data(X_train, 0)
    for epoch in range(num_epochs):
        lr = lr_start*(1-epoch/(num_epochs-1)) + lr_end*epoch/(num_epochs-1)
        r = int(r_start*(1-(epoch/(num_epochs-1))) + r_end*epoch/(num_epochs-1))

        W = compete1(X_train, W, lr, r)

        # plt.clf()
        # show_data(X_train, 0)
        # show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %(epoch+1, num_epochs, r, lr))
        # plt.draw()
        # plt.pause(0.001)
    

    show_data(X_train, 0)
    show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %(epoch+1, num_epochs, r, lr))
    plt.savefig('homework/ann-hw3/picture/jieguo3.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
