import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
# 数据生成：(0, 1)x(0, 1)
def data_generateor(num=100):
    X = np.random.random(num)
    Y = np.random.random(num)
    sample_data = [[x,y] for x,y in zip(X,Y)]

    return np.array(sample_data)


# ======================================================
# 胜者为王：输出层神经元竞争得到内积最大的id
def WTA(x, W):
    innerdot = np.array([(x-ww).dot(x-ww) for ww in W])
    return np.where(innerdot == np.min(innerdot))[0][0]


# ======================================================
# 领域函数
def fu_neighbor(center_id, length, r, d_shape):
    row_c = center_id // d_shape[1]
    col_c = center_id % d_shape[1]

    nb_list = []
    for i in range(length):
        row = i // d_shape[1]
        col = i % d_shape[1]
        if np.abs(row-row_c) <= r and np.abs(col-col_c) <= r:
            nb_list.append(i)

    return nb_list


# ======================================================
# 竞争策略：2维拓扑结构
def compete2(X, W, lr, r, d_shape):
    for xx in X:
        id = WTA(xx, W)

        nb_list = fu_neighbor(id, W.shape[0], r, d_shape)

        for iid in nb_list:
            W[iid] = W[iid] + lr*(xx-W[iid])

    return W


# ======================================================
# 出图
def show_data(data, lineflag=0,  title='', d_shape=None):
    dim = data.shape[1]
    if lineflag == 0:
        plt.scatter(data[:, 0], data[:, 1], s=10, c='blue')
    else:
        plt.scatter(data[:, 0], data[:, 1], s=35, c='red')

        W_reshape = data.reshape(d_shape[0], d_shape[1], dim)
        print(W_reshape.shape)
        for ww in W_reshape:
            plt.plot(ww[:,0], ww[:,1], color='red', linewidth=1, linestyle='--')

        for ww in W_reshape.transpose((1,0,2)):
            plt.plot(ww[:,0], ww[:,1], color='red', linewidth=1, linestyle='--')

    plt.vlines(0, 0, 1, color='green', linewidth=1, linestyles='--')
    plt.vlines(1, 0, 1, color='green', linewidth=1, linestyles='--')
    plt.hlines(0, 0, 1, color='green', linewidth=1, linestyles='--')
    plt.hlines(1, 0, 1, color='green', linewidth=1, linestyles='--')
    plt.axis([-0.1, 1.1, -0.1, 1.2])
    plt.xlabel("x1")
    plt.ylabel("x2")

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
    d_shape = [10, 10]
    d = 100
    lr_start = 0.1;  lr_end = 0.01
    r_start = 3;     r_end = 0

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

        W = compete2(X_train, W, lr, r, d_shape)

        # plt.clf()
        # show_data(X_train, 0)
        # show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %
        #       (epoch+1, num_epochs, r, lr), d_shape)
        # plt.draw() 
        # plt.pause(0.001)
    
    show_data(X_train, 0)
    show_data(W, 1, "Step:%d/%d, r:%d, lr:%4.2f" %
              (epoch+1, num_epochs, r, lr), d_shape)
    plt.savefig('homework/ann-hw3/picture/jieguo3-2.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
