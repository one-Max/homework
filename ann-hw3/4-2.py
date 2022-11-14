import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import nan


# ======================================================
# 计算hermit函数值
def f_hermit(x):
    return 1.1 * (1 - x + 2 * x**2) * np.exp(-x**2/2)


# ======================================================
# 函数可视化
def show_res(x, y):
    x_line = np.linspace(-4, 4, 250)
    plt.clf()
    plt.plot(x_line, f_hermit(x_line), '--', c='grey',
             linewidth=1, label='Hermit Func')
    plt.scatter(x, y, s=10, c='darkviolet', label='Train Data')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()


# ======================================================
# 获得最近邻id
def WTA_nearest(x, v):
    dist = [np.dot((x - vv), (x - vv)) for vv in v]

    return np.argmin(dist)

# ======================================================
# 更新聚类中心v


def K_neighbor(v, x, lr):
    for xx in x:
        id_ = WTA_nearest(xx, v)
        v[id_] = v[id_] + lr*(xx-v[id_])

    return v

# ======================================================
# 获得X对V的最近邻矩阵H


def CPN_v_out(x, v):
    H = np.zeros((v.shape[0], x.shape[0]))
    for i, xx in enumerate(x):
        id_ = WTA_nearest(xx, v)
        H[id_, i] = 1

    return H

# ======================================================
# 获得隐层到输出层的权重矩阵W


def CPN_W(H, y):
    H1 = np.dot(H.T, H) + 0.000001 * np.eye(H.shape[1])
    W = np.matmul(np.matmul(y.T, np.linalg.inv(H1)), H.T)

    return W


def main():

    np.random.seed(0)

    # -------------------------------------------------------------------
    # 生成数据
    data_num = 80
    x = np.random.uniform(-4, 4, data_num).reshape(-1, 1)
    y = f_hermit(x)

    # -------------------------------------------------------------------
    # initialization
    num_epochs = 100
    d_h = 20
    lr_start = 0.3
    lr_end = 0.02
    

    # -------------------------------------------------------------------
    # training
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.draw()
    plt.pause(.2)
    
    xy_train = np.concatenate([x, y], axis=1)
    
    v_center = xy_train[0:d_h]

    for epoch in range(num_epochs):
        lr = lr_start*(1-epoch/(num_epochs-1)) + lr_end*epoch/(num_epochs-1)
        v_center = K_neighbor(v_center, xy_train, lr)

        if epoch == num_epochs-1:
            H = CPN_v_out(x, v_center[:,0])
            W = CPN_W(H, y)
            
            # 可视化：函数曲线
            show_res(x, y)

            # 可视化：预测输出
            y_pred = np.matmul(W, H)
            plt.scatter(x, y_pred, s=30, c='blue', label='Net out')

            # 可视化：预测输出-泛化到(-4, 4)全部区间
            x_line = np.linspace(-4, 4, 500)
            Hx = CPN_v_out(x_line, v_center[:,0])
            y_line = np.matmul(W, Hx)
            plt.plot(x_line, y_line.reshape(-1,1), c='red',
                    linewidth=1, label='Net Performance')

            # 可视化：聚类中心
            plt.scatter(v_center[:, 0], v_center[:, 1], s=30, c='red', label='Hide Node', alpha=0.5)

            plt.title('Step:%d, lr:%4.2f' % (epoch+1, lr), fontsize=10)
            plt.tight_layout()
            plt.legend(loc='upper right')
            plt.draw()
            plt.pause(0.001)

    plt.savefig('homework/ann-hw3/picture/jieguo4-1.png', dpi=500)
    plt.show()


#------------------------------------------------------------
if __name__ == '__main__':
    main()
