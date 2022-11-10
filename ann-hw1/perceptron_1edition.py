import os, sys, math, time
import numpy as np
import matplotlib.pyplot  as plt
from numpy import *

def sigmoid(z):  # sigmoid函数
    return 1/(1 + math.exp(-z))

def main():
    xdim = [(-0.1,0.3),(0.5,0.7),(-0.5,0.2),(-0.7,0.3),(0.7,0.1),(0,0.5)]
    ldim = [1,-1,1,1,-1,1]
    dimension = len(xdim[0])

    data = []  # 数据结构[[-0.1, 0.3, 1], [-0.5, -0.7, -1], [-0.5, 0.2, 1], [-0.7, 0.3, 1], [-0.7, -0.1, -1], [0.0, 0.5, 1]]
    for i in range(len(xdim)):
        if ldim[i] == 1:
            data.append([xdim[i][0], xdim[i][1], ldim[i]])
        elif ldim[i] == -1:
            data.append([-xdim[i][0], -xdim[i][1], ldim[i]])
    data = np.array(data)

    w = np.zeros(dimension+1)  # 权重向量w=[[],w0]
    lr = 2  # 学习率
    epochs = 4  # 次数

    w_history=[]
    for epoch in range(epochs):
        for i in range(len(data)):
            output = np.dot(w, data[i])
            if output > 0:
                delta_w = 0
            else:
                delta_w = lr * data[i]
            w = w + delta_w
        w_history.append(w)
        print(w)

    plt.figure()
    # 绘制分类线
    x = np.linspace(-1, 1, 50)
    for w in w_history:
        y = (-w[2]-w[0]*x)/w[1]
        plt.plot(x, y)
    # 绘制散点图
    x_scar = [[],[]]
    for index1, point in enumerate(xdim):
        for index2, cordinate in enumerate(point):
            x_scar[index2].append(cordinate)
    plt.scatter(x_scar[0], x_scar[1])
    plt.axis([-1,1,0,1])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
