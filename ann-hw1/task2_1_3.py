import sys,os,math,time
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D

def perceptron(w,x,l,lr):
    x1 = np.array([i for i in x])
    x1 = np.append(x1, [1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else -1
    w1 = np.array(w+lr*(l-o)*x1)
    return w1

def main():
    # 双极性数据，其实只要label变成双极性，输入保持不变，收敛速度也会大大加快
    xdim = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]
    ldim = [1,-1,-1,-1]

    dimension = len(xdim[0])
    # w = np.zeros(dimension+1) # 如果初始权重为0向量，学习率大小对收敛无影响
    # w = np.array([1,0,1,0])
    # lr = 1
    lr_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 2, 3] # 学习率存在一个最优值，记不太小也不太大
    epochs = 30
    flag = []
    for lr in lr_list:
        w = np.array([1, 0, 1, 0])
        w_history = []
        for epoch in range(epochs):
            for x,l in zip(xdim,ldim):
                w = perceptron(w,x,l,lr)
            w_history.append(w)
            # print(w)
            if epoch >= 1 and sum([square(i-j) for i,j in zip(w,w_history[epoch-1])]) == 0:
                flag.append(epoch+1)
                break
    print(w)
    # 绘制散点图
    x_scar = [[], [], []]
    for index1, point in enumerate(xdim):
        for index2, cordinate in enumerate(point):
            x_scar[index2].append(cordinate)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_scar[0], x_scar[1], x_scar[2], s=100, c='r',)

    # 绘制曲面
    xx = np.arange(-1, 1.5, 0.05)
    yy = np.arange(-1, 1.5, 0.05)
    X, Y = np.meshgrid(xx, yy)
    # Z = np.sin(X) + np.cos(Y)
    Z = -(w[3]+w[0]*X+w[1]*Y)/w[2]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow') # rstride和cstride为横竖方向的步长

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_zlabel('z', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.title('Perceptron', fontsize=20)
    plt.tight_layout()
    plt.savefig('./task2_picture/task2_1_3.png')
    plt.show()

    # 绘制lr曲线
    plt.figure()
    plt.plot(lr_list, flag)
    plt.xlabel('learning rate', fontsize=14)
    plt.ylabel('epochs', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.title('The relation between leaning rate and convergence', fontsize=15)
    plt.tight_layout()
    plt.savefig('./task2_picture/lr_jixing.png')
    plt.show()
    print(flag)

if __name__ == '__main__':
    main()