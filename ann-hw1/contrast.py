import sys,os,math,time
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from mpl_toolkits.mplot3d import Axes3D

def perceptron2(w,x,l,lr):
    x1 = np.array([i for i in x])
    x1 = np.append(x1, [1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else -1
    w1 = np.array(w+lr*(l-o)*x1)
    return w1

def perceptron(w,x,l,lr):
    x1 = np.array([x[0],x[1],x[2],1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else 0
    w1 = np.array(w+lr*(l-o)*x1)
    return w1

def main():
    xdim = [(1,1,1), (1,1,0), (1,0,1), (0,1,1)]
    ldim = [1,0,0,0]
    xdim2 = [(1,1,1), (1,1,-1), (1,-1,1), (-1,1,1)]
    ldim2 = [1,-1,-1,-1]

    lr_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 2, 3] # 学习率存在一个最优值，记不太小也不太大
    epochs = 30

    flag = []
    for lr in lr_list:
        w = np.array([1, 0, 1, 0])
        w_history = []
        for epoch in range(epochs):
            for x, l in zip(xdim, ldim):
                w = perceptron(w, x, l, lr)
            w_history.append(w)
            # print(w)
            if epoch >= 1 and sum([square(i - j) for i, j in zip(w, w_history[epoch - 1])]) == 0:
                flag.append(epoch + 1)
                break

    flag2 = []
    for lr in lr_list:
        w = np.array([1, 0, 1, 0])
        w_history = []
        for epoch in range(epochs):
            for x,l in zip(xdim2,ldim2):
                w = perceptron2(w,x,l,lr)
            w_history.append(w)
            # print(w)
            if epoch >= 1 and sum([square(i-j) for i,j in zip(w,w_history[epoch-1])]) == 0:
                flag2.append(epoch+1)
                break

    # 绘制lr曲线
    plt.figure()
    plt.plot(lr_list, flag, label='Binary')
    plt.plot(lr_list, flag2, label='Bipolar')
    plt.xlabel('learning rate', fontsize=14)
    plt.ylabel('epochs', fontsize=14)
    plt.legend()
    plt.tick_params(labelsize=10)
    plt.title('The relation between leaning rate and convergence', fontsize=15)
    plt.tight_layout()
    plt.savefig('./task2_picture/lr_hunhe.png')
    plt.show()
    print(flag)
    print(flag2)

if __name__ == '__main__':
    main()