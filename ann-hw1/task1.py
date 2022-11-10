import sys,os,math,time
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

def sigmoid(z):
    return 1/(1 + math.exp(-z))

def derivative_of_sigmoid(z):
    return z*(1-z)

def hebbian(w,x,l,lr):
    x1 = np.array([x[0], x[1], 1])
    o = sigmoid(np.dot(w, x1))
    w1 = np.array(w + lr*o*x1)
    return w1

def perceptron(w,x,l,lr):
    x1 = np.array([x[0],x[1],1])
    net = np.dot(w, x1)
    o = 1 if net >= 0 else -1
    w1 = np.array(w+lr*(l-o)*x1)
    return w1

def delta(w,x,l,lr):
    x1 = np.array([x[0], x[1], 1])
    o = sigmoid(np.dot(w, x1))
    deri = derivative_of_sigmoid(o)
    w1 = np.array(w + lr*(l-o)*deri*x1)
    return w1

def widraw_Hoff(w,x,l,lr):
    x1 = np.array([x[0], x[1], 1])
    w1 = np.array(w + lr*(l-np.dot(w, x1))*x1)
    return w1

def correlation(w,x,l,lr):
    x1 = np.array([x[0], x[1], 1])
    w1 = w + lr*l*x1
    return w1

def main():
    xdim = [(-0.1, 0.3), (0.5, 0.7), (-0.5, 0.2), (-0.7, 0.3), (0.7, 0.1), (0, 0.5)]
    ldim = [1, -1, 1, 1, -1, 1]
    dimension = len(xdim[0])
    w = np.zeros(dimension+1)  # 权重向量w=[w...,w0]
    lr = 1
    epochs = 5

    w_history = []
    for epoch in range(epochs):
        for x,l in zip(xdim, ldim):
            w = widraw_Hoff(w,x,l,lr)
            # print(w)
        w_history.append(w)
        print(w)
        # print()

    w_epo1 = w_history[0]
    print(f'The value of weight after the first epoch is {w_epo1}.')

    plt.figure()
    # 绘制分类线
    x = np.linspace(-2, 2, 50)
    for index, w_h in enumerate(w_history):
        y = (-w_h[2]-w_h[0]*x)/w_h[1]
        plt.plot(x, y, label=f'{index+1} epoch')

    # 绘制散点图
    x_scar = [[],[]]
    for index1, point in enumerate(xdim):
        for index2, cordinate in enumerate(point):
            x_scar[index2].append(cordinate)
    plt.scatter(x_scar[0], x_scar[1])
    plt.axis([-1,1,0,1])
    plt.xlabel("X1")
    plt.ylabel("X2")
    # plt.title('Correlation')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend()
    # plt.savefig('./task1_picture/Correlation.png')
    plt.show()

if __name__ =='__main__':
    main()