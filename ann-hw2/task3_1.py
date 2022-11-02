import sys
import os
import math
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, exp, sum, ones, square, zeros

#------------------------------------------------------------
#--------             BP Neural Network             ---------
#------------------------------------------------------------


#------------------------------------------------------------
# build the architecture of neural network
# input: neurons num of input , hidden layer, output
# output: middle result of hidden layer, output layer, initialized parameters matrix

def build_nn(d_i=None, d_h=None, d_o=None):

    W_ih = 0.5*np.random.random([d_i, d_h])         # 2x5
    W_ho = 0.5*np.random.random([d_h, d_o])         # 5x3
    b_h = -zeros([1, d_h])                       # 1x5
    b_o = -zeros([1, d_o])                       # 1x5

    parameters = {'W_ih': W_ih,
                  'W_ho': W_ho,
                  'b_h': b_h,
                  'b_o': b_o}

    return parameters


#------------------------------------------------------------
# forward prpogation

def forward_popagation(X, para):
    m = X.shape[0]
    A = dot(X, para['W_ih']) + dot(ones([m, 1]),
                                   para['b_h'])   # 9x2 2x5 + 9x1 1x5 --> 9x5
    H = 1/(1 + exp(-A))                                         # 9x5

    B = dot(H, para['W_ho']) + dot(ones([m, 1]),
                                   para['b_o'])   # 9x5 5x3 + 9x1 1x5 --> 9x5
    # O = 1/(1 + exp(-B))                                         # 9x3
    O = B                                                       # 9x3

    storage = {'A': A,
               'H': H,
               'B': B,
               'O': O}

    return storage


#------------------------------------------------------------
# calculate the loss

def cal_loss(storage, y):
    O = storage['O']
    m = y.shape[0]
    loss = sum((1 / (2*m)) * square(y-O))

    return loss


#------------------------------------------------------------
# backward propogation:to calculate the gradient
# output: matrix derativate

def backward_propagation(para, storage, X, y):
    m = X.shape[0]
    W_ho = para['W_ho']
    O = storage['O']
    H = storage['H']

    # h->o
    # d1 = (O - y) * (O * ( 1 - O))               # 9x3
    d1 = O - y
    dW_ho = (1/m) * dot(H.T, d1)                # 5x9 9x3 --> 5x3
    db_o = (1/m) * dot(ones([1, m]), d1)        # 1x9 9x3 --> 1x3

    # i->h
    d2 = (dot(d1, W_ho.T)) * (H * (1 - H))      # 9x5
    dW_ih = (1/m) * dot(X.T, d2)                # 2x9 9x5 --> 2x5
    db_h = (1/m) * dot(ones([1, m]), d2)        # 1x9 9x5 --> 1x5

    grads = {'dW_ho': dW_ho,
             'dW_ih': dW_ih,
             'db_o': db_o,
             'db_h': db_h}

    return grads


#------------------------------------------------------------
# update the parameters
# input: weight matrix, gradient, learning rate
# output: updated parameters

def update_parameters(para, grads, lr):
    W_ho = para['W_ho'] - lr * grads['dW_ho']
    W_ih = para['W_ih'] - lr * grads['dW_ih']
    b_o = para['b_o'] - lr * grads['db_o']
    b_h = para['b_h'] - lr * grads['db_h']

    parameters = {'W_ih': W_ih,
                  'W_ho': W_ho,
                  'b_h': b_h,
                  'b_o': b_o}

    return parameters


#------------------------------------------------------------
# trainning process

def train_one_epoch(X, y, parameters, lr=0.2):
    storage = forward_popagation(X, parameters)
    loss = cal_loss(storage, y)
    grads = backward_propagation(parameters, storage, X, y)
    parameters = update_parameters(parameters, grads, lr)

    return parameters, loss

#------------------------------------------------------------
# if the sample is misclassified




def main():
    # dataset
    alphas = []
    with open('/data1/zjw/homework/ann_hw2/ascii8_16.txt', 'r', encoding='utf-8')as f:
        for line in f.readlines():
            line = line.strip()
            line = list(line)
            line = [int(x)for x in line]
            alphas.append(line)

    X = np.array(alphas)
    y = X

    # create model
    np.random.seed(0)
    error = []
    plt.figure(figsize=(10, 6))

    for hh in range(1, 16):
        parameters = build_nn(d_i=len(X[0]), d_h=hh, d_o=len(X[0]))
        """Different seed(different initial weight) leads to different loss curve"""

        # train
        loss_history = []
        num_epochs = 2000
        for epoch in range(num_epochs):
            parameters, loss = train_one_epoch(X, y, parameters, lr=0.1)
            loss_history.append(loss)

            # if epoch % 50 ==0:
            #     print(f'Loss after {epoch} epochs: {loss}')
            #     # stopping criterion
            #     if loss < 0.005:
            #         break
            if epoch == 25:
                plt.plot(loss_history, label=f'd_h={hh}')

        error.append(loss)
    plt.legend()
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/3.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 16),  error)
    plt.xlabel('d_h', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/error.png')
    plt.show()

    # # the contrast between y_pred and y_true of final epoch
    # storage = forward_popagation(X, parameters)
    # O = storage['O']
    # # print('y_pred: -------------------------------')
    # # print(f'{O}');print()
    # # print('y_true: -------------------------------')
    # # print(f'{y}');print()
    # cls_YorN(O, y)

    # # test in the dataset includes noise
    # X_noise = []
    # y_noise = []
    # for index,i in enumerate(X):
    #     for num in range(2):
    #         a = 0.5*(np.random.rand(2)-0.5)         # (-0.25, 0.25)
    #         X_noise.append(i + a)
    #         if index // 3 == 0:
    #             y_noise.append([1, -1, -1])
    #         elif index // 3 == 1:
    #             y_noise.append([-1, 1, -1])
    #         else:
    #             y_noise.append([-1, -1, 1])
    # XX = np.array(X_noise)
    # YY = np.array(y_noise)

    # storage_test = forward_popagation(XX, parameters)
    # O_test = storage_test['O']
    # cls_YorN(O_test, YY)


if __name__ == '__main__':
    main()
