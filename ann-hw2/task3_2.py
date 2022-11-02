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
    A = dot(X, para['W_ih']) + dot(ones([m, 1]), para['b_h'])
    H = 1/(1 + exp(-A))

    B = dot(H, para['W_ho']) + dot(ones([m, 1]),
                                   para['b_o'])   
    O = B

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

    parameters = build_nn(d_i=len(X[0]), d_h=15, d_o=len(X[0]))
    """Different seed(different initial weight) leads to different loss curve"""

    # train
    loss_history = []
    num_epochs = 50000
    for epoch in range(num_epochs):
        parameters, loss = train_one_epoch(X, y, parameters, lr=0.05)
        loss_history.append(loss)
        
        if epoch % 50 ==0:
            print(f'Loss after {epoch} epochs: {loss}')
            # stopping criterion
            if loss < 0.005:
                break
    
    plt.figure(figsize=(10, 6))   
    plt.plot(loss_history)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/3.png')
    # plt.show()


    # the contrast between y_pred and y_true of final epoch
    storage = forward_popagation(X, parameters)
    O = storage['O']

    # print('y_pred: -------------------------------')
    # print(f'{O}');print()
    # print('y_true: -------------------------------')
    # print(f'{y}');print()
    # Show result
    plt.figure(figsize=(10, 3), dpi=150)
    for i in range(26):
        plt.subplot(2, 13, i + 1)
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.axis("off")
        plt.imshow(O[i].reshape(16, 8))
    plt.savefig('/data1/zjw/homework/ann_hw2/77.png')
    plt.show()


if __name__ == '__main__':
    main()
