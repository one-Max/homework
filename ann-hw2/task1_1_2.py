import sys, os, math, time, copy
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, exp, sum, ones, square

#------------------------------------------------------------
#--------        Neural Network architecture-2      ---------
#------------------------------------------------------------

#------------------------------------------------------------
# build the architecture of neural network
# input: neurons num of input , hidden layer, output
# output: middle result of hidden layer, output layer, initialized parameters matrix

def build_nn(d_i=None, d_h=None, d_o=None):
 
    W = np.random.random([d_i, d_h])            # 2x1
    V = np.random.random([d_i, d_o])            # 2x1
    V0 = np.random.random([d_h, d_o])           # 1x1
    b_h = -ones([1, d_h])                       # 1x1
    b_o = -ones([1, d_o])                       # 1x1

    parameters = {'W':W,
                  'V':V,
                  'V0':V0,
                  'b_h':b_h,
                  'b_o':b_o}

    return parameters 


#------------------------------------------------------------
# forward prpogation

def forward_popagation(X, para):
    m = X.shape[0]
    A = dot(X, para['W']) + dot(ones([m, 1]), para['b_h'])              # 4x2 2x1 + 4x1 1x1 --> 4x1
    H = 1/(1 + exp(-A))                                                 # 4x1

    B = dot(X, para['V']) + dot(H, para['V0']) + dot(ones([m, 1]), para['b_o'])   # 4x2 2x1 + 4x1 1x1 + 4x1 1x1 --> 4x1
    # O = 1/(1 + exp(-B))                                                 # 4x1
    O = B

    storage = {'A':A,
               'H':H,
               'B':B,
               'O':O}
    return storage


#------------------------------------------------------------
# calculate the loss

def cal_loss(storage, y):
    O = storage['O']
    m = y.shape[0]
    loss = sum((1 / (2*m)) * square(y-O))
    return loss.flatten()


#------------------------------------------------------------
# backward propogation:to calculate the gradient
# output: matrix derativate

def backward_propagation(para, storage, X, y):
    m = X.shape[0]
    V0 = para['V0']
    O = storage['O']
    H = storage['H']
    
    # d1 = (O - y) * (O * ( 1 - O))               # 4x1
    d1 = O - y
    dV0 = (1/m) * dot(H.T, d1)                  # 1x4 4x1 --> 1x1
    dV = (1/m) * dot(X.T, d1)                   # 2x4 4x1 --> 2x1
    db_o = (1/m) * dot(ones([1, m]), d1)        # 1x4 4x1 --> 1x1
 
    d2 = (dot(d1, V0.T)) * (H * (1 - H))        # 4x1
    dW = (1/m) * dot(X.T, d2)                   # 2x4 4x1 --> 2x1
    db_h = (1/m) * dot(ones([1, m]), d2)        # 1x4 4x1 --> 1x1

    grads = {'dW':dW,
            'dV':dV,
            'dV0':dV0,
            'db_o':db_o,
            'db_h':db_h}
    return grads


#------------------------------------------------------------
# update the parameters
# input: weight matrix, gradient, learning rate
# output: updated parameters

def update_parameters(para, grads, lr):
    W = para['W'] - lr * grads['dW']
    V = para['V'] - lr * grads['dV']
    V0 = para['V0'] - lr * grads['dV0']
    b_o = para['b_o'] - lr * grads['db_o']
    b_h = para['b_h'] - lr * grads['db_h']

    parameters = {'W':W,
                  'V':V,
                  'V0':V0,
                  'b_h':b_h,
                  'b_o':b_o}
    # for i in parameters:
    #     print(parameters[i])
    
    return parameters


#------------------------------------------------------------
# trainning process

def train_one_epoch(X, y, parameters, lr=0.5):
    storage = forward_popagation(X, parameters)
    loss = cal_loss(storage, y)
    grads = backward_propagation(parameters, storage, X, y)
    parameters = update_parameters(parameters, grads, lr)

    return parameters, loss


def main():
   
    # data:binary
    xor_x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])          # row->sample 4x2
    xor_y = np.array([0, 1, 1, 0]).reshape(-1,1)                # col->sample 4x1
    # data:bipolar
    xor_x0 = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])     # row->sample 4x2
    xor_y0 = np.array([-1, 1, 1, -1]).reshape(-1,1)             # col->sample 4x1
    
    # create model
    np.random.seed(0)
    parameters = build_nn(d_i=2, d_h=1, d_o=1)

    # train
    loss_history = []
    num_epochs = 10000
    for epoch in range(num_epochs):
        parameters, loss = train_one_epoch(xor_x, xor_y, parameters, lr=0.5)
        loss_history.append(loss)

        if epoch % 50 ==0:
            print(f'Loss after {epoch} epochs: {loss}')
            # stopping criterion
            if loss < 0.005:
                break
    
    # loss visulization
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/2.png')
    plt.show()
    
    # the contrast between y_pred and y_true of final epoch
    storage = forward_popagation(xor_x, parameters)
    O = storage['O']
    print('y_pred: ----------------------------')
    print(f'{O.flatten()}')
    print('y_true: ----------------------------')
    print(f'{xor_y.flatten()}')


if __name__ == '__main__':
    main()