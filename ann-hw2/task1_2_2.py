import sys, os, math, time, copy
import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, exp, sum, ones, square
from scipy.linalg import pinv

#------------------------------------------------------------
#--------       Normalized RBF Neural Network       ---------
#------------------------------------------------------------


#------------------------------------------------------------
# calculate the rbf hide layer output
# x: Input vector
# H: Hide node : row->sample
# sigma: variance of RBF

def rbf_hide_out(x, H, sigma):
    Hx = H - x
    Hxx = [exp(-dot(e,e)/(sigma**2)) for e in Hx]       # 9x1

    return Hxx

#------------------------------------------------------------
# if the sample is misclassified
def cls_YorN(O, y):
    flag = []
    for i, j in zip(O, y):
        k = 0
        for n in i*j:
            if n < 0:
                k += 1
        if k > 0:
            flag.append(1)
        else:
            flag.append(0)
    print('flag = 1 if this sample is misclassified')
    print(f'the misclassification flags are {flag}')
    print(f'num of misclassified sample: {sum(flag)} out of {len(y)}')
    print()

def main():
    # dataset
    X = np.array([[0.25, 0.5],
                  [0.0, 0.25],
                  [0.75, 0.0],
                  [0.0, 0.0],
                  [0.75, 0.5],
                  [1.0, 0.5],
                  [0.75, 0.75],
                  [1.0, 0.25],
                  [0.0, 0.75]])
    y = np.array([[1, -1, -1],
                  [1, -1, -1],
                  [1, -1, -1],
                  [-1, 1, -1],
                  [-1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [-1, -1, 1],
                  [-1, -1, 1]])

    # the hidden layer output 
    # columm: h_i col-vector corresponds to i_th sample [9x1]
    # row: j_th hidden layer neuron
    Hdim = np.array([rbf_hide_out(x, X, 0.5) for x in X]).T     # 9x9
    
    # get W without iteration
    W = dot(y.T,pinv(np.eye(9) * 0.001 + Hdim))                 # 3x9 9x9 --> 3x9

    # y_pred
    O = dot(W, Hdim)                                            # 3x9 9x9 --> 3x9

    # contrast result
    cls_YorN(O.T, y)
    


if __name__ == '__main__':
    main()