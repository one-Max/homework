import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi


#------------------------------------------------------------

def shownet(s, w, title):
    # plt.clf()
    plt.scatter(s[:, 0], s[:, 1], c='b', s=20.0, alpha=1, label='train data')
    plt.scatter(w[:, 0], w[:, 1], c='r', s=30.0, alpha=1, label='W')
    a = np.linspace(0, 2*pi, 100)
    plt.plot(cos(a), sin(a), 'g--', linewidth=1)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

#------------------------------------------------------------

def WTA1(x, W):
    """ Win-Take-All
    In: x-sample(x1,x2)
           w-net argument
    Ret: id-Win ID of w
    """
    innerdot = W @ x
    return np.where(innerdot == np.max(innerdot))[0][0]


def normvect(v):
    
    return v/np.sqrt(v @ v)

#------------------------------------------------------------

def compete(X, W, lr=0.1):
    for x in X:
        win_id = WTA1(x, W)
        w_update = W[win_id] + lr * (x-W[win_id])
        W[win_id] = normvect(w_update)

    return W
 
#------------------------------------------------------------

def main():
    # 获得样本数据
    sample_angle = np.array([-5, 5, 10, 75, 115, 210, 240, 300])
    sample_xy = np.array([[cos(a*pi/180), sin(a*pi/180)]
                        for a in sample_angle]).astype('float32')

    # 参数设置及初始化，设置竞争神经元的初始权值
    W12_angle = [-45, 270, 30]
    # W12_angle = np.random.randint(0, 360, 3)
    # W12_angle = sample_angle[np.random.randint(0,len(sample_angle)-1,3)]

    W12 = np.array([[cos(b*pi/180), sin(b*pi/180)]
                for b in W12_angle]).astype('float32')
    xy = sample_xy
    num_epochs = 100
    learing_rate = 0.2
    
    # train
    plt.rcParams['figure.figsize'] = (5, 5)
    plt.draw()
    plt.pause(0.2)

    plt.scatter(W12[:, 0], W12[:, 1], c='y', s=80.0, label='W original', alpha=0.7)

    for epoch in range(num_epochs):
        # # 变学习率，线性下降
        lr = learing_rate * (1 - epoch / 99)

        # 变学习率，等比下降
        # if epoch != 0:
        #     lr = 0.75 * lr
        # else:
        #     lr = learing_rate

        W12 = compete(xy, W12, lr)
        # break

    shownet(sample_xy, W12, 'Step:%d, η=%4.2f' % (epoch+1, lr))
        
    print('w1=%s' % W12[0])
    print('w2=%s' % W12[1])
    plt.savefig('homework/ann-hw3/picture/1-1.png')
    plt.show()


if __name__ == '__main__':
    main()
