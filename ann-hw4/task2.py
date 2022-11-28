import random
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from numpy import nan
from typing import Optional


# ------------------------------------------------------------
# 生成一个随机0 1向量，代表全部权值的编码
def rand01(l=72):
    r = np.random.rand(l)
    rr = [1 if a > 0.5 else 0 for a in r]
    return np.array(rr)


# ------------------------------------------------------------
# 二进制数 - 十进制数
def bit8(r):
    bytestr = ''.join(['%d'%rr for rr in r])
    n = int(bytestr, 2)
    if n >= 128:
        n -= 256
    return n


# ------------------------------------------------------------
# 二进制序列 - 十进制序列: 1x72 -> 1x9
def transform(wb):
    wwbb = np.split(np.array(wb), 9)
    wb = [bit8(r) for r in wwbb]
    return wb


# ------------------------------------------------------------
# 带放大倍数的sigmoid激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x/20))


# ------------------------------------------------------------
# 网络的前向，输出预测值
def forward(wb, x1, x2):
    v1 = sigmoid(x1*wb[0] + x2*wb[1] + wb[6])
    v2 = sigmoid(x1*wb[2] + x2*wb[3] + wb[7])
    y = sigmoid(v1*wb[4] + v2*wb[5] + wb[8])
    return y


# ------------------------------------------------------------
def xornnout(x, wb):
    return [forward(wb,a[0],a[1]) for a in x]


# ------------------------------------------------------------
def xorerr(y, y_pred):
    return sum([abs((x-y)**2) for x,y in zip(y, y_pred)])


# ------------------------------------------------------------
# 个体适应度函数: MSE -> C
def funcC(error):
    if error == 0: return 100000
    else: return 1/error


# ------------------------------------------------------------
def a01C(x, y, a01):
    wb = transform(a01)         # 将二值编码的权值换源为十进制数 1x9
    y_pred= xornnout(x, wb) # 前向输出预测结果 1x4
    e = xorerr(y, y_pred)   # 计算MSE误差
    return funcC(e)         # 给出适应度


# ------------------------------------------------------------
# 选择
def GA_select(x, y, a, num=0):
    if num == 0:
        num = len(a)

    ac = [a01C(x, y, aa) for aa in a]
    ac01sort = sorted(zip(ac, a), key=lambda x:x[0], reverse=True)
    return [b[1] for b in  ac01sort[:num]]


# ------------------------------------------------------------
# 变异
def GA_mutate(a, r=0.25):
    outa = []
    mutnum = int(len(a[0]) * r)

    for aa in a:
        s01 = [1] * len(aa)
        s01[:mutnum] = [-1]*mutnum
        np.random.shuffle(s01)
        newa = [((a*2-1)*b+1)//2 for a,b in zip(aa, s01)]
        outa.append(newa)

    return outa


# ------------------------------------------------------------
# 交叉
def GA_cross(a, r=0.25):
    num = len(a)
    crossnum = int(num*r)

    listA = list(range(num))
    listB = list(range(num))
    np.random.shuffle(listA)
    np.random.shuffle(listB)

    for i in range(crossnum):
        aa = listA[i]
        bb = listB[i]
        c1 = a[aa]
        c2 = a[bb]

        crossP = np.random.randint(0, len(a[0]))
        c1[crossP:],c2[crossP:] = c2[crossP:],c1[crossP:]

        a[aa] = c1
        a[bb] = c2

    return a


#------------------------------------------------------------
# 一步GA迭代
def GA_iterate(x, y, a, sr=0.8, mr=0.25, cr=0.25):
    sa = GA_select(x, y, a, int(len(a) * sr))
    am = GA_mutate(a, mr)
    amc = GA_cross(am, cr)
    amcs = GA_select(x, y, amc, len(a) - len(sa))

    sa.extend(amcs)
    return sa


#------------------------------------------------------------
def main():

    np.random.seed(0)

    # train data & label data
    X = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    y = np.array([0,1,1,0])

    # Parameters setting
    GA_SELECT_RATIO             = 0.2   # 选择比率
    GA_MUTATE_RATIO             = 0.8   # 变异比率
    GA_CROSS_RATIO              = 0.25  # 交叉比率
    GA_NUMBER                   = 100   # 种群个数
    GA_STEP                     = 100   # 演化步骤


    # train
    aall = [rand01() for _ in range(GA_NUMBER)]     # 100x72

    cdim = []       # 种群平均适应度
    for i in range(GA_STEP):
        aall = GA_iterate(X, y, aall)
        wb = transform(aall[0])
        y_pred = xornnout(X, wb)
        e = xorerr(y, y_pred)

        print(i, y_pred, e, funcC(e))

        edim = [a01C(X, y, na) for na in aall]
        ea = np.average(edim)
        cdim.append(ea)

    wb = transform(aall[0])
    y_pred = xornnout(X, wb)

    print('w1,w2,w3,w4,w5,w6:%d,%d,%d,%d,%d,%d'%(wb[0],wb[1],wb[2],wb[3],wb[4],wb[5]))
    print('b1,b2,b3:%d,%d,%d'%(wb[6], wb[7], wb[8]))
    print("Y:", y_pred)

    plt.plot(cdim)
    plt.xlabel("Step")
    plt.ylabel("FuncC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()