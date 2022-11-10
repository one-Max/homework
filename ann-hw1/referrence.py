import sys,os,math,time
import matplotlib.pyplot as plt
from numpy import *

xdim = [(-0.1,0.3), (0.5,0.7), (-0.5,0.2),(-0.7,0.3),(0.7,0.1),(0,0.5)]
ddim = [1,-1,1,1,-1,1]

def sigmoid(x):
    return 1/(1+exp(-x))

def hebbian(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = sigmoid(net)
    w1 = [ww+o*xx for ww,xx in zip(w,x1)]
    return w1

def perceptron(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = 1 if net >= 0 else -1
    w1 = [ww+(d-o)*xx for ww,xx in zip(w,x1)]
    return w1

def delta(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = sigmoid(net)
    o1 = o*(1-o)
    w1 = [ww+(d-o)*o1*xx for ww,xx in zip(w,x1)]
    return w1

def widrawhoff(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = sigmoid(net)
    w1 = [ww+(d-o)*xx for ww,xx in zip(w,x1)]
    return w1

def correlation(w,x,d):
    x1 = [1,x[0],x[1]]
    w1 = [ww+d*xx for ww,xx in zip(w,x1)]
    return w1

wb = [0,0,0]                        # [b, w1, w2]
epochs = 5
for epoch in range(epochs):
    for x,d in zip(xdim, ddim):
        wb = delta(wb,x,d)
    print(wb)
