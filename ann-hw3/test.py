import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

sample_angle = [-5, 5, 10, 75, 115, 210, 240, 300]
sample_xy = np.array([[cos(a*pi/180), sin(a*pi/180)]
                     for a in sample_angle]).astype('float32')

W12_angle = [-45, 270, 30]
W12 = np.array([[cos(b*pi/180), sin(b*pi/180)]
               for b in W12_angle]).astype('float32')

sxy = sample_xy
# print(sxy)
# np.random.shuffle(sxy)
# print(sxy)

# print(W12)
# print(sxy[0].T)
# a = W12 @ (sxy[0].reshape(-1,1))
# b = W12 @ sxy[0]
# print(a[0])
# print(a[0][0])
# print(b[0])

ss = np.array([[1,2,5,0,2]])
ee = np.array([[2,3,7,8,2]])
# print(ss @ ee)
# print(np.random.rand(3, 1))
a = np.concatenate((ss,ee), axis=0)
print(a)
# a = np.array([1,0,3]).reshape(-1,1)
print([np.where(mark == np.max(mark))[0][0] for mark in a.T])
cc = np.zeros(10)


plt.figure
plt.savefig('homework/ann-hw3/picture/jieguo2.png', dpi=1000)
plt.show()
