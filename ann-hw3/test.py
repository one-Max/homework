import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

ss = np.array([[1,2,5,0,2]])
ee = np.array([[2,3,7,8,2]])
# print(ss @ ee)
# print(np.random.rand(3, 1))
a = np.concatenate((ss,ee), axis=0)
# print(a)
# a = np.array([1,0,3]).reshape(-1,1)
# print([np.where(mark == np.max(mark))[0][0] for mark in a.T])
cc = np.zeros(10)

X = np.random.random(1000)
Y = np.random.random(1000)

point_group = []

for x, y in zip(X, Y):
    if y <= (3**0.5)*x and y <= 3**0.5-(3**0.5)*x and y >= 0:
        point_group.append([x, y])
id_list = np.random.randint(0, len(point_group)-1, 100)
point_group = np.array(point_group)
sample_data = point_group[id_list]

# plt.figure()
# plt.scatter(sample_data[:,0], sample_data[:,1])
# plt.margins(0.2)
# plt.show()


for i in range(2, 3):
    print(i)
print(np.random.random((100,100)))

q = np.ones((8,2))
q[1] = np.array([1,1])
q[2] = np.array([5,7])
q[5] = np.array([2,9])
qq = q.reshape(2, 4, 2)
print(qq)
print('===================')
print(qq.transpose(2,1,0))
# print(np.stack((np.random.random((5,5)),
#       np.random.random((5, 5)))).T)

