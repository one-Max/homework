import numpy as np
from numpy import *
# x = np.array([0,1,2,3])
# y = np.array([1,2,3,4])
# print(np.dot(x,y))
# for i, j in ((0.1, 1.2), (2.3, 539)):
#     print(i)
#     print(j)
#
# print(np.arange(-10,10,0.5))
#
# xdim = [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
# ldim = [1, 0, 0, 0]
# for  x,l in zip(xdim, ldim):
#     print(x,l)
#
# print(square(5-2))
#
# x = [1, -1]
# x1 = np.array([i for i in x])
# x1 = np.append(x1,[1])
# print(x1)
C = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]])
H = np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]])
L = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]])

a = np.reshape(C,25)
print(np.reshape(a,[5,5]))
print(30//25)

t = [1,2,3,4,5,6]
print(t[:5])

# labels = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
# labels2 = [[1, 9, -1], [8, 1, -1], [-1, -1, 5]]
# labels3 = [[],[],[]]
# for index, i in enumerate(labels):
#     labels3[index] = labels2[index] + i
#
# print(labels3)

for index, i in enumerate([]):
    print('ok')