import random
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import time

print(bin(16165165))
a = np.array([1, 1, 11 ,56, 4, 5641 , 68542, 5, 0, 5, 3, 0, 0, 0])
b = np.ones(a.shape)
a[a == 1]=999
print(a)

# python 内置all()
temp = np.array([1, 2, 3])
ss = np.array([1, 2, 3])
if ((temp==ss).all()):
    print('sss')

print((temp==ss).all())
print(temp==ss)
print(all(temp==ss))
print(ss[np.where(ss==2)])
print(np.sign(0.000))
