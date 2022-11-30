import random
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import time

# bin()转化为二进制数，开头是0b
print(bin(16165165))
print("-----------------------------------------")

# list == 1 生成 一个同样大小的list,元素是true or false，可以借此改变list中符合某条件的值
a = np.array([1, 1, 11 ,56, 4, 5641 , 68542, 5, 0, 5, 3, 0, 0, 0])
b = np.ones(a.shape)
a[a == 1]=999
print(a == 1)
print(a)
print("-----------------------------------------")

# python 内置all()，判断两个一模一样大小的矩阵或列表是否所有元素都相等，若是则返回True
temp = np.array([1, 2, 3])
ss = np.array([1, 2, 3])
if ((temp==ss).all()):
    print('Yes, they are equal')

print(f'judgement:{(temp==ss).all()}')
print(f'judgement:{all(temp==ss)}')
print("-----------------------------------------")

# list1 == list2 生成一个同样大小的list，返回的是里面每项是否相等，相等返回True
print(temp==ss)
print("-----------------------------------------")

# np.where()返回符合这个条件的ss里的值 对应的索引位置
print(np.where(ss==3)[0])
print(ss[np.where(ss==3)])
print("-----------------------------------------")

# 符号函数np.sign(), + -> 1; - -> -1; 0 -> 0
print(np.sign(0.000))
print("-----------------------------------------")


# 一个变量赋给另一个变量，他们存储的地址一致，id一致
r = b
print(id(r), id(b))
# 一个list的一个元素赋给另一个变量，深拷贝，id不一致
rr = np.array([[1, 2, 3],[2,3,4]])
x = rr[0]
print(id(x), id(rr[0]))
print("-----------------------------------------")

# np.squeeze()暂时理解成去掉外层不必要的维度
q = np.ones((8,2))
print([q[:,0]])
print(np.squeeze([q[:,0]]))
print("-----------------------------------------")

# iter()函数生成迭代器，他是一次性的所有元素依序只能调用一次
r = np.random.rand(10)
re = np.array([1 if a > 0.5 else 0 for a in r])

wwbb = list(zip(*([iter(re)]*2)))
wb = [r for r in wwbb]

print(wb)
print(np.split(re, 5))

for i in zip(*([iter(re)]*3)):
    print(i)

a = [3,4,5,6,7]
b = iter(a)
for i in b:
    print(i)
    if i==5:
        break
print()
for i in b:
    print(i)

print("-----------------------------------------")



"""
#------------------------------------------------------------
wb_2 = rand01()             # 生成一个随机0 1向量，代表全部权值的编码   9x8=72
wb = transform(wb_2)        # 将二值编码的权值换源为十进制数    1x9
print('encoded vetor:')
print(wb_2)
print('transform to int')
print(f'{wb}')

yout = xornnout(wb)     # 前向输出预测结果  1x4
print(yout)
print(xorerr(yout))     # MSE误差
print(a01C(wb_2))       # 适应度

#------------------------------------------------------------
aa = [rand01(20) for _ in range(4)]      #4x72的一个种群
print('population:')
print(np.array(aa))
print()

sa = GA_mutate(aa)                      # 变异
print('mutation:')
print(np.array(sa))
print()

a = [[i]*20 for i in range(4)]         # 10x20的不知道啥，为了演示交叉是怎么进行的
print('population:')
print(np.array(a))
print()

aa = GA_cross(a, 0.5)
print('crossover:')
print(np.array(aa))

"""
