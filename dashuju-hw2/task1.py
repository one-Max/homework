import sys, os, math, time, copy
import matplotlib.pyplot as plt
import numpy as np

# ==================================================
# 载入矩阵数据
X_train = np.load('/data1/zjw/homework/dashuju-hw2/data/x_train.npy')
X_test = np.load('/data1/zjw/homework/dashuju-hw2/data/x_test.npy')
# plt.spy(X_test)
# plt.savefig('/data1/zjw/homework/0.png')

t0 = time.perf_counter()

# ==================================================
# 协同过滤生成预测值
norm = np.sqrt(np.sum(X_train * X_train, axis=1)).reshape(-1, 1)
norm_matrix = norm @ norm.T
X1 = X_train @ X_train.T
cos_matrix = X1 / norm_matrix
weight = cos_matrix / np.sum(cos_matrix, axis=1).reshape(-1, 1)
X_pred = weight @ X_train

t1 = time.perf_counter()

# ==================================================
# 计算RMSE:只对那些X_test中非0的(非空)数据计算损失

loss_matrix = X_pred - X_test

RMSE = 0
rows, cols = np.nonzero(X_test)
for row, col in zip(list(rows), list(cols)):
    RMSE += loss_matrix[row][col] ** 2
RMSE = np.sqrt(RMSE / len(rows))

t2 = time.perf_counter()

print(f'RMSE: {RMSE:.8f}')
print(f'run time pred: {t1 - t0:.8f}s')
print(f'run time loss: {t2 - t1:.8f}s')
print(f'total run time: {t2 - t0:.8f}s')



