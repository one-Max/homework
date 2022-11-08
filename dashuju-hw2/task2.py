import sys, os, math, time, copy
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.system('echo $CUDA_VISIBLE_DEVICES')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

class UV(nn.Module):
    def __init__(self, user_dim, film_dim, k):
        super().__init__()
        self.user_dim = user_dim
        self.film_dim = film_dim
        self.k = k
        self.U = nn.Parameter(torch.rand(self.user_dim, self.k))
        self.V = nn.Parameter(torch.rand(self.film_dim, self.k))
    
    def forward(self):
        X_pred = torch.matmul(self.U, self.V.T)
        return X_pred

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)


def main():
    # ==================================================
    # 载入矩阵数据
    X_train = torch.tensor(
        np.load('/data1/zjw/homework/dashuju-hw2/data/x_train.npy'), dtype=torch.float32).cuda()
    X_test = np.load('/data1/zjw/homework/dashuju-hw2/data/x_test.npy')

    # ==================================================
    # 参数初始化设置
    user_dim = 10000; film_dim = 10000
    k = 50
    lr = 1
    num_epochs = 100

    # ==================================================
    # 初始化UV参数
    uv = UV(user_dim, film_dim, k).cuda()
    uv.print_network()

    parameters = uv.parameters()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    
    # ==================================================
    # 梯度下降得到U， V
    loss_history = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X_pred = uv.forward()
        loss = loss_fn(X_pred, X_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # ==================================================
    # 在测试集上得到RMSE
    predicted = uv.forward().cpu().detach().numpy()
    
    RMSE = 0
    rows, cols = np.nonzero(X_test)
    for row, col in zip(list(rows), list(cols)):
        RMSE += (predicted[row][col] - X_test[row][col]) ** 2
    RMSE = (RMSE / len(rows)) ** 0.5
    print(RMSE)


if __name__ == '__main__':
    main()