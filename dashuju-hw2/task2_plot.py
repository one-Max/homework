import sys, os, math, time, copy
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
            # print(param)
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)


class MyLoss(nn.Module):
    """
    自定义损失函数: homework2.pdf中关于损失函数的定义不准确
    match_loss应除以归一化系数length
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, X_pred, X_gt, U, V, A, lam, length):
        match_loss =  (1/length) * 0.5 * torch.sum(torch.pow((torch.mul(A, (X_gt-X_pred))), 2)) 
        norm_loss =  lam *(torch.sum(torch.pow(U, 2)) + torch.sum(torch.pow(V, 2)))
        return match_loss + norm_loss


def main():
    # ==================================================
    # 载入矩阵数据
    X_train = torch.tensor(
        np.load('/data1/zjw/homework/dashuju-hw2/data/x_train.npy'), dtype=torch.float32).cuda()
    X_test = np.load('/data1/zjw/homework/dashuju-hw2/data/x_test.npy')

    # ==================================================
    # 参数初始化设置
    user_dim = 10000; film_dim = 10000
    lr = 0.1
    num_epochs = 500
    k = 20          # U，V矩阵降到m(n)xk维
    lam = 0.0001      # 正则项系数lambda

    # ==================================================
    # 获得A矩阵和计算RMSE时的归一化系数lenth
    rows, cols = np.nonzero(X_test)
    A = np.zeros((10000, 10000))
    A[rows, cols] = 1
    length = len(rows)
    A = torch.tensor(A, dtype=torch.float32).cuda()

    # ==================================================
    # 初始化UV参数，损失函数，优化器
    uv = UV(user_dim, film_dim, k).cuda()
    uv.print_network()
    parameters = uv.parameters()
    loss_fn = MyLoss()
    optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)

    X_test = torch.tensor(X_test, dtype=torch.float32).cuda()

    # ==================================================
    # 梯度下降得到U, V
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        X_pred = uv.forward()
        loss = loss_fn(X_pred, X_train, uv.U, uv.V, A, lam, length)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        
        test_e = torch.pow((X_test - torch.mul(A, X_pred)), 2)
        test_l = torch.pow((torch.sum(test_e) / length), 0.5)
        test_loss.append(test_l.item())

        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], train Loss: {:.4f}, test loss: {:.4f}'.format(epoch +
              1, num_epochs, loss.item(), test_l.item()))

    # ==================================================
    # 目标函数值、测试误差可视化 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,num_epochs+1), train_loss, label='train loss')
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/dashuju-hw2/train-loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), test_loss, label='test loss')
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/dashuju-hw2/test-loss.png')


if __name__ == '__main__':
    main()