import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pdb

#------------------------------------------------------------
#--------      Generalized RBF Neural Network       ---------
#------------------------------------------------------------

torch.manual_seed(0)


class RBFN(nn.Module):
    
    def __init__(self, *args, d_i=None, d_h=None, d_o=3, center_assigh=False):
        super().__init__()
        self.d_o = d_o
        self.num_centers = d_h
        self.dim_centers = d_i
        # center of RBF kernel
        if center_assigh:
            self.centers = nn.Parameter(torch.as_tensor(*args))
        else:
            self.centers = nn.Parameter(torch.rand(self.num_centers, self.dim_centers))
        self.sigmas = nn.Parameter(0.5*torch.ones(1, self.num_centers), requires_grad=True)
        # linear projection from hidden layer to output layer
        self.Linear = nn.Linear(in_features=self.num_centers, 
                                out_features=self.d_o, 
                                bias=True)
        # initialize weight of linear projection layer
        self.Linear.weight.data.normal_(0, 0.02)
        self.Linear.bias.data.zero_() 

    def forward(self, batches):
        N = batches.size(0)

        A = batches.unsqueeze(1).repeat(1, self.num_centers, 1) # 9x2 --> 9x1x2 --> 9x5x2
        B = self.centers.repeat(N, 1, 1)        # 5x2 --> 9x5x2
        H = torch.exp(-torch.sum((A-B)**2, dim=2) / self.sigmas**2)
        # print(H)
        O = self.Linear(H)
        return O

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)


def cls_YorN(O, y):
    flag = []
    for i, j in zip(O, y):
        k = 0
        for n in i*j:
            if n < 0:
                k += 1
        if k > 0:
            flag.append(1)
        else:
            flag.append(0)
    print(f'The misclassification flags are {flag}')
    print(f'Num of misclassified sample: {sum(flag)} out of {len(y)}')
    print(f'Train accuracy: {(len(flag)-sum(flag))/len(flag):.7f}')


def main():
    X = torch.tensor([[0.25, 0.5],
                    [0.0, 0.25],
                    [0.75, 0.0],
                    [0.0, 0.0],
                    [0.75, 0.5],
                    [1.0, 0.5],
                    [0.75, 0.75],
                    [1.0, 0.25],
                    [0.0, 0.75]], dtype=torch.float32)
    y = torch.tensor([[1, -1, -1],
                    [1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, 1],
                    [-1, -1, 1]], dtype=torch.float32)
    centers = X[[1,4,7], :]
    # pdb.set_trace()
    rbf = RBFN(centers, d_i=2, d_h=3, d_o=3, center_assigh=True)
    rbf.print_network()
    
    parameters = rbf.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)

    num_epochs = 5000
    loss_history = []

    for epoch in range(num_epochs):
        # zero the parameter gradients of the last batch
        optimizer.zero_grad()
        O = rbf.forward(X)
        loss = loss_fn(O, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().numpy())

        if epoch % 50 == 0:
            print(f'Loss after {epoch:>5} epochs: {loss:.7f}')
            # stopping criterion
            if loss < 0.005:
                break
    
    # loss visulization
    # plt.figure(figsize=(10, 6))
    # plt.plot(loss_history)
    # plt.xlabel('epoch', fontsize=15)
    # plt.ylabel('Loss', fontsize=15)
    # plt.grid(linestyle='--')
    # plt.tight_layout()
    # plt.savefig('/data1/zjw/homework/ann_hw2/99.png')
    # plt.show()

    # test
    O = rbf.forward(X)
    cls_YorN(O.detach().numpy(), y.numpy())
    # print(O.data)
    # print(y.data)


if __name__ =='__main__':
    main()
        