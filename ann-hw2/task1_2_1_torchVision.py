import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pdb

torch.manual_seed(0)


class BPN(nn.Module):
    
    def __init__(self, d_i=None, d_h=None, d_o=None):
        super().__init__()
        self.d_o = d_o
        self.d_h = d_h
        self.d_i = d_i

        self.Linear1 = nn.Linear(in_features=self.d_i, out_features=self.d_h, bias=True)
        self.Linear2 = nn.Linear(in_features=self.d_h, out_features=self.d_o, bias=True)
        self.initialize_weights()

    def forward(self, batches):
        A = self.Linear1(batches)
        H = 1 / (1 + torch.exp(-A))
        # print(H)
        O = self.Linear2(H)
        return O

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)
    
    # initialize weight of linear projection layer
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.zero_() 


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
    X = 2*(X-0.5)
    y = torch.tensor([[1, -1, -1],
                    [1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [-1, -1, 1],
                    [-1, -1, 1]], dtype=torch.float32)
                
    # pdb.set_trace()
    bp = BPN(d_i=2, d_h=5, d_o=3)
    bp.print_network()
    
    parameters = bp.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=0.5, momentum=0.9)

    num_epochs = 5000
    loss_history = []

    for epoch in range(num_epochs):
        # zero the parameter gradients of the last batch
        optimizer.zero_grad()
        O = bp.forward(X)
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
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/4.png')
    plt.show()

    # test
    O = bp.forward(X)
    cls_YorN(O.detach().numpy(), y.numpy())
    # print(O.data)
    # print(y.data)


if __name__ =='__main__':
    main()
        