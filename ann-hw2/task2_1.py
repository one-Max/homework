import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import dot, exp, sum, ones, square
from mpl_toolkits.mplot3d import axes3d

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
        # H = torch.relu
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
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_() 


def peak(x, y):
    z = 3 * (1-x)**2 * exp(-(x**2)-(y+1)**2) - 10 * \
        (x/5-x**3-y**5) * exp(-x**2-y**2) - 1/3*exp(-(x+1)**2-y**2)
    return z

def main():

    
    #------------------------------------------------------------
    # generate dataset
    np.random.seed(1)
    # SAMPLE_NUMBER = 200  
    # xs = np.random.uniform(-4, 4, SAMPLE_NUMBER)
    # ys = np.random.uniform(-4, 4, SAMPLE_NUMBER)
    sample_num1 = 100
    sample_num2 = 100
    np.random.seed(1)
    xs1 = np.random.uniform(-4, 4, sample_num1)
    ys1 = np.random.uniform(-4, 4, sample_num1)
    xs2 = np.random.uniform(-2.5, 2.5, sample_num2)
    ys2 = np.random.uniform(-2.5, 2.5, sample_num2)
    xs = xs1 + xs2
    ys = ys1 + ys2

    zs = peak(xs, ys)

    x_train = np.array(list(zip(xs, ys)))
    label_train = zs.reshape(-1, 1)
    # print(x_train.shape, y_train.shape)

    #------------------------------------------------------------
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(label_train, dtype=torch.float32)
                
    bp = BPN(d_i=2, d_h=20, d_o=1)
    bp.print_network()
    
    parameters = bp.parameters()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=0.05, momentum=0.9)

    num_epochs = 5000
    loss_history = []

    for epoch in range(num_epochs):
        # zero the parameter gradients of the last batch
        optimizer.zero_grad()
        O = bp.forward(X)
        loss = loss_fn(O, Y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.detach().numpy())

        if epoch % 50 == 0:
            print(f'Loss after {epoch:>5} epochs: {loss:.7f}')
            # stopping criterion
            if loss < 0.01:
                break
    
    # loss visulization
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/44.png')
    plt.show()

    #------------------------------------------------------------
    # regression visulization
    x = np.arange(-4, 4, 0.05)
    y = np.arange(-4, 4, 0.05)

    xx, yy = np.meshgrid(x, y)
    zz = peak(xx, yy)

    zz_pred = []
    for xxx, yyy in zip(xx, yy):
        xy3 = np.array(list(zip(xxx, yyy)))
        O = bp.forward(torch.tensor(xy3, dtype=torch.float32))
        zz_pred.append(O.detach().numpy().reshape(1,-1)[0])
    zz_p = np.array(zz_pred)

    #------------------------------------------------------------
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, zz_p, cmap='coolwarm', linewidth=0,
                    antialiased=False, alpha = 0.7)

    plt.contour(xx, yy, zz)
    # ax.scatter(xs, ys, zs, color = 'r')

    ax.set_xlabel('X Axes')
    ax.set_ylabel('Y Axes')
    ax.set_zlabel('Z Axes')
    plt.savefig('/data1/zjw/homework/ann_hw2/5.png')
    plt.show()

if __name__ =='__main__':
    main()
        