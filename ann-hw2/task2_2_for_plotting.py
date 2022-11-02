import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import dot, exp, sum, ones, square
from mpl_toolkits.mplot3d import axes3d

torch.manual_seed(0)

class RBFN(nn.Module):

    def __init__(self, *args, d_i=None, d_h=None, d_o=None, center_assigh=False):
        super().__init__()
        self.d_o = d_o
        self.num_centers = d_h
        self.dim_centers = d_i
        # center of RBF kernel
        if center_assigh:
            self.centers = nn.Parameter(torch.as_tensor(*args))
        else:
            self.centers = nn.Parameter(torch.rand(self.num_centers, self.dim_centers))
        
        # -----------the sigmas participate the iteration-----------
        # self.sigmas = nn.Parameter(1*torch.ones(1, self.num_centers), requires_grad=True)

        # -------------------the sigmas are fixed-------------------
        # self.sigmas = 1.5*torch.ones(1, self.num_centers)

        # linear projection from hidden layer to output layer
        self.Linear = nn.Linear(in_features=self.num_centers,
                                out_features=self.d_o,
                                bias=True)
        # initialize weight of linear projection layer
        self.Linear.weight.data.normal_(0, 0.02)
        self.Linear.bias.data.zero_()

    def forward(self, batches):
        N = batches.size(0)

        A = batches.unsqueeze(1).repeat(1, self.num_centers, 1)  
        B = self.centers.repeat(N, 1, 1)        
        H = torch.exp(-torch.sum((A-B)**2, dim=2) / self.sigmas**2)
        O = self.Linear(H)

        return O

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)


def peak(x, y):
    z = 3 * (1-x)**2 * exp(-(x**2)-(y+1)**2) - 10 * \
        (x/5-x**3-y**5) * exp(-x**2-y**2) - 1/3*exp(-(x+1)**2-y**2)
    return z


def main():
    #------------------------------------------------------------
    # generate dataset
    SAMPLE_NUMBER = 200
    np.random.seed(1)
    xs = np.random.uniform(-4, 4, SAMPLE_NUMBER)
    ys = np.random.uniform(-4, 4, SAMPLE_NUMBER)
    zs = peak(xs, ys)

    x_train = np.array(list(zip(xs, ys)))
    label_train = zs.reshape(-1, 1)
    # print(x_train.shape, y_train.shape)

    #------------------------------------------------------------
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(label_train, dtype=torch.float32)

    accu = []
    plt.figure(figsize=(10, 6))
    for pp in np.arange(0.1, 1.6, 0.1):
        centers = X[0:20, :]
        rbf = RBFN(centers, d_i=2, d_h=20, d_o=1, center_assigh=True)
        rbf.sigmas = pp * torch.ones(1, rbf.num_centers)
        # rbf.print_network()
        
        parameters = rbf.parameters()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)

        num_epochs = 5000
        loss_history = []

        for epoch in range(num_epochs):
            # zero the parameter gradients of the last batch
            optimizer.zero_grad()
            O = rbf.forward(X)
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
        plt.plot(loss_history, label=f'sigmas={pp:.1f}')
    plt.legend()
    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig('/data1/zjw/homework/ann_hw2/000.png')
    plt.show()

    # #------------------------------------------------------------
    # # regression visulization
    # x = np.arange(-4, 4, 0.05)
    # y = np.arange(-4, 4, 0.05)
    # xx, yy = np.meshgrid(x, y)
    # zz = peak(xx, yy)

    # zz_pred = []
    # for xxx, yyy in zip(xx, yy):
    #     xy3 = np.array(list(zip(xxx, yyy)))
    #     O = rbf.forward(torch.tensor(xy3, dtype=torch.float32))
    #     zz_pred.append(O.detach().numpy().reshape(1,-1)[0])
    # zz_p = np.array(zz_pred)

    # #------------------------------------------------------------
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(xx, yy, zz_p, cmap='coolwarm', linewidth=0, antialiased=False)
    # plt.contour(xx, yy, zz)
    # ax.scatter(xs, ys, zs, color = 'r')

    # ax.set_xlabel('X Axes')
    # ax.set_ylabel('Y Axes')
    # ax.set_zlabel('Z Axes')
    # plt.savefig('/data1/zjw/homework/ann_hw2/8.png')
    # plt.show()


if __name__ =='__main__':
    main()
        