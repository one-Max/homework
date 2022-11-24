import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import random
from scipy.stats import multivariate_normal
import copy
import pdb

def show_result(gmm, data, path, with_ori=True):
    '''
    Draw the data points and the fitted mixture model.
    input:
        - title: title of plot and name with which it will be saved.
    '''
    fig = plt.figure(figsize=(8, 8))
    pred = gmm.z.argmax(-1)
    pred = pred.reshape(data.shape)

    colors = ['w', 'b', 'k']

    cmap = mpl.colors.ListedColormap(colors)
    plt.figure('show_result')
    if with_ori:
        plt.imshow(data, cmap='gray')
    plt.imshow(pred, interpolation='none', cmap=cmap, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)

class GMM():
    def __init__(self, dim, k, mu=None, sigma=None, pr=None):
        self.k = k
        self.dim = dim
        self.mu = mu
        self.pr = pr
        if(sigma is None):
            sigma = np.zeros((k, dim, dim))
            for i in range(k):
                sigma[i] = np.eye(dim)
        self.sigma = sigma
        self.colors = random.rand(k, 3)

    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        '''
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    # Expectation
    def expectation(self):
        '''
        E-step of EM algorithm.
        '''
        for i in range(self.k):
            # self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
            self.z[:, i] = self.pr[i] * (1 / np.sqrt(2*np.pi*self.sigma[i])) * np.exp(-0.5*(self.data-self.mu[i])**2 / self.sigma[i])[...,0]
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    # Maximization
    def maximization(self):

        sum_z = self.z.sum(axis=0)
        self.pr = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]

    # 计算对数似然概率
    def log_likelihood(self, X):
        log_like = []
        for d in X:
            tot = 0
            for i in range(self.k):
                # tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
                tot += self.pr[i] * (1 / np.sqrt(2*np.pi*self.sigma[i])) * np.exp(-0.5*(d-self.mu[i])**2 / self.sigma[i])
            log_like.append(np.log(tot))

        return np.sum(log_like)

def main():
    # load brainimage
    data = np.loadtxt(r'/data1/zjw/homework/tongji-hw2/data/brainimage.txt')
    plt.imshow(data, cmap='gray')
    plt.savefig('vis_brain.png')

    # 归一化
    origin_data = copy.deepcopy(data)
    data = (data-data.min())/(data.max()-data.min())
    data = data.reshape(-1,1)

    mu = np.array([0.3, 0.5, 0.7]).reshape(-1,1)
    pr = np.array([1/3, 1/3, 1/3])
    gmm = GMM(1, 3, mu=mu, pr=pr)

    # Initialize EM algo with data
    gmm.init_em(data)
    num_iters = 25
    # Saving log-likelihood
    log_likelihood = [gmm.log_likelihood(data)]
    # plotting
    for e in range(num_iters):
        # E-step
        gmm.expectation()
        # M-step
        gmm.maximization()
        # Computing log-likelihood
        log_likelihood.append(gmm.log_likelihood(data))
        print("Iteration: {}, log-likelihood: {:.4f}".format(e+1, log_likelihood[-1]))
        if e % 10 == 0:
            show_result(gmm, origin_data, 'GMM_result_'+str(e)+'.png')

    # Plot log-likelihood
    plt.figure(figsize=(10,6))
    plt.plot(log_likelihood[1:], marker='.')
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('log likelihood', fontsize=15)
    plt.tight_layout()
    plt.savefig('GMM_log_results.png')

    show_result(gmm, origin_data, 'GMM_result_final.png')
    show_result(gmm, origin_data, 'GMM_result_final_noori.png', with_ori=False)

if __name__ == "__main__":
    main()

