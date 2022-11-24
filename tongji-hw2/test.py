import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, Normalizer, scale

X = np.array([[ 1., -1.,  2.],
            [ 2.,  0.,  0.],
            [ 0.,  1., -1.]])

print(normalize(X, norm='l2', axis=0))
print(normalize(X, norm='l2', axis=1))
print(scale(X, axis=0))
print(scale(X, axis=1))
