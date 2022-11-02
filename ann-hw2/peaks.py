from numpy import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X = np.arange(-4, 4, 0.05)
Y = np.arange(-4, 4, 0.05)
X, Y = np.meshgrid(X, Y)


def fxy(x, y):
    e1 = exp(-(x**2+(y+1)**2))
    e2 = exp(-(x**2+y**2))
    e3 = exp(-((x+1)**2+y**2))

    fvalue = 3*(1-x)**2*e1 - 10*(x/5-x**3-y**5)*e2 - 1/3*e3
    return fvalue


Z = fxy(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax.set_zlim(-8, 8)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
