import random
import clipboard
import winsound
import math
import time
import matplotlib.pyplot as plt
# import tsmodule.tsconfig
from tsdraw import *
# from tsmodule.tsdopop import *
# from tsmodule.tscmd import *
# from tsmodule.tspyt import *
# from tsmodule.tspdata import *
from numpy import *
from threading import Thread
import sys
import os
sys.path.append(r'd:\python\teasoft')
STDFILE = open(r'd:\python\std.txt', 'a', 1)
sysstderr = sys.stderr
sysstdout = sys.stdout
sys.stderr = STDFILE
sys.stdout = STDFILE


def setpltrange(posx=2000, posy=550, width=800, height=640):
    cmfw = plt.get_current_fig_manager().window
    cmfw.setGeometry(posx, posy, width, height)


setpltrange()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
