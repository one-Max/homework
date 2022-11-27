import random
import numpy as np
import matplotlib.pyplot as plt
import os
import copy


#------------------------------------------------------------
chardim = []
filename = '/data1/zjw/homework/ann-hw4/data/8number.txt'
with open(filename, 'r') as f:
   lines = f.readlines()
   for l in lines:
      str01 = [int(c) for c in l if c in ['0', '1']]
      chardim.append(np.array(str01))

#------------------------------------------------------------
def showchar01(c, offsetx=0, offsety=0):
   cc = list(zip(*([iter(c)]*10)))
#    [printf(i) for i in cc]

   x = []
   y = []

   X = []
   Y = []

   for id,a in enumerate(cc):
       YY = offsety+12 - id
       for iidd, b in enumerate(a):
           XX = offsetx+iidd
           if b <= 0:
               x.append(XX)
               y.append(YY)
           else:
               X.append(XX)
               Y.append(YY)


   plt.scatter(x, y, s = 1)
   plt.scatter(X, Y, s = 40)


#------------------------------------------------------------
'''
for id, c in enumerate(chardim):
   offsety = 15
   offsetx = id*12
   if id >= 4:
       offsety = 0
       offsetx = (id-4)*12




   showchar01(c, offsetx, offsety)

plt.show()
'''

#------------------------------------------------------------
chardim = np.array(chardim)
cca = chardim*2-np.ones(chardim.shape)
cclen = len(chardim[0])

for id,c in enumerate(cca):
   if id == 0:
       w = np.outer(c,c)
   else:
       w += np.outer(c,c)

for i in range(cclen):
   w[(i, i)] = 0




#------------------------------------------------------------
def dhnn(w, x):
   xx = np.dot(w, x)
   xx01 = [(lambda x: 1 if x > 0 else -1)(a) for a in xx]
   return xx01

def dhnns(w, x):
   for i in range(len(x)):
       xx = np.inner(w[i], x)
       if xx > 0: xx = 1
       else: xx = -1
       x[i] = xx

   return x

#------------------------------------------------------------
'''
for id, c in enumerate(chardim):
   offsety = 15
   offsetx = id*12
   if id >= 4:
       offsety = 0
       offsetx = (id-4)*12


   x = dhnn(w, array(c)*2-1)
   showchar01(x, offsetx, offsety)

plt.show()
'''

#------------------------------------------------------------
def addnoise(c, noise_ratio = 0.1):
   noisenum = int(len(c) * noise_ratio)
   noisepos = [1]*len(c)
   noisepos[:noisenum] = [-1]*noisenum
   random.shuffle(noisepos)

   cc = np.array([x*y for x,y in zip(c, noisepos)])
   return cc


for i in range(40):
   plt.clf()

   for id, c in enumerate(cca):
       offsety = 15
       offsetx = id*12
       if id >= 4:
           offsety = 0
           offsetx = (id-4)*12

       cn = addnoise(c, 0.3)
       x = cn.copy()
       for _ in range(100):
           x = dhnn(w, x)

       showchar01(x, offsetx, offsety)

   plt.draw()
   plt.pause(.1)









