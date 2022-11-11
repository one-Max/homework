import numpy as np
import math
import matplotlib.pyplot as plt
from time import sleep  
plt.subplots()
xMax = 500
x = np.linspace(0,2*np.pi,xMax)
y = np.sin(x)
plt.plot(x,y,'b--')    #设置轨迹虚线
sleep(0.5)
ax,= plt.plot(x,y,'ro')   
i = 0
while i <= xMax:
    plt.pause(0.002)    #暂停0.002秒,功能同sleep()
    y1 = np.sin(x[:i])   #这里采用逐步增加y轴值范围的方法
    ax.set_xdata(x[:i])   #这里采用逐步增加x轴值范围的方法
    ax.set_ydata(y1)
    plt.draw()     #重画已经修改的图形
    i += 2      #获取范围控制,兼循环控制
plt.show()
