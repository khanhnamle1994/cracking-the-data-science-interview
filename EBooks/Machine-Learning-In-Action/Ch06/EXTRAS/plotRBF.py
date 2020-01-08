'''
Created on Nov 26, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

xcord0 = []; ycord0 = []; xcord1 = []; ycord1 = []
fw = open('testSetRBF2.txt', 'w')#generate data

fig = plt.figure()
ax = fig.add_subplot(111)
xcord0 = []; ycord0 = []; xcord1 = []; ycord1 = []
for i in range(100):
    [x,y] = random.uniform(0,1,2)
    xpt=x*cos(2.0*pi*y); ypt = x*sin(2.0*pi*y)
    if (x > 0.5):
        xcord0.append(xpt); ycord0.append(ypt)
        label = -1.0
    else:
        xcord1.append(xpt); ycord1.append(ypt)
        label = 1.0
    fw.write('%f\t%f\t%f\n' % (xpt, ypt, label))
ax.scatter(xcord0,ycord0, marker='s', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
plt.title('Non-linearly Separable Data for Kernel Method')
plt.show()
fw.close()