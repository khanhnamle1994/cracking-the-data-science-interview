'''
Created on Oct 6, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

n = 1000 #number of points to create
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]
fw = open('testSet.txt','w')
for i in range(n):
    [r0,r1] = random.standard_normal(2)
    myClass = random.uniform(0,1)
    if (myClass <= 0.5):
        fFlyer = r0 + 9.0
        tats = 1.0*r1 + fFlyer - 9.0
        xcord0.append(fFlyer)
        ycord0.append(tats)
    else:
        fFlyer = r0 + 2.0
        tats = r1+fFlyer - 2.0
        xcord1.append(fFlyer)
        ycord1.append(tats)
    #fw.write("%f\t%f\t%d\n" % (fFlyer, tats, classLabel))

fw.close()
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(xcord,ycord, c=colors, s=markers)
ax.scatter(xcord0,ycord0, marker='^', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
plt.plot([0,1], label='going up')
plt.show()