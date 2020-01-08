'''
Created on Jun 1, 2011

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pca

n = 1000 #number of points to create
xcord0 = []; ycord0 = []
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
markers =[]
colors =[]
fw = open('testSet3.txt','w')
for i in range(n):
    groupNum = int(3*random.uniform())
    [r0,r1] = random.standard_normal(2)
    if groupNum == 0:
        x = r0 + 16.0
        y = 1.0*r1 + x
        xcord0.append(x)
        ycord0.append(y)
    elif groupNum == 1:
        x = r0 + 8.0
        y = 1.0*r1 + x
        xcord1.append(x)
        ycord1.append(y)
    elif groupNum == 2:
        x = r0 + 0.0
        y = 1.0*r1 + x
        xcord2.append(x)
        ycord2.append(y)
    fw.write("%f\t%f\t%d\n" % (x, y, groupNum))

fw.close()
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(xcord0,ycord0, marker='^', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50,  c='red')
ax.scatter(xcord2,ycord2, marker='v', s=50,  c='yellow')
ax = fig.add_subplot(212)
myDat = pca.loadDataSet('testSet3.txt')
lowDDat,reconDat = pca.pca(myDat[:,0:2],1)
label0Mat = lowDDat[nonzero(myDat[:,2]==0)[0],:2][0] #get the items with label 0
label1Mat = lowDDat[nonzero(myDat[:,2]==1)[0],:2][0] #get the items with label 1
label2Mat = lowDDat[nonzero(myDat[:,2]==2)[0],:2][0] #get the items with label 2
#ax.scatter(label0Mat[:,0],label0Mat[:,1], marker='^', s=90)
#ax.scatter(label1Mat[:,0],label1Mat[:,1], marker='o', s=50,  c='red')
#ax.scatter(label2Mat[:,0],label2Mat[:,1], marker='v', s=50,  c='yellow')
ax.scatter(label0Mat[:,0],zeros(shape(label0Mat)[0]), marker='^', s=90)
ax.scatter(label1Mat[:,0],zeros(shape(label1Mat)[0]), marker='o', s=50,  c='red')
ax.scatter(label2Mat[:,0],zeros(shape(label2Mat)[0]), marker='v', s=50,  c='yellow')
plt.show()