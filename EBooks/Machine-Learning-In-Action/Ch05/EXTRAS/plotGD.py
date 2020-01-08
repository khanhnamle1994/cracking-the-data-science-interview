'''
Created on Oct 28, 2010

@author: Peter
'''
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-2.0, 2.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = -((X-1)**2)
Z2 = -(Y**2)
#Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
#Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 1.0 * (Z2 + Z1)+5.0

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.annotate('', xy=(0.05, 0.05),  xycoords='axes fraction',
             xytext=(0.2,0.2), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-1.9, -1.8, 'P0')
plt.annotate('', xy=(0.2,0.2),  xycoords='axes fraction',
             xytext=(0.35,0.3), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-1.35, -1.23, 'P1')
plt.annotate('', xy=(0.35,0.3),  xycoords='axes fraction',
             xytext=(0.45,0.35), textcoords='axes fraction',
             va="center", ha="center", bbox=leafNode, arrowprops=arrow_args )
plt.text(-0.7, -0.8, 'P2')
plt.text(-0.3, -0.6, 'P3')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Gradient Ascent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
