## Problem
This problem was asked by Robinhood.

Write a program to calculate correlation (without any libraries except for math) for two lists X and Y.

## Solution
We know that correlation is given by:

`p_{x, y} = \frac{Cov(X, Y)}{σ_X * σ_Y}`

where the numerator is the covariance of X and Y, and the denominator is the product of the standard deviation of X and the standard deviation of Y. Recall the definition of covariance:

`Cov(X,Y) = E[(X − μ_X)(Y − μ_Y)]`

Therefore, a straightforward implementation is to have helper functions for calculating both the mean and standard deviation:

```
import math

def mean(x):
    return sum(x)/len(x)

def sd(x):
    m = mean(x)
    ss = sum((i-m)**2 for i in x)
    return math.sqrt(ss / len(x))

def corr(x, y):
    x_m = mean(x)
    y_m = mean(y)
    xy_d = 0
    for i in range(len(x)):
        x_d = x[i] - x_m
        y_d = x[i] - y_m
        xy_d += x_d * y_d #add product of X_i and Y_i
    return xy_d / (sd(x) * sd(y)) #from formula above
```
