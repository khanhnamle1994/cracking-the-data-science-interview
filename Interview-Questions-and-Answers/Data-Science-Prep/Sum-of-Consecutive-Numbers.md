## Problem
This problem was asked by Robinhood.

Given a number n, return the number of lists of consecutive numbers that sum up to n.

For example, for n = 9, you should return 3 since the lists are: [2, 3, 4], [4, 5], and [9]. Can you do it in linear time?

## Solution
The brute force way is to try all sums starting from a particular number. However, this solution has a runtime of O(N^2) and is inefficient. Here is sample code:

```
def consecutive_sum(n):
    num = 0
    for i in range(1, n+1):
        total = 0
        while total < n:
            total = total + i
            i += 1
        if total == n:
            num += 1
    return num
```

Let's reframe the problem in a mathematical way. Say we start at some number k, and go up to some number k+m-1, so that we have a sequence of m terms. Note that:

```
k + (k + 1) + ... + k + m − 1 = (2k + m − 1) * (m/2)
```

Now if we set this to n, we have an equation quadratic in m, as follows:

```
(2k + m - 1) * (m/2) = (1/2) * m^2 + (k - 1/2) * m = n
```

Using the quadratic equation, the corresponding solution is given by:

```
-k + (1/2) +/- sqrt()(k - 1/2)^2 + 2m)
```

If this result is an integer, then we have a valid m and hence a valid list of numbers k to k+m-1 that sum up to n. Therefore we can get a linear O(N) runtime as follows (the quadratic calculation returns a float, and we can check if that float is an integer, i.e. 3.0 is while 3.5 is not):

```
import math

def quadratic(k, n):
    return -k + 1/2 + math.sqrt((k-1/2)**2+2*n)

def consecutive_sum(n):
    num = 0
    for k in range(2, n+1):
        valid = quadratic(k, n)
        if valid.is_integer():
            num += 1
    return num
```
