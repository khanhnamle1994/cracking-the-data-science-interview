## Problem
Write a program to generate the partitions for a number n. A partition for n is a list of positive integers that sum up to n. For example: if n = 4, we want to return the following partitions: [1,1,1,1], [1,1,2], [2,2], [1,3], and [4]. Note that a partition [1,3] is the same as [3,1] so only the former is included.

## Solution
We can think about the problem recursively: say that we want to generate a partition for n and we are keeping track of each partition via a list. The list will start off as empty, and be passed into each recursive call. We can start by choosing some element i to be included in the list. Then, we only need to recurse on the remainder, keeping track of the current remainder and how many iterations to check further recursive calls. This is done until the base case when the remainder is zero, in which we return the resulting list. Therefore, a straightforward recursive implementation goes as follows:

```
def partition(n, res = [], start = 1):
    if n == 0:
        yield res
    for i in range(start, n + 1): #number of iterations
        for p in partition(n-i, res + [i], i): #recurse using remainder
            yield p

#print results
for j in partition(4):
    print(j)
```
