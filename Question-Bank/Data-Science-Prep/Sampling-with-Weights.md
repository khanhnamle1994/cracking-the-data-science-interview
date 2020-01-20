## Problem
This problem was asked by Lyft.

Say we are given a list of several categories (for example, the strings: A, B, C, and D) and want to sample from a list of such categories according to a particular weighting scheme. Such an example would be: for 100 items total, we want to see A 20% of the time, B 15% of the time, C 35% of the time, and D 30% of the time. How do we simulate this? What if we care about an arbitrary number of categories and about memory usage?

## Solution
For the example with a fixed number of categories, we can just generate a list of size 100 and create a list with the proportional counts as follows:

```
import random
l = ['A'] * 20 + ['B'] * 15 + ['C'] * 35 + ['D'] * 30
return(random.choice(l))
```

However, the solution is not optimal. If we wanted to do this more general (keeping space in mind) we can do the following: 1) calculate the cumulative sum of weights, 2) choose a random number k between 0 and the sum of weights, and 3) assign k the corresponding category where the cumulative sum is above k. Here is an example:

```
import random
c = ['A', 'B', 'C', 'D'] # list of categories
l = [20, 15, 35, 30] # list of weights
cs = [sum(l[:i]) for i in range(1,len(l)+1)] # cumulative sum
k = random.randrange(cs[-1]) # choose random number in range (0, total sum)
i = binary_search(cs, k) # binary search for k
return c[i]
```

where `binary_search` is a binary search subroutine that will find the corresponding index in the cumulative sum array. That exact corresponding index is the category we can return.

If there are k categories and the total sum of weights is n, where n >> k, then this method uses O(k) space and has O(k) runtime (since the binary search part is O(log(k)) < O(k)), whereas the first method uses O(n) space and O(n) time to create the full list.
