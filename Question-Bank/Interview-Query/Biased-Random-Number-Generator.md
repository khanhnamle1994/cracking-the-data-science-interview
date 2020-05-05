## Problem
This question was asked by: Amazon

Given an unfair coin with the probability of heads and tails not equal to 50/50, what algorithm could generate a list of random ones and zeros?

<!-- ## Solution
This problem can be solved with a method called the von neumann corrector. Observe that even if the probability is not 50/50, we can get an equal distribution of two values by taking the combination of outputs.

The algorithm works on pairs of bits, and produces output as follows:
1. When you get heads-tails you count the toss as heads or 1.
2. When you get tails-heads you count it as tails or zero.
3. You ignore the throws that come up twice the same side whether it's TT or HH.

Regardless of the distribution of heads and tails, the output will always have a 50/50 split of 0s and 1s. The algorithm will discard (on average) 75% of all inputs however, even if the original input was perfectly random to start with. -->
