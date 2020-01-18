## Problem
This problem was asked by Google.

Assume you take have a stick of length 1 and you break it uniformly at random into three parts. What is the probability that the three parts form a triangle?

## Solution
Assume that the stick looks like the following, with cut points at X and Y:

------X---|---Y-----

Let M (shown in | above) denote the midpoint of the stick (at length 0.5). Note that if X and Y are on the same side of the midpoint, (either on its left or its right), then no triangle is possible. This is because the length of one of the pieces is > 1/2 in that case (and we have two sides which have a total length < that longest side, and thus no triangle is possible). The probability that X and Y are on the same side (since the cuts are chosen randomly) is simply 1/2.

Now, assume that X and Y are on different sides of the midpoint. If X is further left in its half than Y is in its half, then no triangle is possible either since then the part between X and Y will have a length of > 0.5 (for example: X is at 0.2, Y is at 0.75). This has a 1/2 chance of occurring, but this was conditional on X and Y being on different sides of the midpoint, which itself had a 1/2 chance of occurring. Therefore, this case occurs with probability 1/4. The two cases represent all cases whereby there is no valid triangle formed. It follows that there will be a valid triangle with probability 1 - 1/2 - 1/4 = 1/4.
