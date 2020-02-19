## Problem
Three ants are sitting at the corners of an equilateral triangle. Each ant randomly picks a direction and starts moving along the edge of the triangle. What is the probability that none of the ants collide? Now, what if it is k ants on all k corners of an equilateral polygon?

## Solution
Note that the ants are guaranteed to collide unless they each move in the exact same direction. This only happens when all ants move clockwise or counter-clockwise (picture the triangle in 2D). Let P(N) denote the probability of no collision, P(C) denote the case where all ants go clockwise, and P(D) denote the case where all ants go counter-clockwise. Since every ant can choose either direction with equal probability, then we have:

`P(N) = P(C) + P(D) = (1/2)^3 + (1/2)^3 = 1/4`

If we extend this to k ants, the logic is still the same, so we get:

`P(N) = P(C) + P(D) = (1/2)^k + (1/2)^k = 1/2^{k - 1}`
