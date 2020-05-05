## Problem
Amy and Brad take turns in rolling a fair six-sided die. Whoever rolls a "6" first wins the game. Amy starts by rolling first.

What's the probability that Amy wins?

<!-- ## Solution
Let's set some definitions.
* pA = Probability that Amy wins
* pB = Probability that Brad wins.

Note that pA = P[win if go first].

So we can then deduce that Brad's probability of winning then becomes the probability of going first after Amy loses the first roll. We can represent that with this equation of: *pB = P[Amy loses first roll] * P[win if go first]*.

We also know that the probabilities of either Amy or Brad winning should add up to 1. So mathematically we can create two equations: **pB = 5/6 * pA** and **pA + pB = 1**.

This is now a linear algebra question. Two equations and two unknowns.
* pA = pB - 1 -> pB = 5/6 * (pB - 1)
* pB = 5/6pB - 5/6 -> 5/6 = 11/6pB -> pB = 5/6 * 11/6 = 5/11

The answer is then **pA = 1 - 5/11 -> 6/11** -->
