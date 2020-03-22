## Problem
This problem was asked by Stripe.

Say we have two random variables X and Y. What does it mean for X and Y to be independent? What about uncorrelated? Give an example where X and Y are uncorrelated but not independent.

## Solution
Independence is defined by: `P(X,Y) = P(X) P(Y)`

where P(X, Y) is the joint probability distribution of X and Y. Equivalently we can use the definitions: `P(X|Y) = P(X), P(Y|X) = P(Y)`

Uncorrelated means that the covariance between X and Y is 0, where covariance is: `Cov(X,Y) = E[XY] - E[X]E[Y]`

For an example of uncorrelated but not independent, let X take on values -1, 0, or 1 with equal probability and let Y = 1 if X = 0 and 0 otherwise. Then we can verify that X and Y are uncorrelated:

`E[XY] = 1/3 (-1) (0) + 1/3 (0) (1) + 1/3 (1) (0) = 0`

And E[X] = 0 so the covariance between the two is zero. However, it is clear that the two are not independent since it is not the case that: `P(Y|X) = P(Y)`

since, for example, `P(Y = 1|X = 0) = 1`
