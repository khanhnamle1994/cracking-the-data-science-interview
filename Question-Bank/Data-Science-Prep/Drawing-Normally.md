## Problem
This problem was asked by Quora.

You are drawing from a normally distributed random variable X ~ N(0, 1) once a day. What is the approximate expected number of days until you get a value of more than 2?

## Solution
Since X is normally distributed, we can look at the cumulative distribution function (CDF) of the normal distribution:

```
\gamma(x) = P(X ≤ x)
```

To check the probability X is at least 2, we can check (knowing that X is distributed as standard normal):

```
\gamma(2) = P(X ≤ 2) = P(X ≤ μ + 2σ) = 0.97
```

Therefore P(X > 2) = 1 - 0.977 = 0.023 for any given day. Since the draws are independent each day, then the expected time until drawing an X > 2 follows a geometric distribution, with p = 0.023. Let T be a random variable denoting the number of days, then we have:

```
E[T] = 1/p = 1/.024 ≈ 43 days
```
