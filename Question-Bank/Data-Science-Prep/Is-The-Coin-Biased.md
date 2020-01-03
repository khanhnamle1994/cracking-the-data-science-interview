## Problem
This problem was asked by Google.

A coin was flipped 1000 times, and 550 times it showed up heads. Do you think the coin is biased? Why or why not?

## Solution
Because the sample size of flips is large (1000), we can apply the Central Limit Theorem. Since each individual flip is a Bernoulli random variable, we can assume it has a probability of showing up heads as p. Then we want to test whether p is 0.5 (i.e. whether it is fair). The Central Limit Theorem allows us to approximate the total number of heads seen as being normally distributed.

More specifically, the number of heads seen should follow a Binomial distribution since it a sum of Bernoulli random variables. If the coin is not biased (p = 0.5), then we have the following on the expected number of heads, and variance on that outcome:

```
\sigma = np = 1000 * 0.5 = 500
```

```
\sigma^2 = np(1-p) = 1000 * 0.5 * 0.5 = 250

\sigma = \sqrt{250} \approx 16
```

Since this mean and standard deviation specify the normal distribution, we can calculate the corresponding z-score for 550 heads:

```
z = \frac{550 - 500}{16} > 3
```

This means that, if the coin were fair, the event of seeing 550 heads should occur with a < 1% chance under normality assumptions. Therefore, the coin is likely biased.
