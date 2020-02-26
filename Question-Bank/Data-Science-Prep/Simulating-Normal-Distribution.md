## Problem

Say you are given a random Bernoulli trial generator. How would you generate values from a standard normal distribution?

## Solution

Assume we have n Bernoulli trials each with a success probability of p:

`x_1, x_2, ..., x_n, x_i ∼ Ber(p)`

Assuming iid trials, we can compute the sample mean for p from a large number of trials:

`\hat_\mu = \frac{1}{n} \sum_{i=1}^n x_i`

We know the expectation of this sample mean is:

`E[\hat_\mu] = \frac{np}{n} = p`

Additionally, we can compute the variance of this sample mean:

`Var(\hat_\mu) = \frac{np(1-p)}{n^2} = \frac{p(1-p)}{n}`

Assume we sample a large n. Due to the Central Limit Theorem, our sample mean will be normally distributed:

`\hat_\mu ∼ N(p, \frac{p(1-p)}{n})`

Therefore we can take a z-score of our sampled mean as:

`z(\hat_\mu) = \frac{\hat_\mu - p}{ \sqrt{ \frac{p(1-p)}{n} } }`

This z-score will then be a simulated value from a standard normal distribution.
