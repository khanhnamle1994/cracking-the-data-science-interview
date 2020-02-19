## Problem
Given X and Y are independent variables with normal distributions, what is the mean and variance of the distribution of 2X - Y when the corresponding distributions are X ~ N (3, 2²) and Y ~ N(1, 2²)?

## Solution
Because the linear combination of the two independent normal random variables is a normal random variable, we can solve the first problem of the mean by just substituting the given values into the formula for the existing two means in the problem statement.

For the two variables X and Y, the mean is calculated simply by:

`2X - Y = 2(3) - 1 = 5`

The variance however is calculated differently. The variance of `aX - bY` is:

`𝑉𝑎𝑟(𝑎𝑋 − 𝑏𝑌) = 𝑎2 𝑉𝑎𝑟(𝑋) + 𝑏2 𝑉𝑎𝑟(𝑌) − 2𝑎𝑏 * 𝐶𝑜𝑣(𝑋,𝑌)`

where `𝐶𝑜𝑣(𝑋,𝑌)` is the covariance between X and Y. The covariance between both X and Y is zero given the normal random variables. That way we can calculate this out:

```
𝑉𝑎𝑟(𝑎𝑋−𝑏𝑌) = 𝑎2 𝑉𝑎𝑟(𝑋) + 𝑏2 𝑉𝑎𝑟(𝑌) − 2𝑎𝑏 * 𝐶𝑜𝑣(𝑋,𝑌)
= 4·𝑉𝑎𝑟(𝑋) + 𝑉𝑎𝑟(𝑌) − 0
= 4·4 + 4 = 20
```
