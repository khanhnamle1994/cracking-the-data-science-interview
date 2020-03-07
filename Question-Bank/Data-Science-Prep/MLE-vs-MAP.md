## Problem
This problem was asked by Netflix.

What are MLE and MAP? What is the difference between the two?

## Solution
MLE is maximum likelihood estimation and MAP is maximum a posteriori - both are ways of estimating variables in a probability distribution by producing a single estimate of that variable.

Assume that we have a likelihood function `P(X|θ)`. With n iid samples, the MLE is:

```
MLE(θ) = max_{θ} P(X|θ) = max_{θ} \Pi_{i}^{n} P(x_i|θ)
```

Since the product of many numbers between 0 and 1 might be very small, it is more convenient to maximize the log function, which is an equivalent problem since the log function is monotonically increasing. Since the log of a product is equivalent to the sum of logs, then the MLE becomes:

```
MLE(θ) = max_{θ} \Sum_{i=1}^{n} log P(x_i|θ)
```

MAP relies on Bayes rule, using the posterior `P(θ|X)` being proportional to likelihood multiplied by a prior `P(θ)`, i.e. `P(X|θ) * P(θ)`. The MAP for θ is thus:

```
MAP(θ) = max_{θ} P(X∣θ) * P(θ) = max_{θ} \Pi_{i}^{n} P(x_i|θ) * P(θ)
```

Using the same math as before, the MAP becomes:

```
MAP(θ) = max_{θ} \Sum_{i=1}^{n} log P(x_i|θ) + log P(θ)
```

Therefore, the only difference is the inclusion of the prior in MAP - otherwise, the two are identical. MLE can be seen as a special case of MAP where there the uniform is prior.
