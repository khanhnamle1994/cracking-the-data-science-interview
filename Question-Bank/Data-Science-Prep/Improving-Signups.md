## Problem
This problem was asked by Quora.

Assume you want to test whether a new feature increases signups to the site. How would you run this experiment? What statistical test(s) would you use?

## Solution
Here we can use a standard A/B test with the experimental group having the feature and the control group not having them. Say group A has the new feature, and group B does not. The null hypothesis here is that the total number of signups in A is less than or equal to the total number of signups in B. Assuming that the samples are large enough and representative, we know that the total number of signups (each one a Bernoulli random variable) will form a binomial distribution. By the Central Limit Theorem, this binomial distribution will be approximately normally distributed.

Therefore, we can use a one-tailed t-test (either Welch’s or Student’s) to compare whether the two groups have an equal amount of total signups. Statistical significance can be assessed by evaluating the sample test statistic. We can compare this test statistic with the t-distribution to test the null hypothesis. If the resulting p-value is low enough, we can reject the null hypothesis and conclude that the feature increased the number of total signups to the site, and otherwise not make that conclusion.
