## Question
Suppose you're working in a parts manufacturing plant, and you're running quality analysis on the gasket production line. Gaskets produced by your company will be defective 1% of the time, and are distributed in packs of 20.

Your company has a policy where if 2 or more of the 20 gaskets in a given pack is defective, they will replace the entire package. What proportion of packages does the company need to replace?

*Hint: If you're struggling with where to start here, consider that this is a [binomial](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHL4Q9sNf2baV1O4E-2F9EKFxigsjKvnu9TUUP6oeTC0bjjnAq09MOdQqKm-2Bz-2FtAS67LCg-3D-3D_8c6kLYfeKFgEvI6pydPvKIhJlwkKyLn-2B2sUm5OgZWKpQ2qIVcRcaoX8ayewthb05OxL5Ps-2FiNBpHDvbFZUPpraWfj2C-2BRAdIq8yXPoLawUYEFT-2F53wVpKh4cYTMpREZ1HR4ALrb49OPP3d-2BNLmp6qQ5wN83ZzS8BfjVx8wNQbhWiqLJm1A-2FqzyzFh2uwUBggBbuvb1yFUuvAwU8UZMDNftnmfhNAtBkCA3Vu6OB2D6Q-3D) probability problem.*

<!-- ## Solution:
We first need to recognize that this is a binomial probability problem. If you're not familiar with the binomial distribution I would check out this [wikpedia reference](https://en.wikipedia.org/wiki/Binomial_distribution). When you're going to interview, you probably won't be asked a math-heavy statistics problem like this, however you should be familiar to talk through the concept and at a minimum be able to derive the 1st line in LaTeX below.

First, we're going to pull out a few variables from our problem statement and plug them into the binomial distribution formula.

* n = 20 → 20 gaskets in 1 package
* p = 0.01 → 1% defect rate

Given that X is going to be the number of defective gaskets in a package, then X is a binomial random variable with the parameters (n=20, p=0.01).

So the probability of us needing to replace a given package is:

```
P(replacing package) = 1 - P(X = 0) - P(X = 1)
P(replacing package) = 1 - 0.8179 - 0.1652
P(replacing package) = 0.0168
```

So, about 1.68% of the time we will need to replace the package. Looks like the next thing we need to look at is how much this is costing the company! -->
