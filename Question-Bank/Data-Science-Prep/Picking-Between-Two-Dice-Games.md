## Problem
There are two games involving dice that you can play. In the first game, you roll two die at once and get the dollar amount equivalent to the product of the rolls. In the second game, you roll one die and get the dollar amount equivalent to the square of that value. Which has the higher expected value and why?

## Solution
There is the brute force method of computing the expected values by listing all of the outcomes and associated probabilities and payoffs, however, there is a cleaner way of solving the problem.

Let us assume that the outcome of the roll of a dice is given by a random variable X (so it takes on values 1...6 with equal probability). Then the question is equivalent to asking, what is E[X] * E[X] = E[X]^2 (expectation of the product of two separate rolls), versus E[X^2] (expectation of the square of a single roll)?

If you recall, for a given random variable X, the variance is given by:

```
Var(X) = E[X - E[X]^2] = E[X^2] - 2E[X]E[X] + E[X]^2 = E[X^2] - E[X]^2
```

This variance term turns out to exactly be the difference between the two games (payoff the second game minus the payoff the first game). It will always be positive unless the two games are the exact same (having the same payoffs and probabilities), which in this case they are not. Therefore it must be the case that the above variance term is positive, implying that the second game has a higher expected value.
