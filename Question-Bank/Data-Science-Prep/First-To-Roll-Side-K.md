## Problem

This problem was asked by Lyft.

A and B are playing the following game: a number k from 1-6 is chosen, and A and B will toss a die until the first person sees the side k, and that person gets $100. How much is A willing to pay to play first in this game?

## Solution

To assess the amount A is willing to pay, we need to calculate the expected probabilities of winning for each player, assuming A goes first. Let the probability of A winning (if A goes first) be given by P(A) and the probability of B winning (if A goes first but doesn’t win on the first roll) be P(B*).

Then we can use the following recursive formulation:

`P(A) = 1/6 + 5/6 * (1 - P(B*))`

since A wins immediately with a 1/6 chance (the first roll is k), or with a 5/6 chance (assuming the first roll is not a k), they win if B does not win, where B now goes first.

However, notice that if A doesn’t roll side k immediately, then P(B*) = P(A), since now the game is exactly symmetric with player B going first.

Therefore, we can write out the above as:

`P(A) = 1/6 + 5/6 * P(A)`

Solving the above yields P(A) = 6/11, and P(B) = 1 - P(A) = 5/11. Since the payout is $100, then A should be willing to pay up to the difference in expected values in going first, which is 100 * (6/11 - 5/11) = 100/11, or about $9.09.
