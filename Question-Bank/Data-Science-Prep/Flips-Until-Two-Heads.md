## Problem
This problem was asked by Lyft.

What is the expected number of coin flips needed to get two consecutive heads?

## Solution
Let X be the number of coin flips needed until two heads. Then we want to solve for E[X]. Let H denote a flip that resulted in heads, and T denote a flip that resulted in tails. Note that E[X] can be written in terms of E[X|H] and E[X|T], i.e. the expected number of flips needed, conditioned on a flip being either heads or tails respectively.

Conditioning on the first flip, we have:

```
E[X] = 1/2 * (1 + E[X|H]) + 1/2 * (1 + E[X|T])
```

Note that E[X|T] = E[X] since if a tail is flipped, we need to start over in getting two heads in a row.

To solve for E[X|H], we can condition it further on the next outcome: either heads (HH) or tails (HT).

Therefore, we have:

```
E[X|H] = 1/2 * (1 + E[X|HH]) + 1/2 * (1 + E[X|HT])
```

Note that if the result is HH, then E[X|HH] = 0 since the outcome was achieved, and that E[X|HT] = E[X] since a tail was flipped, we need to start over again, so:

```
E[X|H] = 1/2 * (1 + 0) + 1/2 * (1 + E[X]) = 1 + 1/2 * E[X]
```

Plugging this into the original equation yields E[X] = 6 coin flips.
