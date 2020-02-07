## Problem
Say you are given an unfair coin, with an unknown bias towards heads or tails. How can you generate fair odds using this coin?

## Solution
Let p(H) be the probability of landing on heads, and p(T) be the probability of landing tails for any given flip, where p(H) + p(T) = 1. Note that it is impossible to generate fair odds using only one flip. If we use two flips, however, we have four outcomes: HH, HT, TH, and TT. Of these four outcomes, note that two (HT, TH) have equal probabilities since p(H) * p(T) = p(T) * p(H).

Therefore, it is possible to generate fair odds by flipping the unfair coin twice and assigning heads to the HT outcome on the unfair coin, and tails to the TH outcome on the unfair coin.
