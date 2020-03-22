## Problem
This problem was asked by Facebook.

You and your friend are playing a game. The two of you will continue to toss a coin until the sequence HH or TH shows up. If HH shows up first, you win. If TH shows up first, your friend wins. What is the probability of you winning?

## Solution
Although there is a formal way to apply Markov chains to this problem, there is a simple trick that simplifies the problem. Note that if T is ever flipped, you can't reach HH before your friend reaches TH since the first heads thereafter will result in them winning. Therefore, the probability of you winning is limited to just flipping an HH initially, which we know is given by:

`P(HH) = 1/2 * 1/2 = 1/4`

Therefore you have a 1/4 chance of winning whereas your friend has a 3/4 chance.
