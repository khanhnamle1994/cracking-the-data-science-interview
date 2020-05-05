## Problem
You're given a fair coin. You flip the coin until either **Heads Heads Tails** (HHT) or **Heads Tails Tails** (HTT) appears. Is one more likely to appear first? If so, which one and with what probability?

<!-- ## Solution
Okay, given the two scenarios, we can assess that both sequences need H first. Once H appears, the probability of HHT is now equivalent to 1/2.

Why is this the case? Because in this scenario all you need for HHT is one H. The coin does not reset as we are flipping the coin continuously in sequence until we see the string of HHT or HTT happening in a row. Given that the first letter starts with H, this increases the chances of HHT occuring versus HTT.

Look at these scenarios where we flip the coin four times but with H showing up in the beginning each time and survey the entire sample space.

H-H-H-T = HHT
H-T-H-T = NA
H-H-T-H = HHT
H-T-H-H = NA
H-T-T-H = HTT
H-T-T-T = HTT
H-H-T-T = HHT
H-H-H-H = NA

HHT shows up more than HTT given the limited sample space of 4 flips. Increase this to 5 and it will show up even more.

The probability of HTT is 1/4 because TT needs to occur which is `(1/2) * (1/2)`. Thus HHT is twice is likely to appear first. So, if the probability that HTT appears first is X, then the probability that HHT appears first is 2X. Since these are disjoint and together exhaust the whole probability space, X+2X=1. Therefore **X = 1/3 = HTT**.

HHT is more likely to appear first than HT and the probability of HHT appearing first is 2/3. -->
