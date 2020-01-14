## Problem
This problem was asked by Affirm.

Assume we have some credit model, which has accurately calibrated (up to some error) score of how credit-worthy any individual person is. For example, if the model’s estimate is 92% then we can assume the actual score is between 91 and 93. If we take 92 as a score cutoff and deem everyone above that score as credit-worthy, are we over-estimating or underestimating the actual population’s credit score?

## Solution
We are over-estimating the actual population's credit score. To see this, consider the scores on the boundary of the cutoff, i.e. ones such as 91.5 and 92.5. Note that under the current setup, the scores of 91.5 will not get a loan even though their actual true score maybe 92+. Similarly, those with a score of 92.5 will be deemed as credit-worthy and get a loan, even though their score might be < 92. This means that the true credit scores are likely to be overestimated.
