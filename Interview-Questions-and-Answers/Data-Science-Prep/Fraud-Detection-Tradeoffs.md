## Problem
This problem was asked by Affirm.

Assume we have a classifier that produces a score between 0 and 1 for the probability of a particular loan application being fraudulent. In this scenario: a) what are false positives, b) what are false negatives, and c) what are the trade-offs between them in terms of dollars and how should the model be weighted accordingly?

## Solution
A false positive is when the model decides the application is a fraud when in reality it is not. In this case, there is some immediate loss of revenue - if the loan was for some amount X, and we assume some interest rate, say 10%, then the immediate loss is 10% of X. A false negative is when the model decides the application is not a fraud when in reality it is - the loss, in this case, is the immediate loan amount, say X.

Combining the two points above, this means that weighting by revenue should yield a false negative to be worth about 10 false positives, i.e. one bad loan is the same as missing out on 10 good loans. Therefore, for the model, there should be weighted 10:1 in terms of cost for false negatives to false positives.
