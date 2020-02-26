## Problem

Assume we have a classifier that produces a score between 0 and 1 for the probability of a particular loan application behind a fraud. Say that for each application’s score, we take the square root of that score. How would the ROC curve change? If it doesn’t change, what kinds of functions would change the curve?

## Solution

Recall that the ROC curve plots the true positive rate versus the false positive rate. If all scores change simultaneously, then none of the actual classifications change (since thresholds are adjusted), leading to the same true positive and false positive rates since only the relative ordering of the scores matter. Therefore, taking a square root would not change anything about the ROC curve because the relative ordering is maintained: if application one had a score of X and application two had a score of Y, and Y > X, then it is still the case that sqrt(Y) > sqrt(X). So only the thresholds on the models would change.

In contrast, any function that is not monotonically increasing would change the ROC curve since the relative ordering is not maintained. Some simple examples are: f(x) = -x, f(x) = -x^2, or a stepwise function.
