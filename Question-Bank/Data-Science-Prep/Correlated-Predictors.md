## Problem
This problem was asked by Google.

Say you are running a multiple linear regression and believe there are several predictors that are correlated. How will the results of the regression be affected if they are indeed correlated? How would you deal with this problem?

## Solution
There will be two main problems. The first is that the coefficient estimates and signs will vary dramatically, depending on what particular variables you include in the model. In particular, certain coefficients may even have confidence intervals that include 0 (meaning it is difficult to tell whether an increase in that X value is associated with an increase or decrease in Y). The second is that the resulting p-values will be misleading - an important variable might have a high p-value and deemed insignificant even though it is actually important.

You can deal with this problem by either removing or combining the correlated predictors. In removing the predictors, it is best to understand the causes of the correlation (i.e. did you include extraneous predictors or such as both X and 2X). For combining predictors, it is possible to include interaction terms (the product of the two). Lastly, you should also 1) center data, and 2) try to obtain a larger sample size (which will lead to narrower confidence intervals).
