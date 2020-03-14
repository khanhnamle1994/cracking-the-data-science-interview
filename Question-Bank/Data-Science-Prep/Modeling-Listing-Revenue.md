## Problem
This problem was asked by Airbnb.

Say you are modeling the yearly revenue of new listings. What kinds of features would you use? What data processing steps need to be taken, and what kind of model would run?

## Solution
The relevant features would include the price details (nightly rate, how the pricing compares to comparable listing), availability (nights available), location (country, market, neighborhood), etc. In terms of data processing steps, any categorical variables should be encoded to numerical ones using one-hot encoding or ordinal encoding depending on the number of categories. Additionally, missing data should be imputed if the data are missing at random, and the causes of why the data is missing should be examined.

The model does not need to be very interpretable since only the prediction accuracy matters (and feature engineering can result in hundreds of features). Good candidates for models would be tree-based models, like random forests and gradient-boosted trees rather than less flexible models such as linear regression, ridge regression, etc.
