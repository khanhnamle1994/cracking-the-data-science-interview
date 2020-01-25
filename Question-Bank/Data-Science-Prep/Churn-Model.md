## Problem
This problem was asked by Robinhood.

What is user churn and how can you build a model to predict whether a user will churn? What features would you include in the model and how do you assess importance?

## Solution
Churn is when a user stops using the platform - in the case of Robinhood this could be defined as when a user’s account value falls below a minimum threshold for some number of days (say 28). We can build a classifier to predict user churn by incorporating features such as time horizon of the user (i.e. long term investor or short term investor), how much they initially deposited, time spent on the app, etc.

To assess feature importance, we can either run a model like logistic regression and look for weights with a large magnitude, or we can run a decision tree or random forest and look at the feature importance there (which is calculated as the decrease in node impurity weighed by the probability of reaching that node).
