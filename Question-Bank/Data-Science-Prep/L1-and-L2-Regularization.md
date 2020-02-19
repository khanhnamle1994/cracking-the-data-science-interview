## Problem
This problem was asked by Uber.

What is L1 and L2 regularization? What are the differences between the two?

## Solution
L1 and L2 regularization are both methods of regularization that attempt to prevent overfitting in machine learning. For a regular regression model assume the loss function is given by L. L1 adds the absolute value of the coefficients as a penalty term, whereas L2 adds the squared magnitude of the coefficients as a penalty term.

The loss function for the two are:

```
Loss(L_1) = L + \lambda ∣w_i∣
Loss(L_2) = L + \lambda |w_i^2|
```

Where the loss function L is the sum of errors squared, given by the following, where f(x) is the model of interest, for example, linear regression with p predictors:

```
L = \sum_{i=1}^{n} (y_i - f(x_i))^2 = \sum_{i=1}^{n} (y_i - \sum_{j=1}^p (x_ij w_j))^2
```

If we run gradient descent on the weights w, we find that L1 regularization will force any weight closer to 0, irrespective of its magnitude, whereas, for the L2 regularization, the rate at which the weight goes towards 0 becomes slower as the rate goes towards 0. Because of this, L1 is more likely to “zero” out particular weights, and hence removing certain features from the model completely, leading to more sparse models.
