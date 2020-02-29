## Problem
This problem was asked by Microsoft.

What is the bias-variance tradeoff? How is it expressed using an equation?

## Solution
The equation this tradeoff is expressed in is given by the following: Total model error = Bias + Variance + Irreducible error. Flexible models have low bias and high variance, whereas more rigid models have high bias and low variance.

The bias term comes from the error when a model is under-fitting data. Having a high bias means that the machine learning model is too simple and does not capture the relationship between the features and the target. An example would be using linear regression when the underlying relationship is nonlinear.

The variance term comes from the error when a model is overfitting data. Having a high variance means that a model is susceptible to changes in training data, which means it is capturing too much noise. An example would be using a very complex neural network where the true underlying relationship between the features and the target is linear.

The irreducible term comes from the error that cannot be addressed directly by the model, such as from the noise in measurements of data.
