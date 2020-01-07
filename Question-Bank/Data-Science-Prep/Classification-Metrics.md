## Problem
This problem was asked by Uber.

Say you need to produce a binary classifier for fraud detection. What metrics would you look at, how is each defined, and what is the interpretation of each one?

## Solution
Some main important ones are precision, recall, and the AUC of the ROC curve. Let us define TP as a true positive, FP as a false positive, TN as a true negative, and FN as a false negative.

Precision is defined by TP / (TP + FP). It answers the question “what percent of fraudulent predictions were correct?” and is important to maximize since you want your classifier to be as correct as possible when it identifies a transaction as fraudulent.

Recall is defined by TP / (TP + FN). It answers the question “what percent of fraudulent cases were caught?” and is important to maximize since you want your classifier to have caught as many of the fraudulent cases as possible.

AUC of the ROC curve is defined by the area under an ROC curve, which is constructed by plotting the true positive rate, TP / (TP + FN) versus the false positive rate, FP / (FP + TN) for various thresholds, which determines the label of fraudulent or not fraudulent. The area under this curve, or AUC, is a measure of separability of the model, and the closer to 1 it is, the higher the measure. It answers the question “is my classifier able to discriminate between fraud and not-fraud effectively” and is important to maximize since you want your classifier to separate the two classes accordingly.
