## Problem
Let's say that you work at a bank that wants to build a model to detect fraud on the platform. The bank wants to implement a text messaging service in addition that will text customers when the model detects a fraudulent transaction in order for the customer to approve or deny the transaction with a text response.
1. What kind of model would need to be built?
2. Given the scenario, if you were building the model, which model metrics would you be optimizing for?

<!-- ## Solution
1. **Binary classifier**. Given that fraud is binary, there either is a fraudulent transaction or there isn't.
2. There are a lot of different ways to analyze model performance but let's take into account what's specified. We know that in binary classification problems there are precision versus recall trade-offs.

Precision is defined as the number of true positives divided by model predicted positives. In our example this would be the percentage of correct fraudulent transactions out of predicted fraudulent transactions.

`Precision = (True Positive / (True Positive + False Positive))`

Recall is defined as the number of true positives divided by number of actual true positives. In our example this would be the number of correct fraudulent transactions out of actual fraudulent transactions.

`Recall = (True Positive / (True Positive + False Negative))`

Given these two metrics for evaluating a binary classifier, which metric would a bank prefer to be higher? Low recall in a fraudulent case scenario would be a disaster. With low predictive power on false negatives, fraudulent purchases would go under the rug with consumers not even knowing they were being defrauded.

Meanwhile if there was low precision, customers would think their accounts would be under fraud all the time. But since the question prompts for a text messaging service, this would be okay since the end customer would just have to approve or deny transactions that were false fraud transactions. -->
