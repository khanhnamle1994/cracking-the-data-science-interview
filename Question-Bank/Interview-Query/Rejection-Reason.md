## Question
This question was asked by: Affirm

Suppose we have a binary classification model that classifies whether or not an applicant should be qualified to get a loan. Because we are a financial company we have to provide each rejected applicant with a reason why.

Given we don't have access to the feature weights, how would we give each rejected applicant a reason why they got rejected?

<!-- ## Solution
Let's pretend that we have three people: Alice, Bob, and Candace that have all applied for a loan. Simplifying the financial lending loan model, let's assume the only features are **total number of credit cards**, **dollar amount of current debt** and **credit age**.

Let's say Alice, Bob, and Candace all have the same number of credit cards and credit age but not the same dollar amount of curent debt.
* Alice: 10 credit cards, 5 years of credit age, **$20K** of debt
* Bob: 10 credit cards, 5 years of credit age, **$15K** of debt
* Candace: 10 credit cards, 5 years of credit age, **$10K** of debt

Alice and Bob get rejected for a loan but **Candace gets approved**. We would assume that given this scenario, we can logically point to the fact that Candace's 10K of debt has swung the model to approve her for a loan.

How did we reason this out? If the sample size analyzed was instead thousands of people who had dthe same number of credit cards and credit age with varying levels of debt, we could figure out the model's average loan acceptance rate for each numerical amount of current debt. Then we could plot these on a graph to **model out the y-value, average loan acceptance, versus the x-value, dollar amount of current debt.**

These graphs are called **partial depedence plots**!

The partial dependence plot is calculated only after the model has been fit. The model is fit on the real data. In that real data, loans are given dependent on different feaures. But after the model is fit, we could start by taking all the characteristics of a loan and plotting them against the dependent variable **whilst keeping all of the other features the same** except for the one feature variable we want to test.

We then use the model to predict the loan qualification but we change the debt dollar amount before making a prediction. We first predict the loan qualification for an example person by setting it to 20K. We then predict it again at $19K. Then predict again for $18K. And so on. We trace out how predicted probability of loan qualification (on the vertical axis) as we move from small values of debt dollar amount to large values (on the horizontal axis). This way we are able to see how a model's features affect the score without digging into the classifier feature weights. -->
