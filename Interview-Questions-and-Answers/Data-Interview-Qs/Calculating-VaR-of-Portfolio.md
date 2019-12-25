Suppose you're given a portfolio of equities and asked to calculate the 'value at risk' (VaR) via the [variance-covariance method](https://www.investopedia.com/ask/answers/041715/what-variancecovariance-matrix-or-parametric-method-value-risk-var.asp).

The VaR is a statistical risk management technique measuring the maximum loss that an investment portfolio is likely to face within a specified time frame with a certain degree of confidence. The VaR is a commonly calculated metric used within a suite of financial metrics and models to help aid in investment decisions.

In order to calculate the VaR of your portfolio, you can follow the steps below:
1. Calculate periodic returns of the stocks in your portfolio
2. Create a covariance matrix based on (1)
3. Calculate the portfolio mean and standard deviation (weighted based on investment levels of each stock in the portfolio)
4. Calculate the inverse of the normal cumulative distribution with a specified probability, standard deviation, and mean
5. Estimate the value at risk for the portfolio by subtracting the initial investment from the calculation in step 4

To help get you started, you can reference this [Google Colab](https://colab.research.google.com/drive/1dPrUZocrhG1dWyZP33jGaXiKg1oFSdpi) notebook with the historical returns for a portfolio of the following equities:

```['AAPL','FB', 'C', 'DIS']```
