## Question
Suppose you're given a portfolio of equities and asked to calculate the 'value at risk' (VaR) via the [variance-covariance method](https://www.investopedia.com/ask/answers/041715/what-variancecovariance-matrix-or-parametric-method-value-risk-var.asp).

The VaR is a statistical risk management technique measuring the maximum loss that an investment portfolio is likely to face within a specified time frame with a certain degree of confidence. The VaR is a commonly calculated metric used within a suite of financial metrics and models to help aid in investment decisions.

In order to calculate the VaR of your portfolio, you can follow the steps below:
1. Calculate periodic returns of the stocks in your portfolio
2. Create a covariance matrix based on (1)
3. Calculate the portfolio mean and standard deviation (weighted based on investment levels of each stock in the portfolio)
4. Calculate the inverse of the normal cumulative distribution with a specified probability, standard deviation, and mean
5. Estimate the value at risk for the portfolio by subtracting the initial investment from the calculation in step 4

To help get you started, you can reference this [Google Colab](https://colab.research.google.com/drive/1dPrUZocrhG1dWyZP33jGaXiKg1oFSdpi) notebook with the historical returns for a portfolio of the following equities:

```
['AAPL','FB', 'C', 'DIS']
```

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1uVhaxrX7QJwPr9MM9SAD0PyDKBhYqWAw) to view this solution in an interactive Colab (Jupyter) notebook.

```
from scipy.stats import norm

#Generate Var-Cov matrix
cov_matrix = returns.cov()

#Calculate mean returns for each stock
avg_rets = returns.mean()

#Calculate mean returns for portfolio overall,
#using mean, using dot product formula to
#normalize against investment weights
port_mean = avg_rets.dot(weights)

#Calculate portfolio standard deviation
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))

#Calculate mean of given investment
mean_investment = (1+port_mean) * initial_investment

#Calculate standard deviation of given investmnet
stdev_investment = initial_investment * port_stdev

#Select our confidence interval (I'll choose 95% here)
conf_level1 = 0.05

#Using SciPy ppf method to generate values for the
#inverse cumulative distribution function to a normal distribution
#Plugging in the mean, standard deviation of our portfolio
#as calculated above
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)

#Finally, we can calculate the VaR at our confidence interval
var_1d1 = initial_investment - cutoff1
var_1d1

#output (will vary depending on when you run this since time window
#is pulling in current data)
22347.7792230231
```

Here we are saying with 95% confidence that the loss of our portfolio will not exceed ~$22.3k over a one day period. -->
