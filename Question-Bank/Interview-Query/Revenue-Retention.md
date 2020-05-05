## Problem
Let's say that you work for a software as a subscription (SAAS) company that has existed for just over a year. The chief revenue officer wants to know the average lifetime value.

We know that the product costs 100 dollars per month, averages 10% in monthly churn, and the average customer sticks around for around 3.5 months.

Calculate the formula for the average lifetime value.

<!-- ## Solution
This is a trick question given that the candidate is given multiple pieces of supposedly relevant information. Let's break it down by looking at just the essential pieces and zeroing in on what pieces of data we need for a calculation.

The chief revenue officer wants to know the average lifetime value. Otherwise known as LTV, average lifetime value is defined by the **prediction of the net revenue attributed to the entire future relationship with all customers averaged**. Given that we don't know the future net revenue, we can estimate it by taking the total amount of revenue generated divided by the total number of customers acquired over the same period time.

So given that we know that the average customer length on the platform is 3.5 months, couldn't we just calculate the LTV as $100 x 3.5 months = $350? Not exactly.

Notice how the question states that the company has existed for just over a year. This is a pretty short time for a business to be around when the lifetime subscription of a customer could be much longer. Our **average customer length** is then biased based on the fact that the business hasn't been around long enough to correctly measure a sample average that is indicative of the true mean. For example, if the business had existed for three months then our average customer length might have been 1.5 months.

Given it is a subscription business, we can then use the information provided by product price and churn to measure an actual lifetime value. In this case, we are already given the average product value and churn. The product costs 100 dollars a month and the product averages 10% in monthly churn. Therefore we can calculate the **expected value of the customer at each month as a multiplier of retention times the product cost**.

Let's look at this example.

First month the expected value is 100 dollars with 100% of customers retained. The customer has to pay the entire value the first month and it's assumed they will pay upfront. In the second month, since the company is dealing with an average 10% churn, the company will retain only 90% of the customers.

And so the expected value drops down on the second month to:

`Second Month EV = $100 * 90% = $90`

If N is the total number of months the company has been in business and will be in business, our calculation then becomes:

`LTV = 100*.9^(0) + 100*.9^(1) + 100*.9^(2) + ..... + 100*.9^(N)` -->
