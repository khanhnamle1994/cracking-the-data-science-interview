## Problem
This problem was asked by Uber.

Assume you are given the below table for spending activity by product type. Write a query to calculate the cumulative spend for each product over time in chronological order.

**total_transactions**

| column_name |   type   |
|:-----------:|:--------:|
|   order_id  |  integer |
|   user_id   |  integer |
|  product_id |  string  |
|    spend    |   float  |
|     date    | datetime |

## Solution
We can use a window function as follows (since we don’t care about the particular order_id or user_id):

```
SELECT date, product_id, SUM(spend)
    OVER (PARTITION BY product_id ORDER BY date) AS cum_spend
FROM total_trans ORDER BY product_id, date ASC
```
