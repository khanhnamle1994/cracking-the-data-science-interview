## Problem
This problem was asked by Robinhood.

Assume you are given the below tables for trades and users. Write a query to list the top 3 cities which had the highest number of completed orders.

**Trades**

| column_name |   type   |
|:-----------:|:--------:|
|   order_id  |  integer |
|   user_id   |  integer |
|    symbol   |  string  |
|    price    |   float  |
|   quantity  |  integer |
|     side    |  string  |
|    status   |  string  |
|  timestamp  | datetime |

**Users**

| column_name |   type   |
|:-----------:|:--------:|
|   used_id   |  integer |
|     city    |  string  |
|    email    |  string  |
| signup_date | datetime |

## Solution
We can write an inner query to join the trades and users table, filtering for complete orders, and then do a simple group-by city on the result, as follows:

```
SELECT city, COUNT(DISTINCT order_id) AS num_orders
    FROM (
        SELECT t.order_id, t.user_id, u.city
            FROM trades t LEFT JOIN users u ON t.user_id = u.user_id
            WHERE t.status = ‘complete’
        )
WHERE city is NOT NULL
GROUP BY 1 ORDER BY 2 DESC LIMIT 3
```
