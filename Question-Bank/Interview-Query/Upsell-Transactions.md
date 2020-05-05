## Question
This question was asked by: Coinbase

Given a transaction table of product purchases, write a query to get the number of customers that were upsold by purchasing additional products.

Note that if the customer purchased two things on the same day that does not count as an upsell as they were purchased within a similar timeframe. Each row in the transactions table also represents an individual user product purchase.

**transactions**

|   column   |   type   |
|:----------:|:--------:|
|   user_id  |    int   |
| created_at | datetime |
| product_id |    int   |
|  quantity  |    int   |
|    price   |   float  |

<!-- ## Solution
This question is a little tricky because we have to note the dates that each user purchased products. We can't just group by `user_id` and look where the number of products purchased is greater than one because of the upsell condition. We have to group by both the date field and the `user_id` to get individual transaction days for each user.

```
SELECT
    user_id
    , DATE(created_at) AS date
FROM transactions
GROUP BY 1,2
```

The query above will now give us a `user_id` and DATE field for each row. If there exists duplicate user_ids then we know that the user purchased on multiple days, which satisfies the upsell condition.

Given this data, now we just have to filter the users that purchased on only one date, and get the count.

```
SELECT COUNT(*)
FROM (
    SELECT user_id
    FROM (
        SELECT
            user_id
            , DATE(created_at) AS date
        FROM transactions
        GROUP BY 1,2
    ) AS t
    GROUP BY 1
    -- Filter out users that only bought once
    HAVING COUNT(*) > 1
) AS s
``` -->
