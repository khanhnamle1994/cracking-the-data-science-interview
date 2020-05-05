## Problem
This question was asked by: Uber

**`transactions` table**

|   columns  |   type   |
|:----------:|:--------:|
|     id     |    int   |
|   user_id  |    int   |
|    item    |  varchar |
| created_at | datetime |
|   revenue  |   float  |

Given the revenue transaction table above that contains a user_id, created_at timestamp, and transaction revenue, write a query that finds the third purchase of every user.

<!-- ## Solution
This problem set is relatively straight forward. We can first find the order of purchases for every user by looking at the created_at column and ordering by user_id and the created_at column. However, we still need an indicator of which purchase was the third value.

In this case, we need to apply the RANK function to the transactions table. The RANK function is a window function that assigns a rank to each row in the partition of the result set.

`RANK() OVER (PARTITION BY user_id ORDER BY created_at ASC) AS rank_value`

In this example, the **PARTITION BY** clause distributes the rows in the result set into partitions by one or more criteria.

Second, the **ORDER BY** clause sorts the rows in each partition by the column we indicated, in this case, created_at.

Finally, the **RANK()** function is operated on the rows of each partition and re-initialized when crossing each partition boundary. The end result is a column with the rank of each purchase partitioned by user_id.

All we have to do is then wrap the table in a subquery and filter out where the new column is then equal to 3, which is equivalent for subsetting for the third purchase.

```
SELECT *
FROM (
    SELECT
        user_id
        , created_at
        , revenue
        , RANK() OVER (PARTITION BY user_id ORDER BY created_at ASC) AS rank_value
    FROM transactions
) AS t
WHERE rank_val = 3;
``` -->
