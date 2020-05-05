## Problem
This question was asked by: Dropbox

**`bank_transactions` table**

|       column      |   type   |
|:-----------------:|:--------:|
|      user_id      |    int   |
|     created_at    | datetime |
| transaction_value |   float  |

We're given a table bank transactions with three columns, user_id, a deposit or withdrawal value, and created_at time for each transaction.

Write a query to get the total three day rolling average for deposits by day.

**Example**:

**Input**

| user_id | created_at | transaction_value |
|:-------:|:----------:|:-----------------:|
|    1    | 2019-01-01 |         10        |
|    2    | 2019-01-01 |         20        |
|    1    | 2019-01-02 |        -10        |
|    1    | 2019-01-02 |         50        |
|    2    | 2019-01-03 |         5         |
|    3    | 2019-01-03 |         5         |
|    2    | 2019-01-04 |         10        |
|    1    | 2019-01-04 |         10        |

**Output**

|     dt     | rolling_three_day |
|:----------:|:-----------------:|
| 2019-01-01 |       30.00       |
| 2019-01-02 |       40.00       |
| 2019-01-03 |       30.00       |
| 2019-01-04 |       23.33       |

<!-- ## Solution
Usually if the problem states to solve for a moving/rolling average, we're given the dataset in the form of a table with two columns, the date and the value. This problem however is taken one step further with a table of just transactions with values conditioned to filtering for only deposits. Which means we have to aggregate to a daily aggregation table first.

```
WITH valid_transactions AS (
    SELECT DATE_TRUNC('day', created_at) AS dt
        , SUM(transaction_value) AS total_deposits
    FROM bank_transactions AS bt
    WHERE transaction_value > 0
    GROUP BY 1
)
```

Notice that we're filtering for deposits by setting the `transaction_value` column to greater than 0. Then we group by the `created_at` field and truncate the value to each day.

|     dt     | total_deposits |
|:----------:|:--------------:|
| 2019-01-01 |       30       |
| 2019-01-02 |       50       |
| 2019-01-03 |       10       |
| 2019-01-04 |       20       |

Cool, now how do we get rolling three days?

Let's think about how we would compute it manually. If I would look at the table, I would take the last three dates and compute an average of the values for the relevant date rolling average.

SQL however does not work in this cursor format that a regular Python expression could compute by running something like (values[i:i-3])/3 where the variable `values` is a list of daily transactions.

Instead we have to do a **self-join**. If we join the existing `deposits` CTE back onto itself on dates that are **between three days ago from the referenced data and the current referenced date**, we will for each datetime, have three rows of the last three days. All that is left to do is sum the values.

```
SELECT vt1.dt
    AVERAGE(vt2.total_deposits) AS rolling_three_day
FROM valid_transactions AS vt1
INNER JOIN valid_transactions AS vt2
    -- set conditions for greater than three days
    ON vt1.dt > DATE_ADD('DAY', -3, vt2.created_at)
    -- set conditions for max date threshold
        AND vt1.dt <= vt2.created_at
GROUP BY 1
``` -->
