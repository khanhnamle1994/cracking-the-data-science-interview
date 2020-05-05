## Problem
This question was asked by: Microsoft

**`user_dimension` table**

|   columns  | type |
|:----------:|:----:|
|   user_id  |  int |
| account_id |  int |

**`account_dimension` table**

|     columns     |   type  |
|:---------------:|:-------:|
|    account_id   |   int   |
| paying_customer | boolean |

**`download_facts` table**

|  columns  | type |
|:---------:|:----:|
|    date   | date |
|  user_id  |  int |
| downloads |  int |

Given three tables, user_dimension, account_dimension, and download_facts, find the average number of downloads for free vs paying customers broken out by day.

<!-- ## Solution

It's pretty clear here we have to join all three tables together. The `user_dimension` table represents the mapping between account ids and user ids while the `account_dimension` table holds the value if the customer is paying or not. Lastly the `download_facts` table has the date and number of downloads per user.

Since we want the *average number of downloads for free vs paying customers broken out by day*, we can assume that the query will need two GROUP BYs to break out the aggregation of free and paying customers and the other by date.

Lastly to get the average number of downloads, we'll have to use the SUM function divided by the total count and run on distinct user_ids just in case there exists duplicates.

```
SELECT
    date
    , paying_customer
    , SUM(downloads)/COUNT(DISTINCT user_id) AS average_downloads
FROM user_dimension AS ud
INNER JOIN account_dimension AS ad
    ON ud.account_id = ad.account_id
LEFT JOIN download_facts AS df
    ON ud.user_id = df.user_id
GROUP BY 1,2
```

One lingering question is, why doesn't the AVERAGE function work in this case? The user to account mapping provides an answer. Given that it's likely that an account has many users since the `user_dimension` is a mapping table, we can't run the AVERAGE function as it takes an **average grouped on the user level when we want it on the account level**. -->
