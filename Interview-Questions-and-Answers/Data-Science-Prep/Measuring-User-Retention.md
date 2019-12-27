## Problem
This problem was asked by Facebook.

Assume you have the below tables on user actions. Write a query to get the active user retention by month.

Table: **user_actions**

|  Column Name |                  Data Type              |
|:------------:|:---------------------------------------:|
|    user_id   |                   integer               |
|    event_id  |  string ("sign-in", "like", "comment")  |
|   timestamp  |                  datetime               |


## Solution
In the actions table, we can first define a temporary table called "mau" to get the monthly active users by month. Then we can do a self join by comparing every user_id across last month vs. this month, using the add_months() function to compare this month versus last month, as follows:

```
WITH mau AS
    (SELECT DISTINCT DATE_TRUNC(‘month’, timestamp) AS month, user_id FROM user_actions)

SELECT
    mau.month,
    COUNT(DISTINCT user_id) AS retained_users
FROM
    mau curr_month
JOIN
    mau last_month
ON curr_month.month = add_months(last_month.month, 1)
    AND curr_month.user_id = last_month.user_id
GROUP BY 1 ORDER BY 1 ASC
```
