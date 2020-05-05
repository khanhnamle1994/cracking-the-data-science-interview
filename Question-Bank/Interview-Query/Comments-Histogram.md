## Question
This question was asked by: Facebook

**`users` table**

|  columns  |   type   |
|:---------:|:--------:|
|     id    |    int   |
|    name   |  varchar |
| joined_at | datetime |
|  city_id  |    int   |
|   device  |    int   |

**`user_comments` table**

|   columns  |   type   |
|:----------:|:--------:|
|   user_id  |    int   |
|    body    |   text   |
| created_at | datetime |

Write a SQL query to create a histogram of number of comments per user in the month of January 2019. Assume bin buckets class intervals of one.

<!-- ## Solution
Let's break down the solution. Here are the things we have to note.

A histogram with bin buckets of one means that we can avoid the logical overhead of grouping frequencies into specific intervals when we run a simple GROUP BY in SQL. Since a histogram is a bar chart of frequencies for each user, the only thing we need to do is select the total count of user comments in the month of January 2019 and then group by that count again.

At first look, it seems like we don't need to do a join between the two tables. The `user_id` column exists in both tables. But if we ran a GROUP BY on just `user_id` on the `user_comments` table it would return the number of comments for each user right?

But what happens when a user does not make a comment? Then the user_id won't show up in the `user_comments` table.

Because we still need to account for users that did not make any comments in January 2019, we need to do a LEFT JOIN on `users` to `user_comments`, and then take the COUNT of a field in the `user_comments` table. That way we can get a 0 value for any users that did not end up commenting in January 2019.

```
SELECT users.id, COUNT(user_comments.user_id) AS comment_count
FROM users
LEFT JOIN user_comments
    ON users.id = user_comments.user_id
WHERE created_at BETWEEN '2019-01-01' AND '2019-01-31'
GROUP BY 1
```

The above CTE gives us a table with each user id and their corresponding comment count for the month of January 2019. Now that we have the comment count for each user, all we need to do is to group by the comment count to get a histogram.

```
WITH hist AS (
    SELECT users.id, COUNT(user_comments.user_id) AS comment_count
    FROM users
    LEFT JOIN user_comments
        ON users.id = user_comments.user_id
    WHERE created_at BETWEEN '2019-01-01' AND '2019-01-31'
    GROUP BY 1
)

SELECT comment_count, COUNT(*) AS frequency
FROM hist
GROUP BY 1
``` -->
