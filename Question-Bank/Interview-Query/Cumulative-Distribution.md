## Question
This question was asked by: Square

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
|   user_id  |  integer |
|    body    |  string  |
| created_at | datetime |

Given the two tables, write a SQL query that creates a cumulative distribution of number of comments per user. Assume bin buckets class intervals of one.

<!-- ## Solution
This question is similar to the question about creating a histogram from writing a query. However creating a cumulative distribution plot requires another couple of steps.

We can start out with the query to count the frequency of each user by joining `users` to `user_comments` and then grouping by the user id to get the number of comments per user.

```
WITH hist AS (
    SELECT users.id, COUNT(user_comments.user_id) AS frequency
    FROM users
    LEFT JOIN user_comments
        ON users.id = user_comments.user_id
    GROUP BY 1
)
```

Now we can group on the frequency column to get the distribution per number of comments. This is our general histogram distribution of number of comments per user. Note that since we are getting the COUNT of the `user_comments` table, users that comment 0 times will show up in the 0 frequency bucket.

```
WITH freq AS (
    SELECT frequency, COUNT(*) AS num_users
    FROM hist
    GROUP BY 1
)
```

Now that we have our histogram, how do we get a cumulative distribution? Specifically we want to see our frequency table go from:

| frequency | count |
|:---------:|:-----:|
|     0     |   10  |
|     1     |   15  |
|     2     |   12  |

to:

| frequency | count |
|:---------:|:-----:|
|     0     |   10  |
|     1     |   25  |
|     2     |   27  |

Let's see if we can find a pattern and logical grouping that gets us what we want. The constraints given to us are that we will probably have to self-join since we can compute the cumulative total from the data in the existing histogram table.

If we can model out that computation, we'll find that the cumulative is taken from the sum all of the frequency counts **lower than the specified frequency index**. In which we can then run our self join on a condition where we set the **left f1 table frequency index as greater than the right table frequency index**.

```
FROM freq AS f1
LEFT JOIN freq AS f2
    ON f1.frequency >= f2.frequency
```

Now we just have to sum up the `num_users` column while grouping by the f1.frequency index.

```
WITH hist AS (
    SELECT users.id, COUNT(user_comments.user_id) AS frequency
    FROM users
    LEFT JOIN user_comments
        ON users.id = user_comments.user_id
    GROUP BY 1
)

WITH freq AS (
    SELECT frequency, COUNT(*) AS num_users
    FROM hist
    GROUP BY 1
)

SELECT f1.frequency, SUM(f1.num_users) AS cum_total
FROM freq AS f1
LEFT JOIN freq AS f2
    ON f1.frequency >= f2.frequency
GROUP BY 1
``` -->
