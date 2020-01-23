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
