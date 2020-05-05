## Problem
This question was asked by: Calm

**'likes' table**

|   column   |   type   |
|:----------:|:--------:|
|   user_id  |    int   |
| created_at | datetime |
|  liked_id  |    int   |

A dating websites schema is represented by a table of people that like other people. The table has three columns. One column is the *user_id*, another column is the *liker_id* which is the user_id of the user doing the liking, and the last column is the date time that the like occured.

Write a query to count the number of liker's likers (the users that like the likers) if the liker has one.

**Example**:

**Input**:

| user | liker |
|:----:|:-----:|
|   A  |   B   |
|   B  |   C   |
|   B  |   D   |
|   D  |   E   |

**Output**:

| user | count |
|:----:|:-----:|
|   B  |   2   |
|   D  |   1   |

<!-- ## Solution
The solution prompt in itself is a bit confusing and it helps to look at the examples to understand the exact link between the users and likers.

Each user has a specified liker but since the likers can also be users, they can also show up in the left most column as we see with values B and D. B and D happen to be the only ones that are **both users and likers**, which means that they would be only values that could be specified as the liker's likers.

Given we've figured out this relationship, let's solve the SQL question.

Since we want to join on the condition where a value can be a user and liker, we need to run an INNER JOIN between the table's liker column to the user column and then group by the liker column.

```
SELECT *
FROM liker AS l1
INNER JOIN liker AS l2
    ON l1.liker = l2.user
```

**SQL output example from join**

| user_l1 | liker_l1 | user_l2 | liker_l2 |
|:-------:|:--------:|:-------:|:--------:|
|    A    |     B    |    B    |     C    |
|    A    |     B    |    B    |     D    |
|    B    |     D    |    D    |     E    |

Looking at the output of our join, we can see that the second degree likers are represented in the `liker_l2` column and the first degree likers are represented in the `liker_l1` column. Now all we have to do is group by the `liker_l1` column and count the distinct values of the `liker_l2` column.

```
SELECT
    l1.liker
    , COUNT(DISTINCT l2.liker) AS second_likers
FROM liker AS l1
INNER JOIN liker AS l2
    ON l1.liker = l2.user
GROUP BY 1
ORDER BY 1
``` -->
