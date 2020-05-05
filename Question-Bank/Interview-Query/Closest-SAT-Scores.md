## Question
This question was asked by: Dropbox

Given a table of students and their SAT test scores, write a query to return the two students with the closest test scores with the score difference. Assume a random pick if there are multiple students with the same score difference.

**`scores` table**

| Column Name | Data Type |
|:-----------:|:---------:|
|      id     |  integer  |
|   student   |  varchar  |
|    score    |  integer  |

**Example**:

Input:

| id | student | score |
|:--:|:-------:|:-----:|
|  1 |   Jack  |  1700 |
|  2 |  Alice  |  2010 |
|  3 |  Miles  |  2200 |
|  4 |  Scott  |  2100 |

Output:

| one_student | other_student | score_diff |
|:-----------:|:-------------:|:----------:|
|    Alice    |     Scott     |     90     |

<!-- ## Solution
The question requires thinking about SQL in a creative manner. Given it's only one table with two columns, we have to self-reference different creations of the same table. It's helpful to think about these problems in the form of two different tables with the same values.

There are two parts to this question, the first part is figuring out each combination of two students and their SAT scores. The second part is figuring out which two students scores are the closest.

Let's work on the first part and assume that since we are self-referencing the same table. We have two of the same tables, one called s1 and one called s2. Since we want to compare each student against each other student, we can do a variation of the CROSS JOIN by setting:

```
INNER JOIN scores AS s2
    ON s1.student != s2.student
```

This way we're comparing each student to every other student by not comparing a student to his/herself. However, if we run this statement let's look at what the output would look like with an example of just two students:

| s1.id | s1.student | s1.score | s2.id | s2.student | s2.score |
|:-----:|:----------:|:--------:|:-----:|:----------:|:--------:|
|   1   |    Jack    |   1700   |   2   |    Alice   |   2010   |
|   2   |    Alice   |   2010   |   1   |    Jack    |   1700   |

We're seeing the duplication from the CROSS JOIN. How do we de-dupe the values so we only get one? Simply enough, we can add a **one-way condition in the self-reference** in which the join will satisfy and return one of the two values. In this case we can add another condition in which s1.id > s2.id or s2.id < s1.id.

Therefore all we need to do now is subtract each score from each other and to grab the absolute value and order by smallest to largest score difference and then limit the first row.

```
SELECT
    s1.student AS one_student
    , s2.student AS other_student
    , ABS(s1.score - s2.score) AS score_diff
FROM scores AS s1
INNER JOIN scores AS s2
    ON s1.id != s2.id
        AND s1.id > s2.id
ORDER BY 3 ASC
LIMIT 1
``` -->
