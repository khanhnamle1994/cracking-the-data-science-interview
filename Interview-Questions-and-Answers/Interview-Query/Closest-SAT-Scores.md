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
