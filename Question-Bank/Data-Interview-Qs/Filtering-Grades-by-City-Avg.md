## Question
Given the following table, called 'students', write a SQL query to count the # of students with grades above the Detroit average.

|   | age |      city     | grade |       name       |
|:-:|:---:|:-------------:|:-----:|:----------------:|
| 0 |  20 |    Detroit    |   88  |  Willard Morris  |
| 1 |  19 |    Detroit    |   95  |    Al Jennings   |
| 2 |  22 |    New York   |   92  |   Omar Mullins   |
| 3 |  21 | San Francisco |   70  | Spencer McDaniel |

<!-- ## Solution
```
# Base selection of the grade, count of students
SELECT
  grade,
  COUNT (*)
FROM students
# Need to include grade in group by as we're using aggregation function (count)
GROUP BY grade
# Here we can use SQL's 'having' function to filter on rows where the grade is > than the avg grade in Detroit
HAVING grade >
    (SELECT AVG(grade)
     FROM customer
     WHERE city = 'Detroit');
``` -->
