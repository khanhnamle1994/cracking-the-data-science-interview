## Question
This question was asked by: Tinder

There are two tables. One table is called `swipes` that holds a row for every Tinder swipe and contains a boolean column that determines if the swipe was a right or left swipe called `is_right_swipe`. The second is a table named `variants` that determines which user has which variant of an AB test.

Write a SQL query to output the average number of right swipes for two different variants of a feed ranking algorithm by comparing users that have swiped the first 10, 50, and 100 swipes on their feed.

*Tip: Users have to have swiped at least 10 times to be included in the subset of users to analyze the mean number of right swipes.*

**Example Input:**

**`variants`**

| id |  experiment | variant | user_id |
|:--:|:-----------:|:-------:|:-------:|
|  1 | feed_change | control |   123   |
|  2 | feed_change |   test  |   567   |
|  3 | feed_change | control |   996   |

**`swipes`**

| id | user_id | swiped_user_id | created_at | is_right_swipe |
|:--:|:-------:|:--------------:|:----------:|:--------------:|
|  1 |   123   |       893      | 2018-01-01 |        0       |
|  2 |   123   |       825      | 2018-01-02 |        1       |
|  3 |   567   |       946      | 2018-01-04 |        0       |
|  4 |   123   |       823      | 2018-01-05 |        0       |
|  5 |   567   |       952      | 2018-01-05 |        1       |
|  6 |   567   |       234      | 2018-01-06 |        1       |
|  7 |   996   |       333      | 2018-01-06 |        1       |
|  8 |   996   |       563      | 2018-01-07 |        0       |

*Note: created_at doesn't show timestamps but assume it is a datetime column.*

**Output:**

| mean_right_swipes | variant | swipe_threshold | num_users |
|:-----------------:|:-------:|:---------------:|:---------:|
|        5.3        | control |        10       |    9560   |
|        5.6        |   test  |        10       |    9450   |
|        20.1       | control |        50       |    2001   |
|        22.0       |   test  |        50       |    2019   |
|        33.0       | control |       100       |    590    |
|        34.0       |   test  |       100       |    568    |
