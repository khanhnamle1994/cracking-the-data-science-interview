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

<!-- ## Solution
If you're a data scientist in charge of improving recommendations at a company and you develop an algorithm, how do you know if it performs better than the existing one?

One metric to measure performance is called **precision** (also called positive predictive value), which has applications in machine learning as well as information retrieval. It is defined as the **fraction of relevant instances among the retrieved instances.**

Given the problem set of measuring two feed ranking algorithms, we can break down this problem as measuring the mean precision between two different algorithms by comparing **average right swipes for two different populations** for the users first 10, 50, and 100 swipes.

We're given two tables, one called `variants` that essentially breaks down which test variant each user has received. It contains a column named `experiments` that we have to filter on for the `feed_change` experiment. We know we have to join this table back to the `swipes` table in order to differentiate both of the variants from each other.

The other table, `swipes`, is a transaction type table, meaning that it logs each users activity in the app. In this case, it's left and right swipes on other users.

Given the problem set, the first step is to formulate a way to average the right swipes for each user that satisfies the conditions of swiping at least 10, 50, and 100 total swipes. Given this condition, the first thing we have to do is **add a rank column to the swipe table**. That way we can look at each user's first X swipes.

```
WITH swipe_ranks AS (
    SELECT
        swipes.user_id
        , variant
        , RANK() OVER (
            PARTITION BY user_id ORDER BY created_at ASC
        ) AS rank
        , is_right_swipe
    FROM swipes
    INNER JOIN variants
        ON swipes.user_id = variants.user_id
    WHERE experiment = 'feed_change'
)
```

Observe how we implement a RANK function by partitioning by user_id and ordering by the created_at field. This gives us a rank of 1 for the first swipe the user made, 2 for the second, and etc...

Now our swipe_ranks table looks like this:

| user_id | variant | created_at | rank | is_right_swipe |
|:-------:|:-------:|:----------:|:----:|:--------------:|
|   123   | control | 2018-01-01 |   1  |        0       |
|   123   | control | 2018-01-02 |   2  |        1       |
|   567   |   test  | 2018-01-04 |   1  |        0       |
|   123   | control | 2018-01-05 |   3  |        0       |
|   567   |   test  | 2018-01-05 |   2  |        1       |
|   567   |   test  | 2018-01-06 |   3  |        1       |
|   996   | control | 2018-01-06 |   1  |        1       |
|   996   | control | 2018-01-07 |   2  |        0       |

Notice how the rank value does not reach above 3 in our sample data. Since each user needs to swipe on at least 10 users to reach the minimum swipe threshold, each of these users would be subsetted out of the analysis.

Constructing a query, we can create a subquery that specifically gets all the users that swiped at least 10 times. We can do that by using a `COUNT(*)` function or by looking where the **rank column is greater than 10.**

```
SELECT user_id
FROM swipe_ranks
WHERE rank > 10
GROUP BY 1
```

Then we can rejoin these users into the original swipe ranks table, group by the experiment variant, and take an average of the number of right swipes each user made where the rank was less than 10.

Remember that we have to specify a filter for the rank column because we cannot analyze swipe data greater than the threshold we are setting since the recommendation algorithm is intended to move **more relevant matches to the top of the feed.**

```
SELECT
    variant
    , CAST(SUM(is_right_swipe) AS DECIMAL)/COUNT(*) AS mean_right_swipes
    , 10 AS swipe_threshold
    , COUNT(DISTINCT user_id) AS num_users
FROM swipe_ranks AS sr
INNER JOIN (
    SELECT user_id
    FROM swipe_ranks
    WHERE rank > 10
    GROUP BY 1
) AS subset
    ON subset.user_id = sr.user_id
WHERE rank <= 10
GROUP BY 1
```

Awesome! This should work. Notice this value gives us the value for only the threshold of rank under 10. We can copy most of the code and re-use it for 50 and 100 by unioning the tables together. Putting it all together now.

```
WITH swipe_ranks AS (
    SELECT
        swipes.user_id
        , variant
        , RANK() OVER (
            PARTITION BY user_id ORDER BY created_at ASC
        ) AS rank
        , is_right_swipe
    FROM swipes
    INNER JOIN variants
        ON swipes.user_id = variants.user_id
    WHERE experiment = 'feed_change'
)

SELECT
    variant
    , CAST(SUM(is_right_swipe) AS DECIMAL)/COUNT(*) AS mean_right_swipes
    , 10 AS swipe_threshold
    , COUNT(DISTINCT user_id) AS num_users
FROM swipe_ranks AS sr
INNER JOIN (
    SELECT user_id
    FROM swipe_ranks
    WHERE rank > 10
    GROUP BY 1
) AS subset
    ON subset.user_id = sr.user_id
WHERE rank <= 10
GROUP BY 1

UNION ALL

SELECT
    variant
    , CAST(SUM(is_right_swipe) AS DECIMAL)/COUNT(*) AS mean_right_swipes
    , 50 AS swipe_threshold
    , COUNT(DISTINCT user_id) AS num_users
FROM swipe_ranks AS sr
INNER JOIN (
    SELECT user_id
    FROM swipe_ranks
    WHERE rank > 50
    GROUP BY 1
) AS subset
    ON subset.user_id = sr.user_id
WHERE rank <= 50
GROUP BY 1

UNION ALL

SELECT
    variant
    , CAST(SUM(is_right_swipe) AS DECIMAL)/COUNT(*) AS mean_right_swipes
    , 100 AS swipe_threshold
    , COUNT(DISTINCT user_id) AS num_users
FROM swipe_ranks AS sr
INNER JOIN (
    SELECT user_id
    FROM swipe_ranks
    WHERE rank > 100
    GROUP BY 1
) AS subset
    ON subset.user_id = sr.user_id
WHERE rank <= 100
GROUP BY 1
``` -->
