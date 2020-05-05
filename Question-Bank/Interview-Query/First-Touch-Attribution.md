## Question
This question was asked by: Google

The schema above is for a retail online shopping company.

**`attribution` table**

|   column   |   type   |
|:----------:|:--------:|
|     id     |    int   |
| created_at | datetime |
| session_id |    int   |
|   channel  |  varchar |
| conversion |  boolean |

**`user_sessions` table**

|   column   | type |
|:----------:|:----:|
| session_id |  int |
|   user_id  |  int |

The attribution table logs each user visit where a user comes onto their site to go shopping. If conversion = 1, then on that session visit the user converted and bought an item. The channel column represents which advertising platform the user got to the shopping site on that session. The `user_sessions` table maps each session visit back to the user.

First touch attribution is defined as the channel to which the converted user was associated with when they first discovered the website. Calculate the first touch attribution for each user_id that converted.

**Example output:**

| user_id |  channel |
|:-------:|:--------:|
|   123   | facebook |
|   145   |  google  |
|   153   | facebook |
|   172   |  organic |
|   173   |   email  |


<!-- ## Solution
First touch attribution is tricky because we have to look at the full span of the user's visits ONLY if they converted as a customer.

Therefore we need to do two actions: subset all of the users that converted to customers and figure out their first session visit to attribute the actual channel.

```
-- grab all of the users that converted and create a CTE
WITH conv AS (
    SELECT user_id
    FROM attribution AS a
    INNER JOIN user_sessions AS us
        ON a.session_id = us.session_id
    WHERE conversion = 1
    GROUP BY 1 -- group by to get distinct user_ids
)
```

The CTE above subsets for distinct users that have converted. Given these converted user_ids, we now have to figure out on which session they actually first visited the site. Given that the session that the user first visited could be different from the one they converted, we need to identify the first session.

We can do this by using the `created_at` field to identify the first time the user visited the site. If we group by the `user_id`, we can aggregate the`created_at` date time and apply the **MIN()** function to find the user's first visit. Afterwards all we then have to do is join the minimum `created_at` date time and `user_id` back to the attribution table to find the session and channel for which the user first arrived on the site.

```
-- grab all of the users that converted. Create a CTE from it.
WITH conv AS (
    SELECT user_id
    FROM attribution AS a
    INNER JOIN user_sessions AS us
        ON a.session_id = us.session_id
    WHERE conversion = 1
    GROUP BY 1 -- group by to get distinct user_ids
)

-- get the first session by user_id and created_at time.
first_session AS (
    SELECT
        min(created_at) AS min_created_at
        , user_id
    FROM user_sessions AS us
    INNER JOIN conv
        ON us.user_id = conv.user_id
    INNER JOIN attribution AS a
        ON a.session_id = us.session_id
    GROUP BY user_id
)

-- join user_id and created_at time back to the original table.
SELECT user_id, channel
FROM attribution
JOIN user_sessions AS us
    ON attribution.session_id = us.session_id
-- now join the first session to get a single row for each user_id
JOIN first_session AS ft
-- double join
    ON ft.created_at = attribution.created_at
        AND ft.user_id = us.user_id
``` -->
