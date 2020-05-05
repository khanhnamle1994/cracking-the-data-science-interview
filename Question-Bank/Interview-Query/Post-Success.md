## Question
This question was asked by: Facebook

**post_events**

|   column   |   type   |
|:----------:|:--------:|
|   user_id  |    int   |
| created_at | datetime |
| event_name |  varchar |

In the table above, column `event_name` represents either ('enter', 'post', 'cancel') for when a user starts a post (enter), ends up canceling it (cancel), or ends up posting it (post).

Write a query to get the post success rate for each day over the past week.

Sample:

| user_id | created_at | event_name |
|:-------:|:----------:|:----------:|
|   123   | 2019-01-01 |    enter   |
|   123   | 2019-01-01 |    post    |
|   456   | 2019-01-02 |    enter   |
|   456   | 2019-01-02 |   cancel   |

<!-- ## Solution
Let's see if we can clearly define the metrics we want to calculate before just jumping into the problem. We want **post success rate for each day over the past week.**

To get that metric let's assume post success rate means: **(total posts entered) / (total posts created)**. Additionally since the success rate must be broken down by day, we must make sure that a post that is entered **must be completed on the same day**.

Cool, now that we have these requirements, it's time to calculate our metrics. We know we have to GROUP BY the date to get each day's posting success rate. We also have to break down how we can compute our two metrics of *total posts entered and total posts actually created.*

Let's look at the first one. Total posts entered can we calculated by a simple query such as filtering for where the event is equal to 'enter'.

```
SELECT COUNT(DISTINCT user_id)
FROM post_events
WHERE event_name = 'enter'
```

Now we have to get all of the users that also successfully created the post in the same day. We can do this with a join and set the correct conditions. The conditions are:
- Same user
- Successfully posted
- Same day

We can get those by doing a LEFT JOIN to the same table, and adding in those conditions. Remember we have to do a LEFT JOIN in this case because we want to use the join as a filter to where the conditions have been successfully met.

```
FROM post_events AS c1
LEFT JOIN post_events AS c2
    ON c1.user_id = c2.user_id
        AND c2.event = 'post'
        AND DATE(c1.created_at) = DATE(c2.created_at)
```

Bringing it all together now, we can now take the COUNT of users from c2 table and divide them from the base table c1.

```
SELECT
    DATE(created_at) AS dt
    , COUNT(DISTINCT c2.user_id) / COUNT(DISTINCT c1.user_id) AS success_rate
FROM post_events AS c1
LEFT JOIN post_events AS c2
    ON c1.user_id = c2.user_id
        AND c2.event = 'post'
        AND DATE(c1.created_at) = DATE(c2.created_at)
WHERE c1.event_name = 'enter'
    AND c1.created_at >= DATE_SUB(now(), INTERVAL 1 WEEK)
GROUP BY 1
``` -->
