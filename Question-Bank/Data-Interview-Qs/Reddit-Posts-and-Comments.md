## Question
Suppose you're working for Reddit as an analyst. Reddit is trying to optimize its server allocation per subreddit, and you've been tasked with figuring out how much comment activity happens once a post is published.

Use your intuition to select a timeframe to query the data, as well as how you would want to present this information to the partnering team. The solution will be a SQL query with assumptions that you would need to state if this were asked in an interview. You have the following tables:

Table: **posts**

|  Column Name | Data Type |           Description          |
|:------------:|:---------:|:------------------------------:|
|      id      |  integer  |         id of the post         |
| publisher_id |  integer  |       id the user posting      |
|     score    |  integer  |        score of the post       |
|     time     |  integer  | post publish time in unix time |
|     title    |   string  |        title of the post       |
|    deleted   |  boolean  |      is the post deleted?      |
|     dead     |  boolean  |       is the post active?      |
| subreddit_id |  integer  |       id of the subreddit      |

Table: **comments**

|   Column Name  | Data Type |                    Description                    |
|:--------------:|:---------:|:-------------------------------------------------:|
|       id       |  integer  |                 id of the comment                 |
|    author_id   |  integer  |                id of the commenter                |
|     post_id    |  integer  |     id of the post the comment is nested under    |
| parent_comment |  integer  | id of parent comment that comment is nested under |
|     deleted    |  integer  |                is comment deleted?                |

Given the above, write a SQL query to highlight comment activity by subreddit. This problem is intended to test how you can think through vague/open-ended questions.

<!-- ## Solution
The first thing we need to decide is what period we want to analyze. I'm going to say that a post needs to be active for at least 2 hours, and we're going to look at the last 30 days of posts. The query below selects all posts with this criteria.

```
SELECT
  subreddit_id,
  id,
  time
FROM posts
WHERE time <= TIME_NOW() - 60*60*2
 AND time >= TIME_NOW() - 60*60*24*30
```

Since our goal is to optimize server allocation per subreddit, I'm going to aggregate comment activity per subreddit and show the top active subreddits. I'm going to determine how 'active' a subreddit is based on # of comments per day. This of course could be defined a number of ways, but we could easily change our baseline query based on our partner's definition. Proceeding with this assumption, we next need to join the 'posts' table with the 'comments' table. Remember, we only want comments made within 2 hours of the publish time of the post.

```
SELECT DISTINCT
  posts.subreddit_id,
  posts.id as post_id,
  posts.time,
  comments.id as comments_id
FROM posts as posts
  # we're doing a left join so we don't exclude posts that don't have comments
  LEFT JOIN comments as comments
     ON posts.id = comments.post_id
     # we're only joining comments that were posted within 2
     # hours of the post publish time
     AND posts.time + 60*60*2 >= comments.time
WHERE posts.time <= TIME_NOW() - 60*60*2
 AND posts.time >= TIME_NOW() - 60*60*24*30
```

Lastly, we're going to aggregate each the number of comments by the post_id, subreddit_id, and date. Then, on top of that aggregation, we're going to sum the number of comments by each day and subreddit. This will yield the number comments per day per subreddit.

```
SELECT
  subreddit_id,
  date,
  SUM(num_comments_in_2_hours) as num_comments_per_day
FROM(
 SELECT
     subreddit_id,
     post_id,
     DATE_TRUNC(DATE(posts.time), day) as date,
     COUNT(distinct comments_id) as num_comments_in_2_hours
 FROM(
     SELECT DISTINCT
         posts.subreddit_id,
         posts.id as post_id,
         posts.time,
         comments.id as comments_id
     FROM posts as posts
         LEFT JOIN comments as comments
             ON posts.id = comments.post_id
             AND posts.time + 60*60*2 >= comments.time
     WHERE posts.time <= TIME_NOW() - 60*60*2
         AND posts.time >= TIME_NOW() - 60*60*24*30
     )
 GROUP BY 1, 2, 3
 )
GROUP BY 1, 2
```

Part of this question is figuring out "how to present this information to a partner team." I would initially present this by classifying each subreddit by activity level (possibly high, medium, low or maybe using a 10 point scale -- it depends on what the data distribution looks like). Within each activity level, I would then classify the activity distribution visually by using a box and whisker plot. If the partner needs more information, then I would work with them to figure out what else they would need to answer the questions at hand. -->
