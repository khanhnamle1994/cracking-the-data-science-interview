## Problem
Below are two table schemas for a popular music streaming application:

Table 1: **user_song_log**

| Column Name | Data Type |                               Description                               |
|:-----------:|:---------:|:-----------------------------------------------------------------------:|
|   user_id   |     id    |                         id of the streaming user                        |
|  timestamp  |  integer  | timestamp of when the user started listening to the song, epoch seconds |
|   song_id   |  integer  |                              id of the song                             |
|  artist_id  |  integer  |                             id of the artist                            |

Table 2: **song_info**

Given the above, can you write a SQL query to estimate the average number of hours a user spends listening to music daily? You can assume that once a given user starts a song, they listen to it in its entirety.

| Column Name | Data Type |        Description        |
|:-----------:|:---------:|:-------------------------:|
|   song_id   |  integer  |       id of the song      |
|  artist_id  |  integer  |      id of the artist     |
| song_length |  integer  | length of song in seconds |

<!-- ## Solution
There are a couple of ways you can approach this problem, but here I'll step through what I believe is a pretty intuitive way to view the problem. We'll approach this problem in 2 steps:
1, Determine how to get the number of hours listened to per day for a given record in user_song_log.
2. Aggregate step (1) on a daily basis across all users.

1) First, we'll focus on building a query that allows us to calculate the number of hours each user in our table spends listening to music on our stream application each day. Our first query will join the user_song_log to the song_info table in order to get the song length as well as the day the user listened to the song. Then we will aggregate the song_length over the day (which is derived from the timestamp) and the user fields.

```
SELECT
    user_id,
    date_format(from_unixtime(timestamp), '%Y-%m-%d') AS day,
    SUM(song_length)/(60*60) as listen_time_hours
FROM
    user_song_log a
    JOIN
    song_info b
    on a.song_id = b.song_id
GROUP BY 1, 2
```

2) Now that we have the data aggregated at the user/day level, we can now roll it up into a daily level (across all users) by building a query to aggregate the query we created in part 1.

```
SELECT
    day,
    AVG(listen_time_hours) as avg_time_user
FROM(
    SELECT
        user_id,
        date_format(from_unixtime(timestamp), '%Y-%m-%d') AS day,
        SUM(song_length)/(60*60) as listen_time_hours
    FROM
        user_song_log a
        JOIN
        song_info b
        on a.song_id = b.song_id
    GROUP BY 1, 2
)
GROUP BY
    day
``` -->
