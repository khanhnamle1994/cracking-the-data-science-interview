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
