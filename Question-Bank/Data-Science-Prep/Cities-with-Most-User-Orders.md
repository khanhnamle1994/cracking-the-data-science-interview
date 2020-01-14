## Problem
This problem was asked by Robinhood.

Assume you are given the below tables for trades and users. Write a query to list the top 3 cities which had the highest number of completed orders.

**Trades**

| column_name |   type   |
|:-----------:|:--------:|
|   order_id  |  integer |
|   user_id   |  integer |
|    symbol   |  string  |
|    price    |   float  |
|   quantity  |  integer |
|     side    |  string  |
|    status   |  string  |
|  timestamp  | datetime |

**Users**

| column_name |   type   |
|:-----------:|:--------:|
|   used_id   |  integer |
|     city    |  string  |
|    email    |  string  |
| signup_date | datetime |
