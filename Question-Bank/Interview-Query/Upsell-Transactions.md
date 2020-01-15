## Question
This question was asked by: Coinbase

Given a transaction table of product purchases, write a query to get the number of customers that were upsold by purchasing additional products.

Note that if the customer purchased two things on the same day that does not count as an upsell as they were purchased within a similar timeframe. Each row in the transactions table also represents an individual user product purchase.

**transactions**

|   column   |   type   |
|:----------:|:--------:|
|   user_id  |    int   |
| created_at | datetime |
| product_id |    int   |
|  quantity  |    int   |
|    price   |   float  |
