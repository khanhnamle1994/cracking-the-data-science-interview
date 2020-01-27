## Question
Suppose you're an analyst for a major US hotel chain which has locations all over the US. Your marketing team is planning a promotion focused around loyal customers, and they are trying to forecast how much revenue the promotion will bring in. However, they need help from you to understand how much revenue comes from "loyal" customers to plug into their model.

A "loyal" customer is defined as

(1) having a membership with your company's point system,

(2) having >2 stays at each hotel the customer visited,

(3) having stayed at 3 different locations throughout the US.

You have a table showing all transactions made in 2017. The schema of the table is below:

Table: **customer_transactions**

|    Column Name   | Data Type |                         Description                        |
|:----------------:|:---------:|:----------------------------------------------------------:|
|    customer_id   |     id    |                     id of the customer                     |
|     hotel_id     |  integer  |                     unique id for hotel                    |
|  transaction_id  |  integer  |                 id of the given transaction                |
|    first_night   |   string  |   first night of the stay, column format is "YYYY-mm-dd"   |
| number_of_nights |  integer  |          # of nights the customer stayed in hotel          |
|    total_spend   |  integer  |             total spend for transaction, in USD            |
|     is_member    |  boolean  | indicates if the customer is a member of our points system |

Given this, can you write a SQL query that calculates percent of revenue loyal customers brought in 2017?

<!-- ## Solution
We are first going to figure out how to get the customer_ids of "loyal" customers, then we will figure out how to calculate the percent of revenue in 2017.

1) Solving for the first condition, having a membership for the points system is an easy add on and we'll be incorporating this condition in each subquery. Let's skip to the 2nd condition, "having > 2 stays at any hotel location". We'll aggregate the number of transactions for each unique customer and hotel_id, we'll build a query on top of that underlying query to get customer_ids where the # of transactions is > 2.

```
SELECT
     customer_id
FROM (
     SELECT
        customer_id,
        hotel_id,
        COUNT(DISTINCT transaction_id) as num_transactions
    FROM customer_transactions
    WHERE is_member = TRUE
    GROUP BY 1, 2
    )
WHERE num_transactions > 2
```

2) Next lets focus on the 3rd condition, staying at 3 different locations. For this one, we'll count hotel_ids for each customer and build a query on top to ensure we include only customers that have > 3 locations.

```
SELECT
     customer_id
FROM (
     SELECT
        customer_id,
        COUNT(DISTINCT hotel_id) as num_hotels
    FROM customer_transactions
    WHERE is_member = TRUE
    GROUP BY 1
    )
WHERE num_hotels > 3
```

3) Now we have 2 different queries that solve conditions (1 + 2) and (1 + 2), we need to be able to combine these two lists. We can accomplish that with a simple UNION clause

```
SELECT
     customer_id
FROM (
     SELECT
        customer_id,
        hotel_id,
        COUNT(DISTINCT transaction_id) as num_transactions
    FROM customer_transactions
    WHERE is_member = TRUE
    GROUP BY 1, 2
    )
WHERE num_transactions > 2

 UNION

SELECT
     customer_id
FROM (
     SELECT
        customer_id,
        COUNT(DISTINCT hotel_id) as num_hotels
    FROM customer_transactions
    WHERE is_member = TRUE
    GROUP BY 1
    )
WHERE num_hotels > 3
```

4. Now that we have a list of all customers that have met the loyal customer conditions, we can focus on getting revenue for these customers. We're going to build a query on top of the 2 queries and aggregate total revenue.

```
SELECT
     SUM(total_spend) as total_loyal_revenue
FROM customer_transactions
WHERE customer_id in (
     SELECT
         customer_id
     FROM (
          SELECT
             customer_id,
             hotel_id,
             COUNT(DISTINCT transaction_id) as num_transactions
         FROM customer_transactions
         WHERE is_member = TRUE
         GROUP BY 1, 2
         )
     WHERE num_transactions > 2
      UNION
     SELECT
          customer_id
     FROM (
          SELECT
             customer_id,
             COUNT(DISTINCT hotel_id) as num_hotels
         FROM customer_transactions
         WHERE is_member = TRUE
         GROUP BY 1
         )
     WHERE num_hotels > 3
 )
 ```

 5) We still don't have what we're looking for, so we need to pair the query we just built along with total revenue for all hotels across the US. To get the total revenue and total loyal revenue in a single query, I'll join the query we have in step 4 along with a much simplier query to get total revenue. We'll join on 1=1 which will pair everything in the 1st table with everything in the 2nd table (getting us exactly what we need).

 ```
 SELECT
 loyal.total_loyal_revenue*100.0/total.total_revenue as pct_loyal_revenue
FROM (
     (SELECT
          SUM(total_spend) as total_loyal_revenue
     FROM customer_transactions
     WHERE customer_id in (
          SELECT
              customer_id
          FROM (
               SELECT
                  customer_id,
                  hotel_id,
                  COUNT(DISTINCT transaction_id) as num_transactions
              FROM customer_transactions
              WHERE is_member = TRUE
              GROUP BY 1, 2
              )
          WHERE num_transactions > 2
           UNION
          SELECT
               customer_id
          FROM (
               SELECT
                  customer_id,
                  COUNT(DISTINCT hotel_id) as num_hotels
              FROM customer_transactions
              WHERE is_member = TRUE
              GROUP BY 1
              )
          WHERE num_hotels > 3
      )
     ) as loyal
      JOIN

     (
         SELECT
          SUM(total_spend) as total_revenue
      FROM customer_transactions
     ) as total
      on 1=1
)
 ``` -->
