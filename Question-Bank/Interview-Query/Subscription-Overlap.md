## Problem
This question was asked by: Twitch

**subscriptions table**

|   column   | type |
|:----------:|:----:|
|   user_id  |  int |
| start_date | date |
|  end_date  | date |

Given a table of product subscriptions with a subscription start date and end date for each user, write a query that returns true or false whether or not each user has a subscription date range that overlaps with any other user.

**Example:**

| user_id | start_date |  end_date  |
|:-------:|:----------:|:----------:|
|    1    | 2019-01-01 | 2019-01-31 |
|    2    | 2019-01-15 | 2019-01-17 |
|    3    | 2019-01-29 | 2019-02-04 |
|    4    | 2019-02-05 | 2019-02-10 |

**Output**

| user_id | overlap |
|:-------:|:-------:|
|    1    |   True  |
|    2    |   True  |
|    3    |   True  |
|    4    |  False  |

<!-- ## Solution
Let's take a look at each of the conditions first and see how they could be triggered. Given two date ranges, what determines if the subscriptions would overlap?

Let's set an example with two dateranges: A and B.

Let ConditionA>B demonstrate that DateRange A is completely after DateRange B.

```
_                        |---- DateRange A ------|
|---Date Range B -----|                           _
```

ConditionA>B is true if **StartA > EndB**.

Let ConditionB>A demonstrate that DateRange B is completely after DateRange A.

```
|---- DateRange A -----|                       _
 _                          |---Date Range B ----|
```

Condition B>A is true if **EndA < StartB**.

**Overlap then exists if neither condition is held true**. In that if one range is neither completely after the other, nor completely before the other, then they must overlap.

De Morgan's laws says that:

Not (A Or B) <=> *Not A And Not B*.

Which is equivalent to: *Not (StartA > EndB) AND Not (EndA < StartB)*

Which then translates to: **(StartA <= EndB) and (EndA >= StartB)**.

Awesome, we've figured out the logic. Given this condition, how can we apply it to SQL?

Well we know we have to use the condition as logic for our join. In this case we'll be joining to the same table but comparing each user to a different user in the table. We also want to run a left join to match each other user that satisfies our condition. In this case it should look like this:

```
FROM subscriptions AS s1
LEFT JOIN subscriptions AS s2
    ON s1.user_id != s2.user_id
        AND s1.start_date <= s2.end_date
        AND s1.end_date >= s2.start_date
```

We've set s1 as our A and s2 as our B. Given the conditional join, a user_id from s2 should exist for each user_id in s1 on the condition where there exists overlap between the dates.

Wrapping it all together now, we can group by the s1.user_id field and just check if any value exists true for where s2.user_id IS NOT NULL.

```
SELECT
    s1.user_id
    , MAX(CASE WHEN s2.user_id IS NOT NULL THEN 1 ELSE 0 END) AS overlap
FROM subscriptions AS s1
LEFT JOIN subscriptions AS s2
    ON s1.user_id != s2.user_id
        AND s1.start_date <= s2.end_date
        AND s1.end_date >= s2.start_date
GROUP BY 1
``` -->
