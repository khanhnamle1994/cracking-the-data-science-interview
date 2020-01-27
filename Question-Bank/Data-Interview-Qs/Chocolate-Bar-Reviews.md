## Question
You have the following [dataset](https://drive.google.com/file/d/1wG27Iex7Ymg2gc-5UdcnXtuhP7aUBSNK/view) of chocolate bar ratings. You can assume you have this data in a table called InternationalChocolateRatings.

Can you write a SQL query to summarize the BroadBeanOrigin for US manufactured chocolate bars and provide the number of reviews, average rating, and average cocoa percent?

|        Column Name in CSV        | Column Name for SQL | Column Type |                                  Short description                                 |
|:--------------------------------:|:-------------------:|:-----------:|:----------------------------------------------------------------------------------:|
|              Company             |       Company       |    string   |                      Name of the company manufacturing the bar                     |
| Specific Bean Origin or Bar Name |  SpecificBeanOrigin |    string   |                   The specific geo-region of origin for the bar.                   |
|             REF value            |       REFValue      |     int     | Value linked to when the review was entered in the database. Higher = more recent. |
|            Review Date           |      ReviewDate     |     int     |                         Year of publication of the review.                         |
|           Cocoa Percent          |     CocoaPercent    |    double   |          Cocoa percentage (darkness) of the chocolate bar being reviewed.          |
|         Company Location         |   CompanyLocation   |    string   |                             Manufacturer base country.                             |
|              Rating              |        Rating       |     int     |                             Expert rating for the bar.                             |
|             Bean Type            |       BeanType      |    string   |                   The variety (breed) of bean used, if provided.                   |
|         Broad Bean Origin        |   BroadBeanOrigin   |    string   |                    The broad geo-region of origin for the bean.                    |

<!-- ## Solution

```
SELECT
    BroadBeanOrigin,
    COUNT(1) as num_reviewers,
    AVG(Rating) as average_rating,
    AVG(CocoaPercent) as average_cocoa_percent
    RANK() OVER(PARTITION BY Rating DESC) as rank
FROM InternationalChocolateRatings
WHERE CompnayLocation = "U.S.A"
GROUP BY 1
ORDER BY 3 DESC
``` -->
