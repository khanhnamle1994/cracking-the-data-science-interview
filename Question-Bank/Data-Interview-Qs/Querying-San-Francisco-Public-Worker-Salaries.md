## Problem
You have the following [dataset](https://u4221007.ct.sendgrid.net/ls/click?upn=qwT-2Bl0U064-2B7oRNpPgUya8GjH-2BHfJcP4935QwszXa5mSy8U66towob3H-2Bk5f4Ilm0HnVZbp-2FMFUxxwYoPjAKfjYUuHlbwTxO-2BiFniXopj81bErmxj9PbPTOxWgTnFtlqJGsy_UMR2KhLP9Az12hwnQT88B3haJMFeZ0Vp3Mhlf31lRPYhAQLxqX9-2FWHua8sl4YDSlzwkbThei89V5t3R-2FFW1MAnq9PHYjwE-2FRzyM30Xls8aFBMZ2IVQ30gtd6YzDirGQnWoKS6Yp5TPJ8g40nCXrsNJgX0t-2BM55gQao3tMV-2FsM7tulWqz3Sy-2BC9h98lXWGx2ty2oeiO9A0c7smbPvLo5V7E1uEQYiTiCKFDSF6GC6ZcM-3D) containing salaries for public workers of San Francisco, CA. Let's assume this data is in a table called sf_salaries.

Can you write a SQL query to find the top 3 highest paid and bottom 3 lowest paid job titles?

<!-- ## Solution

```
# adding a label called "type" to the highest salaries
 # so we can easily differentiate from low salaries
SELECT
    *,
    'highest_salaries' AS type
FROM
 # selecting the highest salaries
    (SELECT
        JobTitle,
        TotalPayBenefits
    FROM sf_salaries
    ORDER BY TotalPayBenefits DESC
    LIMIT 3)

 #putting these two sets together with a union
UNION

 # adding a label called "type" to the lowest salaries
 # so we can easily differentiate from high salaries
SELECT
    *,
    'lowest_salaries' as type
FROM
 # selecting the lowest salaries
    (SELECT
        JobTitle,
        TotalPayBenefits
    FROM sf_salaries
    ORDER BY TotalPayBenefits ASC
    LIMIT 3)

ORDER BY TotalPayBenefits ASC
 #order by is not 100% necessary by guarantees our data
 #will come out sorted
``` -->
