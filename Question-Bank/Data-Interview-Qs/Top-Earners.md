## Question
You are given the following [dataset](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHL0PTD2m237pPW3WbnV0-2B-2FP-2FNQiZEk5-2BhcPcgF92WTu6nml7qRz9sg-2BLJmzZfqIr5TTlhPoGh7dWZKDgw03CuWYnQ-2BN2Y1XKZ09SJW6iFgoc-2B83p8IGAWpAtnXdIAI5fMow-3D-3D_8c6kLYfeKFgEvI6pydPvKCo5RIOwGXukDLGeEAsdKQMNM5uyNDXenEEAfR2KmaSI3QK-2BA2fuuKbNsyWgnY2M4PuE9BkQVim2fvllI6sYklvB4HMWr76EhkkusiGgbpd74Fc977ixB0vzvIY1Ni-2BVXEOVrtQj9imht81XSdx-2FllxBTd62FqQRls9xwhBOKnrMerqr2lFPa9TnEGgoHqu-2FnNtIfvNSR1Nl1QMFLnbVnCk-3D) of daily sales from a company's sales team. You can assume the table is called DailySales. Given this, write a SQL query to return the top sales person on each given day.

| Column Name | Column Type |   Short description   |
|:-----------:|:-----------:|:---------------------:|
|     Date    |    string   | Date of sales summary |
|     Name    |    string   |  Name of sales person |
|  Num_Sales  |      12     | Total number of sales |

<!-- ## Solution
```
SELECT
    Date,
    Name
FROM (
    SELECT
        Date,
        Name,
# assigning a rank to each sales person split apart by date and ordered by number of sales descending
        RANK() OVER(Partition By Date, Num_Sales DESC) as rank
    FROM DailySales
)
# selecting only the top sales person for each date
# note there may be multiple
WHERE rank = 1
``` -->
