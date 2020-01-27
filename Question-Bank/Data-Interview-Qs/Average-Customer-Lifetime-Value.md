## Question
Given the following [dataset](https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/teleco_user_data.csv), calculate the average lifetime value of a customer.

Below are a couple of definitions to help solve the problem:

* Lifetime value is average revenue divided by the churn rate
* Churn rate is defined as the total number of churned customers / total number of customers

Additionally, here is code to import the relevant packages as well as the dataset shown above, to help get you started:

```
#Importing packages
import pandas as pd
import numpy as np

#Read in the dataset
data = pd.read_csv("https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/teleco_user_data.csv")
#Convert these column types to int
data['TotalCharges'] = pd.to_numeric(data.TotalCharges, downcast='integer', errors='coerce')
data['MonthlyCharges'] = pd.to_numeric(data.MonthlyCharges, downcast='integer', errors='coerce')
```

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1ld0zaEwLK9MMsWQHX_cqYgsTJ1g6iSYz) to view this solution in an interactive Colab (Jupyter) notebook.

```
## 1) The first step is to calculate the average TotalCharges
average_total_charges = data['TotalCharges'].mean()

## 2) Next lets calculate the churn rate we need the total number of customers and we need the number that have churned
num_churned = len(data[data['Churn'] == "Yes"])
total_customers = len(data)

## 3) With the two above numbers we can calculate the overall churn rate
churn_rate = num_churned / total_customers

## 4) The last step is to calculate average lifetime value
ltv = average_total_charges / churn_rate
ltv
```

Output: **8604.218836195445** -->
