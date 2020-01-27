## Question
Given the following datasets, containing both price and calorie information for McDonald's food items:
* [McDonald's menu item prices](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHLx3MSbwZrt8DwLPWUgrfy-2FWYTmmmnVSeu5gKS69ghQpghilpIGfmsHCKyHuT6I8QlIrPzZIKhgjLpm7cmzI1vf0mLWGCYIs2-2BfuOHxZKwSc1U4QpO4sWsUyT8j5UOvI4dVfQPBxww-2BvCb1rAHltl2kc-3D_8c6kLYfeKFgEvI6pydPvKIky8fo6e6q8I8ARrSDNUWIoT8z8rN90QtY6AbniiHNtLiSquNLQg-2B-2BJZonpebuEK-2By8tuaHvcjF2El1euwvLhaKkBxyGOnd-2FL5W3GqMmXJuU5hZapuPXDl5CuB0duk5CYZjiCBbcNJSxpPDluc9-2BjLXyghUzhIY4kfWIHo-2BTTbz-2BEI2iXrWfi9N5vq-2BiRk-2FAW4kfSFAMKdULWieH-2FO6DuI-3D)
* [McDonald's menu item calories](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHLx3MSbwZrt8DwLPWUgrfy-2FWYTmmmnVSeu5gKS69ghQpghilpIGfmsHCKyHuT6I8QlIrPzZIKhgjLpm7cmzI1vf0mLWGCYIs2-2BfuOHxZKwSc18k4GKSng824i09r080325K2fARdP6wfgCCYNYrLLIng-3D_8c6kLYfeKFgEvI6pydPvKIky8fo6e6q8I8ARrSDNUWIoT8z8rN90QtY6AbniiHNtec06wbJc8Y8EKbMLzl0STHScrcVJRSayEgQraw0vjrz1jCMOcIH0WniPdUYIna-2FWmTtaRbwtZXdoVcX5VB6m77ys-2FUDr18zFZhyeKy3q3nLSHxEEfffcaIm8fvFBWW-2BortTiDa7gglMKaNWasZg5uYbLyaVKnOHbgESS37ON0hI-3D)

Write code to merge the two datasets, calculate the price per calorie of food, and stack rank the foods from the 'best value' (most calories per dollar*) to 'worst value' (least calories per dollar*).

*There are of course nuances in nutritional benefits, protein, etc, but we're simplifying here for the purposes of these definitions and assuming someone wants to optimize purely for cost of calories.*

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1-tUGrhYlE2KWRRSRJSwjlb4Q0KMQICOg) to view this solution in an interactive Colab (Jupyter) notebook

### 1) First, we can import + preview the two datasets

```
import numpy as np
import pandas as pd
# to visualize the results
import matplotlib.pyplot as plt
# First, read in, preview our CSV
food_calories = pd.read_csv('https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/mcD_food_calories.csv')
food_prices = pd.read_csv('https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/mcD_food_prices.csv')

food_calories.info()
#Output:
#RangeIndex: 58 entries, 0 to 57
#Data columns (total 2 columns):
#Food        58 non-null object
#Calories    58 non-null int64
#dtypes: int64(1), object(1)
#memory usage: 1008.0+ bytes

food_prices.info()
#Output:
#RangeIndex: 58 entries, 0 to 57
#Data columns (total 2 columns):
#Food         58 non-null object
#Price_USD    58 non-null float64
#dtypes: float64(1), object(1)
#memory usage: 1008.0+ bytes
```

### 2) Next, we can join the two datasets on the key 'Food'.

Note: all food records on both sides here happen to be unique and match, making for an easy join -- in this case it does not matter whether we use left, right, or inner join. However, this would be something to validate in an interview and think through prior to executing a join. For example, how would you adjust if the food price dataframe contained more unique records than the food calorie dataframe? What if both dataframes contained duplicate values? Etc.

```
# Join using built in pd.merge function (similar to a SQL join)
food_merged = food_prices.merge(food_calories, how='left', on="Food")
food_merged.head()
```

### 3) From here, we can create a new column with our cost per calorie calculation, then sort to obtain the requested ranking:

```
# Create the new column
food_merged['price_per_calorie'] = food_merged['Price_USD']/food_merged['Calories']

# Sort the dataframe in ascending order to show the 'best value foods first'
food_merged = food_merged.sort_values(by='price_per_calorie', ascending=True)

# Preview the 10 cheapest foods in terms of cost per calorie from our dataset
food_merged.head(10)

# Preview the 10 most expensive foods in terms of cost per calorie from our dataset
food_merged.tail(10)
```

As expected, premium items and more voluminous items like salads have a higher price per calorie, whereas more standard/dollar menu offerings have a lower cost per calorie. The pricing of course floats around depending on state (and country), but in general this trend checks out. -->
