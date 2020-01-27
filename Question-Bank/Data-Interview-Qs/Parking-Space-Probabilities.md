## Question
Suppose you're analyzing the utilization of a small parking lot you invested in. Customers enter the automated lot, receive a ticket, and select one of 20 spots at random (e.g. they are not assigned). Upon leaving, customers pay in relation to their time in the lot.

You have collected the following [dataset](https://docs.google.com/spreadsheets/d/1rpTArHXCSMV0_WryClhhmQfDZzSEACt_EuLh8rxuvDU/edit#gid=0), which shows spot utilization for a month. Each # in the dataset corresponds to a spot # (1-20) and the # of times it appears in a row corresponds to the frequency of how many customers parked in that spot.

Using all of this information, write code to visualize the [Probability Mass Function](https://en.wikipedia.org/wiki/Probability_mass_function) (PMF) of your customers' spot selections. Your resultant chart should show each spot # (1-20) along with the probability of that spot being chosen based on your dataset. You can ignore seasonality and assume this month represents a standard month of parking at your lot.

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1_XO9hTA5ZZ4NmwwqvSHwJs_SjfbYQe6Q) to view this solution in an interactive Colab (Jupyter) notebook.

1 - Import the data

```
# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# code to connect colab to google drive (save the spreadsheet provided into your drive)
# https://colab.research.google.com/notebooks/io.ipynb#scrollTo=JiJVCmu3dhFa

!pip install --upgrade -q gspread
from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

# Open our new sheet and read in the data
worksheet = gc.open('Parking Spot Selection (interviewqs.com)').get_worksheet(0)

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

# Convert to a DataFrame and render.
parking_choices = pd.DataFrame.from_records(rows)

## Fixing header of dataframe
# rename header so it's the first row
parking_choices.columns = parking_choices.iloc[0]

#drop the first row after we rename the header
parking_choices = parking_choices.reindex(parking_choices.index.drop(0))
```

2 - Convert dataframe column to array, store frequency count of values

```
# grab the first column in our dataframe, save it as an array
parking_choices_array = parking_choices['Spot #'].values

## Create a count of values

# Create a dictionary to store our counts
count = {}

# For each value in the data
for j in parking_choices_array:
    # An a key, value pair, with the observation being the key
    # and the value being +1
    count[j] = count.get(j, 0) + 1
```

3 - Normalize the count to percentages

```
# Calculate the # of observations
n = len(parking_choices_array)

# Create another dictionary to store the probabilities
probability_mass_function = {}

# For each unique value,
for unique_value, count in count.items():
    # Normalize the count by dividing by the
    #length of data, add to the dictionary
    probability_mass_function[unique_value] = count / n
```

4 - Plot the Probability Mass Function

```
# Plot the probability mass function (PMF)

# Note we use a PMF here rather than a probability density function
# --since the potential outcomes are discrete (e.g. one cannot park in spot # 1.5)
plt.bar(list(probability_mass_function.keys()), probability_mass_function.values(), color='g')
plt.show()
```

Here we can see that spot 16 looks like it has the highest probability of being chosen at random by a customer. -->
