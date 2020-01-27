## Question
You are given a [dataset](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHL0PTD2m237pPW3WbnV0-2B-2FP-2FNQiZEk5-2BhcPcgF92WTu6nJRINcM-2FG9-2B9FB4gw8hyZ6-2FKvreX4LF732JXt-2FJB1a-2BVB5TQS3u-2B8tAet3BB-2Bvj21QqWoCwZ3qlTBpwp44pqM5w-3D-3D_8c6kLYfeKFgEvI6pydPvKPLI8aUPZyqwVjt32YrJUfiCo7Z0UcXSDRy-2BPOH5mC0wd5BjaOqz7NTucO6Lxa2dD6xkLlTP4NCjQ4A9wAPpwuQHVdFmL0hCtUBuFT-2B4zDZ2eKwoFMsTQ5VspX1T-2BrkeeowFYS-2Fl7g-2BCiGEWA-2BGSQAbw4rBJjFgOk9G2vDPAERKVn27lyKjEZ2gEQ5Z1lEfNmKvpLza-2FATpAzsZA06WFOX0-3D) with information around messages sent between users in a P2P messaging application. Below is the dataset's schema:

| Column Name | Data Type |                        Description                        |
|:-----------:|:---------:|:---------------------------------------------------------:|
|     date    |   string  | date of the message sent/received, format is 'YYYY-mm-dd' |
|  timestamp  |  integer  |   timestamp of the message sent/received, epoch seconds   |
|  sender_id  |  integer  |                  id of the message sender                 |
| received_id |  integer  |                 id of the message receiver                |

Given this, write code to find the fraction of messages that are sent between the same sender and receiver within five minutes (e.g. the fraction of messages that receive a response within 5 minutes).

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1ioUZSdjhDu-JqFFgLhxgAlkXeNZARWB4#scrollTo=YXNp3ykBU85t) to view this solution in an interactive Colab (Jupyter) notebook.

```
#Importing packages.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

# Code to read csv file into Colaboratory:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

!pip install --upgrade -q gspread
from google.colab import auth
auth.authenticate_user()

import gspread
from oauth2client.client import GoogleCredentials

gc = gspread.authorize(GoogleCredentials.get_application_default())

# Open our new sheet and read some data.
worksheet = gc.open('Sample Message Dataset').get_worksheet(0)

# get_all_values gives a list of rows.
rows = worksheet.get_all_values()

# Convert to a DataFrame and render.
message_df = pd.DataFrame.from_records(rows)

### Fixing header of dataframe
#rename header so it's the first row
message_df.columns = message_df.iloc[0]
#drop the first row after we rename the header
message_df = message_df.reindex(message_df.index.drop(0))

#timestamp
message_df['timestamp'] = pd.to_numeric(message_df.timestamp, downcast='integer')
```

To calculate response time we need to think through a couple of things:

1. a definition for response
2. a way to compare the original message to the response

We first need to figure out how to identify the "response". Let's create a primary key between each unique sender_id to receiver_id and receiver_id to sender_id.

We first need to figure out how to identify the "response". Let's create a primary key between each unique sender_id to receiver_id and receiver_id to sender_id.

```
# creating the primary key
#primary key is comparing the sender_id to the receiver_id,
# which ever value is the lowest will come first
# then we concatenate the two id's to create a unique key
message_df['key'] = np.where(message_df['sender_id']>=message_df['receiver_id'], \
                             message_df['receiver_id']+message_df['sender_id'],
                              message_df['sender_id']+message_df['receiver_id'])
```

```
#sorting the dataframe based on primary key
message_df = message_df.sort_values(by=['key', 'timestamp' ])
```

Next we need to figure out how to compare timestamps so we can figure out if a message was responded within 5 minutes. For example, from the data above, I would want to compare 1519920555 to 1519920410. We can do a simple lag function but there are a few conditions that we need to outline to make this work.

1. We only want to compare timestamps within the same key
2. We want to compare if we have a response e.g. 1st message is from a to b and second message is from b to a
3. We only want to compare if the receiver_id != sender_id (e.g. someone sending messages to themselves)

```
# CONDITION 1
# lagging the key so we can see if we should lag the timestamp
message_df['key_lag'] = message_df['key'].shift(1)
# only returning the lagged timestamp if it's within the same key
message_df['timestamp_lag'] = np.where( message_df['key_lag']==message_df['key'],message_df['timestamp'].shift(1),0)


# CONDITION 2
#lagging the sender_id to compare if it's a response
message_df['sender_id_lag'] = message_df['sender_id'].shift(1)
# the 2nd condition is covering for the fact we don't have an sender_id_lag
message_df['is_response'] = np.where(( message_df['receiver_id']==message_df['sender_id_lag']) ,"yes","no")

# CONDITION 3
# filter for messages sent and received by same id
message_df['same_sender_receiver'] = np.where( message_df['sender_id'] == message_df['receiver_id'] ,"yes","no")
#filtering out the unwanted conditions
message_df_filtered = message_df[(message_df.same_sender_receiver == 'no') & (message_df.is_response == 'yes') ]

#now we can compare the two timestamps calculating the delta between sent and received
message_df_filtered['time_delta'] = round((message_df_filtered['timestamp'] - message_df_filtered['timestamp_lag']) / 60,1)


#creating new DF that filters all messages < 5mins
messages_less_5 = message_df_filtered[(message_df_filtered['time_delta'])<= 5]
```

```
#now we can calculate the result!
all_messages = len(message_df)
five_mins_response_messages = len(messages_less_5)

round(five_mins_response_messages/all_messages,2)
```

**21%** of messages get a response after 5 minutes! -->
