## Question
You are given a [dataset](https://u4221007.ct.sendgrid.net/wf/click?upn=c6wysRx7DxHxCGh5eakHL0PTD2m237pPW3WbnV0-2B-2FP-2FNQiZEk5-2BhcPcgF92WTu6nJRINcM-2FG9-2B9FB4gw8hyZ6-2FKvreX4LF732JXt-2FJB1a-2BVB5TQS3u-2B8tAet3BB-2Bvj21QqWoCwZ3qlTBpwp44pqM5w-3D-3D_8c6kLYfeKFgEvI6pydPvKPLI8aUPZyqwVjt32YrJUfiCo7Z0UcXSDRy-2BPOH5mC0wd5BjaOqz7NTucO6Lxa2dD6xkLlTP4NCjQ4A9wAPpwuQHVdFmL0hCtUBuFT-2B4zDZ2eKwoFMsTQ5VspX1T-2BrkeeowFYS-2Fl7g-2BCiGEWA-2BGSQAbw4rBJjFgOk9G2vDPAERKVn27lyKjEZ2gEQ5Z1lEfNmKvpLza-2FATpAzsZA06WFOX0-3D) with information around messages sent between users in a P2P messaging application. Below is the dataset's schema:

| Column Name | Data Type |                        Description                        |
|:-----------:|:---------:|:---------------------------------------------------------:|
|     date    |   string  | date of the message sent/received, format is 'YYYY-mm-dd' |
|  timestamp  |  integer  |   timestamp of the message sent/received, epoch seconds   |
|  sender_id  |  integer  |                  id of the message sender                 |
| received_id |  integer  |                 id of the message receiver                |

Given this, write code to find the fraction of messages that are sent between the same sender and receiver within five minutes (e.g. the fraction of messages that receive a response within 5 minutes).
