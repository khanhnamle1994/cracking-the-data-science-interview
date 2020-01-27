## Question
Given the data table below, determine if there is a relationship between fitness level and smoking habits:

|                         | Low Fitness Level | Medium-Low Fitness Level | Medium-High Fitness Level | High Fitness Level |
|:-----------------------:|:-----------------:|:------------------------:|:-------------------------:|:------------------:|
|       Never Smoked      |        113        |            113           |            110            |         159        |
|      Former Smokers     |        119        |            135           |            172            |         190        |
| 1 to 9 Cigarettes Daily |         77        |            91            |             86            |         65         |
|  >=10 Cigarettes Daily  |        181        |            152           |            124            |         73         |

You don't have to fully solve for the number here (that would be pretty time-intensive for an interview setting), but lay out the steps you would take to solve such a problem.

<!-- ## Solution
Here, we can set up a [Chi-square test](https://en.wikipedia.org/wiki/Chi-squared_test):
* Null hypothesis H0: fitness level and smoking habits are independent
* Alternative hypothesis Ha: fitness level and smoking habits are independent

First, we can calculate the expected counts using the following formula:
* E = (row total * column total)/grand total

Next, we can apply this to each field in the table (for example, 'Never smoked' and 'Low' fitness level would be: (495*490)/1960 = 123.75

|                         | Low Fitness Level | Medium-Low Fitness Level | Medium-High Fitness Level | High Fitness Level |
|:-----------------------:|:-----------------:|:------------------------:|:-------------------------:|:------------------:|
|       Never Smoked      |        123.8      |            124.0         |            124.3          |         123.0      |
|      Former Smokers     |        154.0      |            154.3         |            154.6          |         153.1      |
| 1 to 9 Cigarettes Daily |         79.8      |            79.9          |             80.1          |         79.3       |
|  >=10 Cigarettes Daily  |        132.5      |            132.8         |            133.0          |         131.7      |

Next, we can solve for X^2 using the following formula:
* X^2 = (observation - estimate)^2 / estimate (where estimate was derived using the formula above for each cell in the table)

Solving this, we arrive at X^2 = 91.73, where we can then use the table of X^2 critical values to find that the P-value is less than 0.001 meaning we can reject our null hypothesis and conclude that there is a relationship between fitness level and smoking habits. -->
