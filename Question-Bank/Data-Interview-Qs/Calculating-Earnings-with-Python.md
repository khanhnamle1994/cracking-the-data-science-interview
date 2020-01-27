## Question

Suppose an individual is taxed 30% if earnings for a given week are > = $2,000. If earnings land < $2,000 for the week, the individual is taxed at a lower rate of 15%.

Write a function using Python to calculate both the pre-tax and post-tax earnings for a given individual, with the ability to feed in the hourly wage and the weekly hours as inputs.

For example, if an individual earns $55/hour and works for 40 hours, the function should return:

* Pre-tax earnings were 55*40 = $2,200 for the week.
* Post-tax earnings were $2,200*.7 (since we fall in higher tax bracket here) = $1,540 for the week

<!-- ## Solution:

[Click here](https://colab.research.google.com/drive/1ir-gaUMwuXGVnzia_qkEn_BiOuETtG_X#scrollTo=YkoCBwgrtZ-w) to view this solution in an interactive Colab (Jupyter) notebook.

We'll set up a function below to determine whether or not a given number is prime, using simple if/else statements. Additionally, when a number is defined as prime we'll append it to our array, a.

```
#Return the post-tax earnings of an individual, assuming varying tax brackets depending on annual wage

def calcEarnings(hours, wage):
    earnings = hours*wage
    if earnings >= 2000:
        #30% tax rate for weekly earnings above $2000
        earnings = earnings*.70
    else:
        #15% tax rate for weekly earnings below $2000
        earnings = earnings*.85
    return earnings

#return the raw earnings of an individual (assuming no varying rates for overtime)
def calcEarnings_pretax(hours, wage):
    earnings = hours*wage
    return earnings


def main():
    hours = float(input('Enter hours worked for the week: '))
    wage = float(input('Enter dollars paid per hour: '))
    total = calcEarnings(hours, wage)
    total_pre_tax = calcEarnings_pretax(hours, wage)
    taxes = total_pre_tax - total
    print('Pre-tax earnings for {hours} hours at ${wage:.2f} per hour are ${total_pre_tax:.2f}.'
          .format(**locals()))
    print('Post-tax earnings for {hours} hours at ${wage:.2f} per hour are ${total:.2f}.'
          .format(**locals()))
    print('You gave uncle sam ${taxes:.2f} this week!'
          .format(**locals()))

main()
``` -->
