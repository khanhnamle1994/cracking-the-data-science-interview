## Question
Suppose you want to generate a sample of heads and tails from a fair coin. However, you only have a biased coin available (meaning the probability of coming up with heads is different than 1/2).

Write a simulation in python that will use your biased coin to generate a sample of heads/tails as if it were coming from a fair coin.

<!-- ## Solution
[Click here](https://colab.research.google.com/drive/1xUDx7GqbzvJMjG9Wp-Dt8Fp1isdIH7yZ) to view this solution in an interactive Colab (Jupyter) notebook.

```
from random import random
# first, we can set up a coin with an arbitrary bias, 0.3 in this case.
# here the coin will return 1 with probability 0.3
def biasedCoin():
   return int(random() < 0.3)
# for example, if we try the following
# we'll get a number that is around 30,000 (0.3 *100k)
sum(biasedCoin() for i in range(100000));
```

Here, we can see the output is approximately 30k:
30191

Now, we can set up our function to turn the biased coin into a fair coin toss. If we have a biased coin, we can simulate a fair coin by tossing pairs of the biased coins until the two results are different. Given that we have different results, the probability that the first is “heads” and the second is “tails” is the same as the probability of “tails” then “heads”. So if we simply return the value of the first coin, we will get “heads” or “tails” with the same probability, i.e. 1/2.

```
def fairCoin(biasedCoin):
   coin1, coin2 = 0,0
   while coin1 == coin2:
      coin1, coin2 = biasedCoin(), biasedCoin()
   # Once the coin values are different, return the value of the first coin
   return coin1

# run simulation of 100k tosses, to check if
# we're approximately at a 50% 'fair' distribution
sum(fairCoin(biasedCoin) for i in range(100000))
```

Here, we can see the output is approximately 50%: 49,910/100,000 -->
