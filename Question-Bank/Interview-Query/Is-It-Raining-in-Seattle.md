## Question
This question was asked by: Facebook

You are about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it's raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that "Yes" it is raining.

What is the probability that it's actually raining in Seattle?

<!-- ## Solution
This question can be solved in two ways in the schools of thought: **Bayesian** or **Frequentist**. Let's do a thought experiment and try both.

### Frequentist
For example. The question prompt states, that each friend has a 2/3 change of telling the truth. Through logical transference, given that all of the friends have told you that it is raining, the question of "what is the probability that it is not raining" is the same thing as "what is the probability that all of your friends are lying?"

```
P(Not Raining) = P(Friend 1 Lying) AND P(Friend 2 Lying) AND P(Friend 3 Lying)
```

Given this logical expression. We can simply the problem to then to calculate the inverse of three AND functions. So the probability of it raining is then equated to:

```
P(Raining) = 1 - P(3 Friend's Lying)
```

Multiple of all independent probabilities:

```
P(3 Friend's Lying) = 1/3 * 1/3 * 1/3 = 1/27
P(Raining) = 1 - 1/27 = 26/27
```

### Bayesian
What if we want to take into the account the probability of rain in Seattle? For instance, if the probability of rain in Seattle is 100%, then I might still ask my friends if it's raining, and they might all lie to me, but in either case it's irrelevant.

What if it's a bit of a mistake to think about the probability your friends are lying specifically? Suppose we imagine the probability of rain is 50% in general. Call rain event A and event B, C, and D can be events where your friends tell you it is raining. This is a Bayesian question fundamentally, and the formula applies:

```
P(A | B & C & D) = P(B & C & D | A) * P(A) / P(B & C & D)
```

* We know P(A) = 0.5.
* P(B & C & D | A) is the probability of all my friends saying its raining given that it is raining, which is the same as the probability of them all telling the truth, which is (2/3)^3 = 8/27 assuming they tell the truth/lie independently.
* P(B & C & D) is the probability of all my friends telling me its raining which is P(B & C & D | A) * P(A) + P(B & C & D | ¬A) * P(¬A) because of the law of total probability, where ¬A  means not raining. Basically we have added the probability of all my friends telling the truth to the probability they are all lying.

So we get

```
P(B & C & D)  =  8/27 * 1/2 + 1/27 * 1/2 = 8/54 + 1/54 = 9 / 54 = 1/6
```

So
```
P(A | B & C & D) = (8/27 * 1/2) / (1/6) = 4/27 * 6 = 24/27 = 8/9
```

Note how B, C, and D are defined in relation to whether your friends tell you it rains, not whether they are lying or not. Because we actually don't know whether they were lying, we just know the answer they gave was the event.

### Intuition
Intuitively, if you might wonder why the answer 8/9 is quite a bit lower than 26/27. Imagine we repeated this task 27 times, we would expect that there was only one time when they all three lied out of 27 trials. But in our situation, we don't want to look at all 27 experiments, because we know that our friends gave the same answer, which only happens when they're all honest or when they all lie. They are all honest with (2/3)^3 = 8/27 probability, so in 8 of the experiments. That leaves us with 9 experiments when they all gave the same answer.

So 8/9 times when they all give the same answer, they're honest. -->
