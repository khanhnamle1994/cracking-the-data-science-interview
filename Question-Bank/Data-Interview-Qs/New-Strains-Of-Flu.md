## Problem
You go to see the doctor about a cough you've had for a while. The doctor selects you at random to have a blood test for a new strain of flu, which for the purposes of this exercise we will say is currently suspected to affect 1 in 10,000 people in the US. The test is 99% accurate, in the sense that the probability of a false positive is 1%.

The probability of a false negative is zero. You test positive. What is the new probability that you have this strain of flu?

<!-- ## Solution
Let P(D) be the probability you have a new strain of flu and P(T) be the probability of a positive test. We want to know P(D | T), which is the probability that you have the flu, given that we know you tested positive. We can apply Bayes' Theorem to solve for P(D | T).

```
P(D|T) = (P(T|D) * P(D)) / P(T)
```

We can rewrite the above to:

```
P(D|T) = (P(T|D) * P(D)) / (P(T|D) * P(D) + P(T|ND) * P(ND))
```

Now lets list out what we know:
* P(D) = 0.0001 (probability you have the flu)
* P(ND) = 0.9999
* P(T|D) = 1 (if you have flu the test is always positive)
* P(T|ND) = 0.01 (1% chance of a false positive)

We can plug these number in and get... P(D|T) = 0.01 -->
