## Question
Suppose you're working for a social media company that has a feature similar to Facebook's newsfeed. The company has two options for serving ads within their newsfeed:

1. Out of every 25 stories, one will be an ad
2. Every story has a 4% chance of being an ad

For each option, what is the expected number of ads shown in 100 news stories? If we go with the second option, what is the chance a user will be shown only a single ad in 100 stories? What about no ads at all?

<!-- ## Solution

**Option 1, expected number of ads**

Expected number of ads in 100 stories = (100 stories) / (25 stories / 1 ad) = 4 ads per 100 stories

**Option 2, expected number of ads**

Expected number of ads in 100 stories = (100 stories) * (4% chance for each story to be an ad) = 4 ads per 100 stories

**Option 2, P(1 ad in 100 stories)**

For this question, we're going to use the [binominal distribution function](https://en.wikipedia.org/wiki/Binomial_distribution).

Pr(1 ad in 100 stories with each story having a 4% chance of being an ad)

Variables given: k = 1, n = 100, p = 0.04

```
Pr(k; n, p) = Pr(0.04; 100, 1)
= (n choose k) * (p)^(k) * (1 - p)^(n - k)
= (100 choose 1) * (0.04)^(1) * (1 - 0.04)^(100 - 1)
= 0.0703
```

There's about a 7.03% chance that there will be 1 ad shown

**Option 2, P(0 ad in 100 stories)**

Again, we're going to use the [binominal distribution function](https://en.wikipedia.org/wiki/Binomial_distribution).

Pr(0 ad in 100 stories with each story having a 4% chance of being an ad)

Variables given: k = 0, n = 100, p = 0.04

```
Pr(k; n, p) = Pr(0.04; 100, 0)
= (n choose k) * (p)^(k) * (1 - p)^(n - k)
= (100 choose 0) * (0.04)^(0) * (1 - 0.04)^(100 - 0)
= 0.01687
```

There's about a 16.87% chance that there will be 0 ads shown -->
