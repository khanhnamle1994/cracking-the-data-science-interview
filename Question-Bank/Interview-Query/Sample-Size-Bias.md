## Question
This question was asked by: Airbnb

Pretend you have to analyze the results of an AB test. One variant of the AB test has a sample size of 50K users and the other has a sample size of 200K users.

Given the unbalanced size between the two groups, can you determine if the test will result in bias towards the smaller group?

<!-- ## Solution
There's a couple ways to test for bias but let's look at the size of each of the populations again. The interviewer in this case is trying to assess how you would approach the problem given an unbalanced group. We're not given context to the situation so we have to either:

1. State assumptions or
2. Ask clarifying questions.

How long has the AB test been running? Have they been running during the same time duration? If the data was collected during different time periods then bias certainly exists from one group being from a different date period than the other.

Let's assume that there is no bias when having unequal sample sizes because the smaller sample size is already very large. For even very small effects, **50K observations may confer quite a powerful test.** The power of the test is heavily dependent on the smaller sample size. But since the sample size is large enough, power is not a concern here.

If the test is run inappropriately, in which case the pooled variance estimate is more heavily weighted toward the larger group, and the variances between the two samples are largely different compared to their means, then we might see bias in the result effects. Otherwise given than 50K already confers a powerful enough test, we might not see bias if we can downsample the other test variation to 50K. -->
