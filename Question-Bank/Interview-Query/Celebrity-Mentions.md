## Question
This question was asked by: Facebook

Let's say you're a product data scientist at Facebook. Facebook is rolling out a new feature called "Mentions" which is an app specifically for celebrities on Facebook to connect with their fans.

How would you measure the health of the Mentions app? And if a celebrity starts using Mentions and begins interacting with their fans more, what part of the increase can be attributed to a celebrity using Mentions versus what part is just a celebrity wanting to get more involved in fan engagement?

<!-- ## Solution
Let's first break down some structure on what the interviewer is looking for. Whenever we're given these open-ended product questions, it makes sense to think about structuring the questions with well-defined objectives so we're not switching between different answers.

*1. Did you begin by stating what the goals of the feature are before jumping into defining metrics? What is the point of the Mentions feature?*

*2. Are your answers structured or do you tend to talk about random points?*

*3. Are the metrics definitions specific or are they generalized in an example like “I would find out if people used Mentions frequently.”*

Let's start with the first question. We shouldn't be jumping into metrics but thoughtfully thinking about what metrics matter when analyzing the goal of the new feature rollout by Facebook. Let's think about what happens when a celebrity starts using Mentions.

Does it drive increases in engagement in celebrities and regular Facebook users? The Mentions app is essentially a way for users to connect to their favorite celebrities easier which describes a two-way marketplace effect. There needs to be enough celebrities using mentions to engage users and enough users for a celebrity to stay engaged.

One way we can test this is by **exposing mentions posts/stories/Q&As to a portion of the users in the top of their newsfeed. Do these users stay engaged on the platform longer?**

We have identified engagement as the thing we want to track. So now we can implement some more defined metrics. We can measure retention in a form of daily active users or weekly active users and compare this metric against the users that do not get the "Mentions" test. Does the weekly active user count go down or up in the test compared to the control?

Now let's look on the celebrity side. Another thing to measure is if the Mentions feature is taking away from any other part of the Facebook app. **Is engagement down on other features with Facebook when celebrities are using Mentions?** If we can cohort by overall usage of Mentions as a percentage of total usage on Facebook, we can analyze if celebrities using Mentions increases the total engagement of the celebrity on Facebook.

Again, we can define our metric specifically. In this case we can **cohort celebrities into different buckets depending on how much they use Mentions** and then look at their average first month retention on the platform, second month retention, etc...

**Example**:

% of FB usage as Mentions = <10%
1st month churn = 20%
2nd month churn = 10%

% of FB usage as Mentions = 10 to 20%
1st month churn = 15%
2nd month churn = 7%

% of FB usage as Mentions = 20 to 30%
....

The churn rate is measuring retention of celebrities split up into **cohorts by the percentage of time that they spend on the mentions app**. If we model general retention curves, we'll see a natural decrease to a plateau after X number of months calculated by: (active celebrities in month X) / (total celebrities that signed up in month 1).

If you imagine each cohort as a percentage of time that user spends on the mention app as <10%, 10%-20%, 20%-30%, etc.. and then model the retention curves for each cohort on a graph over time with the X axis as the number of months since they joined Facebook and the Y axis as the % of active celebrities/total celebrities, then if we see a slower and slower drop in retention rate, we can attribute the increase in celebrity mentions usage as an increase in retention. -->
