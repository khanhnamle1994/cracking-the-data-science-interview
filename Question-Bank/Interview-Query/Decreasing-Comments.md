## Question
This question was asked by: Pinterest

Let's say you work for a social media company that has just done a launch in a new city. Looking at weekly metrics, you see a slow decrease in the average number of comments per user from January to March in this city.

The company has been consistently growing new users in the city from January to March.

What are some reasons on why the average number of comments per user would be decreasing and what metrics would you look into?

<!-- ## Solution
Let's take an approach of investigating into a couple of different metrics. Many candidates like to randomly shoot in the dark and think about external factors. (Maybe there's a winter storm in Chicago and no one has power). Almost 100% of the time the answer is not related to external factors.

Average comments per user is calculated by taking the total number of comments divided by the total number of users. So let's model out an example scenario.
* Jan: 10000 users, 30000 comments, 3 comments/user
* Feb: 20000 users, 50000 comments, 2.5 comments/user
* Mar: 30000 users, 60000 comments, 2 comments/user

We're given information that total user count is increasing linearly which means that the decreasing comments/user is not an effect of a declining user base creating a loss of network effects on the platform.

With this we can hypothesis a couple of answers:

**1. Could more users cause less individual engagement?**

Otherwise known as the crowding effect, we can model this by looking at a cohort of users that started in the first week of January and then another cohort of users that started the first week of March. If we see no difference between the number of comments after the first, second, third weeks, then there likely is no crowding effect happening.

**2. What about active user engagement?**

We know that even as the company is increasing users, it's likely users will fall off and churn off the platform. Let's say we model the user churn by month.
* Month 1: 25% Churn
* Month 2: 20% Churn
* Month 3: 15% Churn

This means that for the cohort of users that starts in January, by February there are now only 7500 (10000 * 75%) active users, then in March 6000 active users from a 20% churn.

This can explain a likely effect of why we see a decrease of comments per user even though the total user count is increasing linearly. Churn decreases the active user counts which is assumed to be directly correlated to how many comments will exist on a platform. Because **we have less active users on the platform, the denominator is in this case a fake proxy for actual platform engagement.**

If we wanted to measure if active users will still commenting, we could then just look at **comments per active user**. -->
