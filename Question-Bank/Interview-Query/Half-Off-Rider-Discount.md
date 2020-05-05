## Problem
You work as a data scientist for Lyft. A VP asks how you would evaluate whether a 50% rider discount promotion is a good or bad idea? How would you implement it? What metrics would you track?

<!-- ## Solution
This is a question with many paths and no correct right answer. Rather the interviewer is looking towards how you can explain your reasoning when facing a practical product question. Let's evaluate the proposal and make sure to state assumptions first. What questions would need to be brought up to make sure you and the interviewer were on the same page? Many times interviewers will have a personal clear idea of how to solve a problem and it requires probing and questions to figure out how they're thinking about the situation.

**Assumptions**
Let's state some assumptions and go down a couple paths towards solving them. The first assumption is that the goal of the discount is greater **retention and revenue**. Let's also assume that this discount is going to be applied uniformly across all users of time length on the platform (i.e. not targeting new users). Lastly let's assume that the 50% discount is applied on only one ride.

Now that we've stated our assumptions, let's figure out how we can evaluate the change. Since the question prompts a feature change in the form of pricing, this means we can propose an **AB test and set metrics for evaluation of the test**. Our AB test would be designed with a control and a test group, with only new users getting bucketed into either variant. The test group would receive a 50% rider discount on one ride and the control group would not.

With this is mind we can take two measurements into account for measuring the unit economics of the 50% off promotion: **Long term revenue vs average cost of the promotion**. If the average long-term revenue of users given the promotion subtracted by the cost of the discount is greater than the average long-term revenue of users **not given the promotion** then the promotion is profitable for the company.

Long term revenue on a per user basis is the single most important key driver of profitability. Given that we are running an AB test, that sets us up with constraints in measuring feedback with a fixed amount of time before we can get results back. If we set a fixed amount of time such as 10, 20, or 30 days, that gives us a nominal amount of time to evaluate LTV (life-time valuation) on a large user sample size.

Now we have to measure the average cost. While we can take the average cost of all rides across the U.S. on Lyft and take 50% of it for the revenue loss, we may be biasing ourself. Riders may be inclined to use more expensive rides for the 50% off ride. We also have to set a ride cap dependent on average cost of ride per city. The same distance of a ride in San Francisco may cost much less in Kansas City.

**Further Analysis**
One problem we may run into is that 30 days might not be enough time for us to run an AB test and compare average revenue numbers between the two groups. Given this problem, we would have to build user lifetime revenue models to extrapolate past the user's first 30 days. One example is that if users ends up requesting on average 5 rides after 30 days and 8 rides on average after 60 days, we can extrapolate that same effect from the group that receives the 50% off promotion.

If we were further this analysis we could also analyze changing the test to look at the frequency of the rider promotions. What if we tested one 50% off ride versus multiple 50% off rides? These tests could tell us the difference between product stickiness given some X number of promotions. If we can increase retention by 10% by giving two 50% off discounts versus one, we can then calculate the LTV increase over time.

Further considerations would be on user segmentation and scale. At Lyft's scale, we have to make assumption of if the launch is in new cities. Some questions we can dig into are:
- Is there a existing ride-sharing market already in place?
- Is this discount specific to a singular city?
- Can the supply side marketplace of drivers limited if we run this test in a place without too many drivers? -->
