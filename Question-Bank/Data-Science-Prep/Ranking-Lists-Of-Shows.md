## Problem
This problem was asked by Netflix.

How would you design a metric to compare rankings of lists of shows for a given user?

## Solution
The metric should have two components of consumption: we want shows that a user is likely to play, and shows that a user is likely to enjoy. For the first component, we can use a show’s popularity, and for the second component, we can use the user’s predicted rating on that show. If we only optimize for the first one, we might miss out on shows that a user would enjoy that is not popular, and if we only optimize for the latter there might be shows that are too niche or unfamiliar.

Therefore the metric should be a linear combination of those two aspects, and the weights can be tuned by A/B testing or having a machine-learning algorithm to tune the weights, given enough data with positive and negative labels on the end outcome. Popularity can be assessed by different segments (country, genre, etc.) and over different time ranges, and ratings (along with other metadata from other user interactions) can be used.
