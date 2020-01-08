## Problem
This problem was asked by Facebook.

Imagine the social graphs for both Facebook and Twitter. How do they differ? What metric would you use to measure how skewed the social graphs are?

## Solution
The main difference is that Facebook is composed of friendships, in which two users are mutual friends, whereas Twitter is composed of followership, in which one user follows another (usually influential figure). This leads to Twitter likely having a small number of people with very large followership, whereas in Facebook that is less often the case (besides the fact that the number of friends on a personal profile is capped).

One way to measure the skewness of the social graphs is to have each graph as a node, and look at the degrees of the nodes. The degree of a node is simply the number of connections it has to other nodes. It is likely that for Twitter, you will see more right skewness, i.e. most nodes have a low degree but a small number of nodes have a very high degree - like a “hub-and-spoke” model.
