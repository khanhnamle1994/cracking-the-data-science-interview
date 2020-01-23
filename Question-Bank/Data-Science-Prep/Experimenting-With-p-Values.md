## Problem
This problem was asked by Airbnb.

How can you decide how long to run an experiment? What are some problems with just using a fixed p-value threshold and how do you work around them?

## Solution
The most basic way would be to use a fixed p-value threshold and run the experiment until that threshold is met or not, but that assumes you have a particular effect size. If you monitor a fixed p-value over time, you may stop too early (assuming significance when it is just noise) or not stop (not finding an effect when there should be one). This can happen for a variety of reasons - the size of the sample, unequal effects over time (users booking at different times throughout the experiment, due to seasonality or adoption differences), etc.

To address this, it is possible to determine a “dynamic” p-value using sequential analysis, which is a type of statistical analysis whereby the p-value is not fixed in advance. In practice, this can be done by running different simulations whereby the parameters of the experiment are varied and measuring various important evaluation metrics such as false positives and false negatives.
