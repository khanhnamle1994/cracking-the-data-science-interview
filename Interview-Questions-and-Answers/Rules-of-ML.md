The content originally comes from: https://developers.google.com/machine-learning/guides/rules-of-ml

# Best Practices for ML Engineering
This document is intended to help those with a basic knowledge of machine learning get the benefit of Google's best practices in machine learning. It presents a style for machine learning, similar to the Google C++ Style Guide and other popular guides to practical programming. If you have taken a class in machine learning, or built or worked on a machine­-learned model, then you have the necessary background to read this document.

## Terminology
The following terms will come up repeatedly in our discussion of effective machine learning:
* Instance: The thing about which you want to make a prediction. For example, the instance might be a web page that you want to classify as either "about cats" or "not about cats".
* Label: An answer for a prediction task ­­ either the answer produced by a machine learning system, or the right answer supplied in training data. For example, the label for a web page might be "about cats".
* Feature: A property of an instance used in a prediction task. For example, a web page might have a feature "contains the word 'cat'".
* Feature Column: A set of related features, such as the set of all possible countries in which users might live. An example may have one or more features present in a feature column. "Feature column" is Google-specific terminology. A feature column is referred to as a "namespace" in the VW system (at Yahoo/Microsoft), or a field.
* Example: An instance (with its features) and a label.
* Model: A statistical representation of a prediction task. You train a model on examples then use the model to make predictions.
* Metric: A number that you care about. May or may not be directly optimized.
* Objective: A metric that your algorithm is trying to optimize.
* Pipeline: The infrastructure surrounding a machine learning algorithm. Includes gathering the data from the front end, putting it into training data files, training one or more models, and exporting the models to production.
* Click-through Rate The percentage of visitors to a web page who click a link in an ad.

## Overview
To make great products: **do machine learning like the great engineer you are, not like the great machine learning expert you aren’t**.

Most of the problems you will face are, in fact, engineering problems. Even with all the resources of a great machine learning expert, most of the gains come from great features, not great machine learning algorithms. So, the basic approach is:
1. Make sure your pipeline is solid end to end.
2. Start with a reasonable objective.
3. Add common­-sense features in a simple way.
4. Make sure that your pipeline stays solid.

This approach will work well for a long period of time. Diverge from this approach only when there are no more simple tricks to get you any farther. Adding complexity slows future releases.

## Before Machine Learning
*Rule #1: Don’t be afraid to launch a product without machine learning.*
Machine learning is cool, but it requires data. Theoretically, you can take data from a different problem and then tweak the model for a new product, but this will likely underperform basic heuristics. If you think that machine learning will give you a 100% boost, then a heuristic will get you 50% of the way there.

For instance, if you are ranking apps in an app marketplace, you could use the install rate or number of installs as heuristics. If you are detecting spam, filter out publishers that have sent spam before. Don’t be afraid to use human editing either. If you need to rank contacts, rank the most recently used highest (or even rank alphabetically). If machine learning is not absolutely required for your product, don't use it until you have data.

*Rule #2: First, design and implement metrics.*
Before formalizing what your machine learning system will do, track as much as possible in your current system. Do this for the following reasons:
1. It is easier to gain permission from the system’s users earlier on.
2. If you think that something might be a concern in the future, it is better to get historical data now.
3. If you design your system with metric instrumentation in mind, things will go better for you in the future. Specifically, you don’t want to find yourself grepping for strings in logs to instrument your metrics!
4. You will notice what things change and what stays the same. For instance, suppose you want to directly optimize one­-day active users. However, during your early manipulations of the system, you may notice that dramatic alterations of the user experience don’t noticeably change this metric.

Google Plus team measures expands per read, reshares per read, plus­ones per read, comments/read, comments per user, reshares per user, etc. which they use in computing the goodness of a post at serving time. Also, note that an experiment framework, in which you can group users into buckets and aggregate statistics by experiment, is important.

By being more liberal about gathering metrics, you can gain a broader picture of your system. Notice a problem? Add a metric to track it! Excited about some quantitative change on the last release? Add a metric to track it!

*Rule #3: Choose machine learning over a complex heuristic.*
A simple heuristic can get your product out the door. A complex heuristic is unmaintainable. Once you have data and a basic idea of what you are trying to accomplish, move on to machine learning. As in most software engineering tasks, you will want to be constantly updating your approach, whether it is a heuristic or a machine­-learned model, and you will find that the machine­-learned model is easier to update and maintain
