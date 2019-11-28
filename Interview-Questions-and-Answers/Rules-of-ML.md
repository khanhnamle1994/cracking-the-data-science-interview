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
A simple heuristic can get your product out the door. A complex heuristic is unmaintainable. Once you have data and a basic idea of what you are trying to accomplish, move on to machine learning. As in most software engineering tasks, you will want to be constantly updating your approach, whether it is a heuristic or a machine­-learned model, and you will find that the machine­-learned model is easier to update and maintain.

## ML Phase I: Your First Pipeline
Focus on your system infrastructure for your first pipeline. While it is fun to think about all the imaginative machine learning you are going to do, it will be hard to figure out what is happening if you don’t first trust your pipeline.

*Rule #4: Keep the first model simple and get the infrastructure right.*
The first model provides the biggest boost to your product, so it doesn't need to be fancy. But you will run into many more infrastructure issues than you expect. Before anyone can use your fancy new machine learning system, you have to determine:
* How to get examples to your learning algorithm.
* A first cut as to what "good" and "bad" mean to your system.
* How to integrate your model into your application. You can either apply the model live, or pre­compute the model on examples offline and store the results in a table. For example, you might want to pre­classify web pages and store the results in a table, but you might want to classify chat messages live.

Choosing simple features makes it easier to ensure that:
* The features reach your learning algorithm correctly.
* The model learns reasonable weights.
* The features reach your model in the server correctly.

Once you have a system that does these three things reliably, you have done most of the work. Your simple model provides you with baseline metrics and a baseline behavior that you can use to test more complex models. Some teams aim for a "neutral" first launch: a first launch that explicitly de­prioritizes machine learning gains, to avoid getting distracted.

*Rule #5: Test the infrastructure independently from the machine learning.*
Make sure that the infrastructure is testable, and that the learning parts of the system are encapsulated so that you can test everything around it. Specifically:
1. Test getting data into the algorithm. Check that feature columns that should be populated are populated. Where privacy permits, manually inspect the input to your training algorithm. If possible, check statistics in your pipeline in comparison to statistics for the same data processed elsewhere.
2. Test getting models out of the training algorithm. Make sure that the model in your training environment gives the same score as the model in your serving environment.

Machine learning has an element of unpredictability, so make sure that you have tests for the code for creating examples in training and serving, and that you can load and use a fixed model during serving. Also, it is important to understand your data.

*Rule #6: Be careful about dropped data when copying pipelines.*
Often we create a pipeline by copying an existing pipeline (i.e., cargo cult programming ), and the old pipeline drops data that we need for the new pipeline. For example, the pipeline for Google Plus What’s Hot drops older posts (because it is trying to rank fresh posts). This pipeline was copied to use for Google Plus Stream, where older posts are still meaningful, but the pipeline was still dropping old posts. Another common pattern is to only log data that was seen by the user. Thus, this data is useless if we want to model why a particular post was not seen by the user, because all the negative examples have been dropped. A similar issue occurred in Play. While working on Play Apps Home, a new pipeline was created that also contained examples from the landing page for Play Games without any feature to disambiguate where each example came from.

*Rule #7: Turn heuristics into features, or handle them externally.*
Usually the problems that machine learning is trying to solve are not completely new. There is an existing system for ranking, or classifying, or whatever problem you are trying to solve. This means that there are a bunch of rules and heuristics. These same heuristics can give you a lift when tweaked with machine learning. Your heuristics should be mined for whatever information they have, for two reasons. First, the transition to a machine learned system will be smoother. Second, usually those rules contain a lot of the intuition about the system you don’t want to throw away. There are four ways you can use an existing heuristic:
* Preprocess using the heuristic. If the feature is incredibly awesome, then this is an option. For example, if, in a spam filter, the sender has already been blacklisted, don’t try to relearn what "blacklisted" means. Block the message. This approach makes the most sense in binary classification tasks.
* Create a feature. Directly creating a feature from the heuristic is great. For example, if you use a heuristic to compute a relevance score for a query result, you can include the score as the value of a feature. Later on you may want to use machine learning techniques to massage the value (for example, converting the value into one of a finite set of discrete values, or combining it with other features) but start by using the raw value produced by the heuristic.
* Mine the raw inputs of the heuristic. If there is a heuristic for apps that combines the number of installs, the number of characters in the text, and the day of the week, then consider pulling these pieces apart, and feeding these inputs into the learning separately. Some techniques that apply to ensembles apply here.
* Modify the label. This is an option when you feel that the heuristic captures information not currently contained in the label. For example, if you are trying to maximize the number of downloads, but you also want quality content, then maybe the solution is to multiply the label by the average number of stars the app received. There is a lot of leeway here.

Do be mindful of the added complexity when using heuristics in an ML system. Using old heuristics in your new machine learning algorithm can help to create a smooth transition, but think about whether there is a simpler way to accomplish the same effect.
