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

### Monitoring
In general, practice good alerting hygiene, such as making alerts actionable and having a dashboard page.

*Rule #8: Know the freshness requirements of your system.*
How much does performance degrade if you have a model that is a day old? A week old? A quarter old? This information can help you to understand the priorities of your monitoring. If you lose significant product quality if the model is not updated for a day, it makes sense to have an engineer watching it continuously. Most ad serving systems have new advertisements to handle every day, and must update daily. For instance, if the ML model for Google Play Search is not updated, it can have a negative impact in under a month. Some models for What’s Hot in Google Plus have no post identifier in their model so they can export these models infrequently. Other models that have post identifiers are updated much more frequently. Also notice that freshness can change over time, especially when feature columns are added or removed from your model.

*Rule #9: Detect problems before exporting models.*
Many machine learning systems have a stage where you export the model to serving. If there is an issue with an exported model, it is a user­-facing issue.

Do sanity checks right before you export the model. Specifically, make sure that the model’s performance is reasonable on held out data. Or, if you have lingering concerns with the data, don’t export a model. Many teams continuously deploying models check the area under the ROC curve (or AUC) before exporting. Issues about models that haven’t been exported require an e­mail alert, but issues on a user-facing model may require a page. So better to wait and be sure before impacting users.

*Rule #10: Watch for silent failures.*
This is a problem that occurs more for machine learning systems than for other kinds of systems. Suppose that a particular table that is being joined is no longer being updated. The machine learning system will adjust, and behavior will continue to be reasonably good, decaying gradually. Sometimes you find tables that are months out of date, and a simple refresh improves performance more than any other launch that quarter! The coverage of a feature may change due to implementation changes: for example a feature column could be populated in 90% of the examples, and suddenly drop to 60% of the examples. Play once had a table that was stale for 6 months, and refreshing the table alone gave a boost of 2% in install rate. If you track statistics of the data, as well as manually inspect the data on occasion, you can reduce these kinds of failures.

*Rule #11: Give feature columns owners and documentation.*
If the system is large, and there are many feature columns, know who created or is maintaining each feature column. If you find that the person who understands a feature column is leaving, make sure that someone has the information. Although many feature columns have descriptive names, it's good to have a more detailed description of what the feature is, where it came from, and how it is expected to help.

### Your First Objective
You have many metrics, or measurements about the system that you care about, but your machine learning algorithm will often require a single objective, a number that your algorithm is "trying" to optimize. I distinguish here between objectives and metrics: a metric is any number that your system reports, which may or may not be important.

*Rule #12: Don’t overthink which objective you choose to directly optimize.*
You want to make money, make your users happy, and make the world a better place. There are tons of metrics that you care about, and you should measure them all. However, early in the machine learning process, you will notice them all going up, even those that you do not directly optimize. For instance, suppose you care about number of clicks and time spent on the site. If you optimize for number of clicks, you are likely to see the time spent increase.

So, keep it simple and don’t think too hard about balancing different metrics when you can still easily increase all the metrics. Don’t take this rule too far though: do not confuse your objective with the ultimate health of the system. And, if you find yourself increasing the directly optimized metric, but deciding not to launch, some objective revision may be required.

*Rule #13: Choose a simple, observable and attributable metric for your first objective.*
Often you don't know what the true objective is. You think you do but then as you stare at the data and side-by-side analysis of your old system and new ML system, you realize you want to tweak the objective. Further, different team members often can't agree on the true objective. The ML objective should be something that is easy to measure and is a proxy for the "true" objective. In fact, there is often no "true" objective. So train on the simple ML objective, and consider having a "policy layer" on top that allows you to add additional logic (hopefully very simple logic) to do the final ranking.

The easiest thing to model is a user behavior that is directly observed and attributable to an action of the system:
* Was this ranked link clicked?
* Was this ranked object downloaded?
* Was this ranked object forwarded/replied to/e­mailed?
* Was this ranked object rated?
* Was this shown object marked as spam/pornography/offensive?

Avoid modeling indirect effects at first:
* Did the user visit the next day?
* How long did the user visit the site?
* What were the daily active users?

Indirect effects make great metrics, and can be used during A/B testing and during launch decisions.

Finally, don’t try to get the machine learning to figure out:
* Is the user happy using the product?
* Is the user satisfied with the experience?
* Is the product improving the user’s overall well­being?
* How will this affect the company’s overall health?

These are all important, but also incredibly hard to measure. Instead, use proxies: if the user is happy, they will stay on the site longer. If the user is satisfied, they will visit again tomorrow. Insofar as well-being and company health is concerned, human judgement is required to connect any machine learned objective to the nature of the product you are selling and your business plan.

*Rule #14: Starting with an interpretable model makes debugging easier.*
Linear regression, logistic regression, and Poisson regression are directly motivated by a probabilistic model. Each prediction is interpretable as a probability or an expected value. This makes them easier to debug than models that use objectives (zero­-one loss, various hinge losses, and so on) that try to directly optimize classification accuracy or ranking performance. For example, if probabilities in training deviate from probabilities predicted in side­-by-­sides or by inspecting the production system, this deviation could reveal a problem.

For example, in linear, logistic, or Poisson regression, there are subsets of the data where the average predicted expectation equals the average label (1- moment calibrated, or just calibrated). This is true assuming that you have no regularization and that your algorithm has converged, and it is approximately true in general. If you have a feature which is either 1 or 0 for each example, then the set of 3 examples where that feature is 1 is calibrated. Also, if you have a feature that is 1 for every example, then the set of all examples is calibrated.

With simple models, it is easier to deal with feedback loops. Often, we use these probabilistic predictions to make a decision: e.g. rank posts in decreasing expected value (i.e. probability of click/download/etc.). However, remember when it comes time to choose which model to use, the decision matters more than the likelihood of the data given the model.

*Rule #15: Separate Spam Filtering and Quality Ranking in a Policy Layer.*
Quality ranking is a fine art, but spam filtering is a war. The signals that you use to determine high quality posts will become obvious to those who use your system, and they will tweak their posts to have these properties. Thus, your quality ranking should focus on ranking content that is posted in good faith. You should not discount the quality ranking learner for ranking spam highly. Similarly, "racy" content should be handled separately from Quality Ranking. Spam filtering is a different story. You have to expect that the features that you need to generate will be constantly changing. Often, there will be obvious rules that you put into the system (if a post has more than three spam votes, don’t retrieve it, et cetera). Any learned model will have to be updated daily, if not faster. The reputation of the creator of the content will play a great role.

At some level, the output of these two systems will have to be integrated. Keep in mind, filtering spam in search results should probably be more aggressive than filtering spam in email messages. This is true assuming that you have no regularization and that your algorithm has converged. It is approximately true in general. Also, it is a standard practice to remove spam from the training data for the quality classifier.
