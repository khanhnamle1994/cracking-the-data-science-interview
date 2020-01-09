# Machine Learning in Action

This is the source code to go with "Machine Learning in Action"
by Peter Harrington published by Manning Inc.
The official page for this book can be found here: http://manning.com/pharrington/

All the code examples were working on Python 2.6, there shouldn't be any problems with the 2.7.  NumPy will be needed for most examples.  

## Part 1 - Classification

### Chapter 1 - Machine Learning Basics

7 Steps to develop a Machine Learning application:

1. *Collect data*. You could collect the samples by scraping a website and extracting data, or you could get information from an RSS feed or an API. You could have a device collect wind speed measurements and send them to you, or blood glucose levels, or anything you can measure. The number of options is endless. To save some time and effort, you could use publicly available data.

2. *Prepare the input data*. Once you have this data, you need to make sure it's in a useable format. The format in this book is Python list. The benefit of having this standard format is that you can mix and match algorithms and data sources. You may need to do some algorithm-specific formatting here. Some algorithms need features in a special format, some algorithms can deal with target variables and features as strings, and some need them to be integers.

3. *Analyze the input data.* This is looking at the data from the previous task. This could be as simple as looking at the data you’ve parsed in a text editor to make sure steps 1 and 2 are actually working and you don’t have a bunch of empty values. You can also look at the data to see if you can recognize any patterns or if there’s anything obvious, such as a few data points that are vastly different from the rest of the set. Plotting data in one, two, or three dimensions can also help. But most of the time you’ll have more than three features, and you can’t easily plot the data across all features at one time. You could, however, use some advanced methods to distill multiple dimensions down to two or three so you can visualize the data.

4. If you’re working with a production system and you know what the data should look like, or you trust its source, you can skip this step. This step takes human involvement, and for an automated system you don’t want human involvement. The value of this step is that it makes you understand you don’t have garbage coming in.

5. *Train the algorithm.* This is where the machine learning takes place. This step and the next step are where the “core” algorithms lie, depending on the algorithm. You feed the algorithm good clean data from the first two steps and extract knowledge or information. This knowledge you often store in a format that’s readily useable by a machine for the next two steps. In the case of unsupervised learning, there’s no training step because you don’t have a target value. Everything is used in the next step.

6. *Test the algorithm.* This is where the information learned in the previous step is put to use. When you’re evaluating an algorithm, you’ll test it to see how well it does. In the case of supervised learning, you have some known values you can use to evaluate the algorithm. In unsupervised learning, you may have to use some other metrics to evaluate the success. In either case, if you’re not satisfied, you can go back to step 4, change some things, and try testing again. Often the collection or preparation of the data may have been the problem, and you’ll have to go back to step 1.

7. *Use it.* Here you make a real program to do some task, and once again you see if all the previous steps worked as you expected. You might encounter some new data and have to revisit steps 1–5.

### Chapter 2 - Classifying with k-Nearest Neighbors

**Pros:** High accuracy, insensitive to outliers, no assumptions about data.

**Cons:** Computationally expensive, requires a lot of memory.

**Works with:** Numeric values, nominal values.

General Approach to kNN:
1. Collect: Any method.
2. Prepare: Numeric values are needed for a distance calculation. A structured data format is best.
3. Analyze: Any method.
4. Train: Does not apply to the kNN algorithm.
5. Test: Calculate the error rate.
6. Use: This application needs to get some input data and output structured numeric values. Next, the application runs the kNN algorithm on this input data and determines which class the input data should belong to. The application then takes some action on the calculated class.

**Summary**

The k-Nearest Neighbors algorithm is a simple and effective way to classify data. kNN is an example of instance-based learning, where you need to have instances of data close at hand to perform the machine learning algorithm. The algorithm has to carry around the full dataset; for large datasets, this implies a large amount of storage. In addition, you need to calculate the distance measurement for every piece of data in the database, and this can be cumbersome.

An additional drawback is that kNN doesn’t give you any idea of the underlying structure of the data; you have no idea what an “average” or “exemplar” instance from each class looks like.

### Chapter 3 - Splitting datasets one feature at a time: Decision Trees

**Pros:** Computationally cheap to use, easy for humans to understand learned results, missing values OK, can deal with irrelevant features.

**Cons:** Prone to overfitting.

**Works with:** Numeric values, nominal values.

General Approach to Decision Trees:
1. Collect: Any method.
2. Prepare: This tree-building algorithm works only on nominal values, so any continuous values will need to be quantized.
3. Analyze: Any method. You should visually inspect the tree after it is built.
4. Train: Construct a tree data structure.
5. Test: Calculate the error rate with the learned tree.
6. Use: This can be used in any supervised learning task. Often, trees are used to better understand the data.

**Summary**

A decision tree classifier is just like a work-flow diagram with the terminating blocks representing classification decisions. Starting with a dataset, you can measure the inconsistency of a set or the entropy to find a way to split the set until all the data belongs to the same class. The ID3 algorithm can split nominal-valued datasets. Recursion is used in tree-building algorithms to turn a dataset into a decision tree. The tree is easily represented in a Python dictionary rather than a special data structure.

Cleverly applying Matplotlib’s annotations, you can turn our tree data into an easily understood chart. The Python Pickle module can be used for persisting our tree. The contact lens data showed that decision trees can try too hard and overfit a dataset. This overfitting can be removed by pruning the decision tree, combining adjacent leaf nodes that don’t provide a large amount of information gain.

### Chapter 4 - Classifying with probability theory: Naive Bayes

**Pros:** Works with a small amount of data, handles multiple classes.

**Cons:** Sensitive to how the input data is prepared.

**Works with:** Nominal values.

General Approach to naïve Bayes:
1. Collect: Any method.
2. Prepare: Numeric or Boolean values are needed.
3. Analyze: With many features, plotting features isn’t helpful. Looking at histograms is a better idea.
4. Train: Calculate the conditional probabilities of the independent features.
5. Test: Calculate the error rate.
6. Use: One common application of naïve Bayes is document classification. You can use naïve Bayes in any classification setting. It doesn’t have to be text.

**Summary**

Using probabilities can sometimes be more effective than using hard rules for classification. Bayesian probability and Bayes’ rule gives us a way to estimate unknown probabilities from known values.

You can reduce the need for a lot of data by assuming conditional independence among the features in your data. The assumption we make is that the probability of one word doesn’t depend on any other words in the document. We know this assumption is a little simple. That’s why it’s known as naïve Bayes. Despite its incorrect assumptions, naïve Bayes is effective at classification.

There are a number of practical considerations when implementing naïve Bayes in a modern programming language. Underflow is one problem that can be addressed by using the logarithm of probabilities in your calculations. The bag-of-words model is an improvement on the set-of-words model when approaching document classification. There are a number of other improvements, such as removing stop words, and you can spend a long time optimizing a tokenizer.

### Chapter 5 - Logistic Regression

### Chapter 6 - Support Vector Machines

### Chapter 7 - Improving classification with the Ada-Boost Meta-Algorithm

## Part 2 - Forecasting numeric values with Regression

### Chapter 8 - Predicting numeric values: Regression

### Chapter 9 - Tree-based Regression

## Part 3 - Unsupervised Learning

### Chapter 10 - Grouping unlabeled items using k-Means Clustering

### Chapter 11 - Association analysis with the Apriori algorithm

### Chapter 12 - Efficiently finding frequent itemsets with FP-growth

## Part 4 - Additional Tools

### Chapter 13 - Using Principal Component Analysis to simplify data

### Chapter 14 - Simplifying data with the Singular Value Decomposition

### Chapter 15 - Big Data and MapReduce
