# Machine Learning in Action
==========================

This is the source code to go with "Machine Learning in Action"
by Peter Harrington published by Manning Inc.
The official page for this book can be found here: http://manning.com/pharrington/

All the code examples were working on Python 2.6, there shouldn't be any problems with the 2.7.  NumPy will be needed for most examples.  If you have trouble running any of the examples us know on the Forum for this book: http://www.manning-sandbox.com/forum.jspa?forumID=728.  

If you want to run these on some other version of Python say--3.0 or IronPython, feel free to fork the code.   

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
