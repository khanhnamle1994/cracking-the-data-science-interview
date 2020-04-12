## [The Machine Learning Algorithms Interview](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/Question-Bank/Workera/Machine-Learning-Algorithms-Interview.pdf)

([Full Resource](https://workera.ai/resources/machine-learning-algorithms-interview/))

1. Derive the binary cross-entropy + mean-squared error loss function.

2. Explain Linear Regression ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf))

3. Explain Logistic Regression ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf))

4. Explain Generalized Linear Models ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf))

5. In Support Vector Machines, what is the kernel trick? ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes3.pdf))

6. Why is the Naive Bayes classifier called Naive? ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes2.pdf))

7. How does a discriminative model differ from a generative model? ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes2.pdf))

8. What is the bias-variance tradeoff? ([Notes](http://cs229.stanford.edu/section/error-analysis.pdf))

9. How to do error analysis in a machine learning pipeline? ([Notes](http://cs229.stanford.edu/section/error-analysis.pdf))

10. How to do cross-validation? ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

11. How to do feature selection? ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

12. What is the Bayesian way to combat overfitting? ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

13. How does k-Means clustering algorithm work? ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes7a.pdf))

14. Why is the EM algorithm useful? ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes7b.pdf))

15. **How does Principal Component Analysis work?** ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes10.pdf))

- PCA tries to identify the subspace in which the data approximately lies.
  - Prior to running PCA, we typically first preprocess the data by normalizing each feature to have mean 0 and variance 1. We do this by subtracting the mean and dividing by the empirical standard deviation.
  - To compute the "major axis of variation" `u` - that is, the direction on which the data approximately lies, we find the unit vector u so that when the data is projected onto the direction onto the direction corresponding to `u`, the variance of the projected data is maximized.
  - More specifically, if we wish to project our data into a k-dimensional subspace, we should choose `u1, ..., uk` to be the top k eigenvectors of `\Sum`, which is the empirical covariance matrix of the data. The `u_i`s now form a new, orthogonal basis for the data.
- PCA is referred to as a **dimensionality reduction** algorithm. The vectors `u1, ..., uk` are called the first k **principal components** of the data.
- PCA has many applications:
  - Compression is an obvious application. If we reduce high dimensional data to `k = 2` or `3` dimensions, then we can also plot the `y^(i)`'s to visualize the data.
  - Another standard application is to preprocess a dataset to reduce its dimension before running a supervised learning algorithm with the inputs. Apart from computational benefits, reducing the data's dimension can also reduce the complexity of the hypohtesis class considered and help avoid overfitting.
  - Lastly, we can also view PCA as a noise reduction algorithm.

## [The Deep Learning Algorithms Interview](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/Question-Bank/Workera/Deep-Learning-Algorithms-Interview.pdf)

1. Explain Neural Networks from first principle ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes-deep_learning.pdf))

2. What is the most effective initialization for neural networks? ([Notes](https://www.deeplearning.ai/ai-notes/initialization/index.html))

3. Explain how back-propagation works in a fully-connected neural network ([Notes](http://cs230.stanford.edu/section/3/))

4. What is the difference between Vanilla, Mini-Batch, and Stochastic Gradient Descent?

5. What is your process of optimizing parameters in neural networks? ([Notes](https://www.deeplearning.ai/ai-notes/optimization/))

6. What is your strategy for hyper-parameter tuning? ([Notes](http://cs230.stanford.edu/section/7/))

7. What is your approach to write a deep learning paper? ([Notes](http://cs230.stanford.edu/section/8/))

## [The Machine Learning Case Study Interview](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/Question-Bank/Workera/Machine-Learning-Case-Study-Interview.pdf)

1. How would you build a trigger word detection algorithm to spot the word "activate" in a 10 second long audio clip?

2. An e-commerce company is trying to minimize the time it takes customer to purchase their selected items. As a ML engineer, what can you do to help them?

3. You are given a dataset of credit card purchases information. Each record is labeled as fraudulent or safe. You are asked to build a fraud detection algorithm. How would you proceed?

4. You are provided with data from a music streaming platform. Each of the 100,000 records indicates the songs a user has listened to in the past month. How would you build a music recommendation system?

**Company Machine Learning Case Studies**

1. [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)

2. [Machine Learning at Facebook: Understanding Inference at the Edge](https://research.fb.com/wp-content/uploads/2018/12/Machine-Learning-at-Facebook-Understanding-Inference-at-the-Edge.pdf)

3. [Empowering personalized marketing with machine learning](https://eng.lyft.com/empowering-personalized-marketing-with-machine-learning-fd36e6bdeca6)

4. [Learning a Personalized Homepage](https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a)
