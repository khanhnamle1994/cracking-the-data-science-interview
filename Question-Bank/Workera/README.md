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

10. **How to do cross-validation?** ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

- In **hold-out cross validation**, we do the following:
  1. Randomly split training set `S` into `S_train` and `S_cv` (the hold-out cross validation set).
  2. Train each model `M_i` on `S_train` only, to get some hypothesis `h_i`.
  3. Select and output the hypothesis `h_i` that had the smallest error on the hold-out cross validation set.
- The disadvantage of using hold-out cross validation is that it "wastes" some portion of the data. Another method is called **k-fold cross validation** that holds out less data each time.
  - A typically choice for the number of folds would be `k = 10`.
  - This procedure may be more computationally expensive, since we now need to train each model `k` times.
  - If we choose `k = m`, this is called **leave-one-out cross validation**.

11. **How to do feature selection?** ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

- Given `n` features, there are `2^n` possible feature subsets, and thus feature selection can be posed as a model selection problem over `2^n` possible models. For large values of `n`, it's usually too expensive to explicitly enumerate over and compare all `2^n` models, and so typically some heuristic search procedure is used to find a good feature subset.
- *Wrapper model feature selection* is a procedure that "wraps" around your learning algorithm, and repeatedly makes calls to the learning algorithm to evaluate how well it does using different feature subsets. This includes *forward search* and *backward search*.
  - **Forward search** starts off with `F is empty` as the initial set of features. For `i = 1, ..., n`, use some version of cross validation to evaluate features `F_i` and set `F` to be the best feature subset found. Finally, select and output the best feature subset that was evaluated during the entire search procedure.
  - **Backward search** starts off with `F = {1, ..., n}` as the set of all features, and repeatedly deletes features one at a time (evaluating single-feature deletions in a similar manner to how forward search evaluates single-feature additions) until `F is empty`.
- *Filter feature selection* methods give heuristic, but computationally much cheaper, ways of choosing a feature subset. The idea here is to compute some simple score `S(i)` that measures how informative each feature `x_i` is about the class labels `y`. Then, we simply pick the `k` features with the largest scores `S(i)`.
  - One possible choice of the score would be define `S(i)` to be the correlation between `x_i` and `y`, as measured on the training data. This would result in our choosing the features that are the most strongly correlated with the class labels.
  - In practice, it is more common to choose `S(i)` to be the **KL divergence** between `x_i` and `y`. If `x_i` and `y` are independent random variables, then the KL-divergence between the two distributions will be 0. This is consistent with the idea if `x_i` and `y` are independent, then `x_i` is clearly very "non-informative" about `y`, and thus the score `S(i)` should be small. Conversely, if `x_i` is very "informative" about `y`, then their KL-divergence would be large.

12. What is the Bayesian way to combat overfitting? ([Notes](http://cs229.stanford.edu/notes/cs229-notes5.pdf))

13. **How does k-Means clustering algorithm work?** ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes7a.pdf))

- The k-Means clustering algorithm is as follows:
  1. Initialize k **cluster centroids** randomly.
  2. Repeat until convergence:
    - "Assigning" each training example to the closest centroid
    - Moving each cluster centroid to the mean of the points assigned to it.
- The **distortion function** measures the sum of squared distances between each training example and the cluster centroid to which it has been assigned. This value monotonically decrease and will eventually converge.
- This distortion function is non-convex, and so coordinate descent on it is not guaranteed to converge to the global minimum. In other words, k-means can be susceptible to local optima.

14. **Why is the EM algorithm useful?** ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes7b.pdf))

- The Expectation-Maximization algorithm is used for density estimation. It is an iterative algorithm that has 2 main steps:
  - In the E-step, it tries to "guess" the values of the random variables `z`s.
  - In the M-step, it updates the parameters of our model based on our guesses. Since in the M-step we are pretending that the guesses in the first part were correct, the maximization becomes easy.
- The EM-algorithm is reminiscent of the K-means clustering algorithm, except that instead of the "hard" cluster assignments `c`, we instead have the "soft" assignments `w_j`. Similar to K-means, it is also susceptible to local optima, so reinitializing at several different initial parameters may be a good idea.

15. **How does Principal Component Analysis work?** ([Notes](http://cs229.stanford.edu/notes-spring2019/cs229-notes10.pdf))

- PCA tries to identify the subspace in which the data approximately lies.
  - Prior to running PCA, we typically first preprocess the data by normalizing each feature to have mean 0 and variance 1. We do this by subtracting the mean and dividing by the empirical standard deviation.
  - To compute the "major axis of variation" `u` - that is, the direction on which the data approximately lies, we find the unit vector u so that when the data is projected onto the direction onto the direction corresponding to `u`, the variance of the projected data is maximized.
  - More specifically, if we wish to project our data into a k-dimensional subspace, we should choose `u_1, ..., u_k` to be the top k eigenvectors of `\Sum`, which is the empirical covariance matrix of the data. The `u_i`s now form a new, orthogonal basis for the data.
- PCA is referred to as a **dimensionality reduction** algorithm. The vectors `u_1, ..., u_k` are called the first k **principal components** of the data.
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
