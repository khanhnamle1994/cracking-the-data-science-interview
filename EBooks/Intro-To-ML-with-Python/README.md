# Introduction to Machine Learning with Python

This repository holds the code for the book "Introduction to Machine Learning with Python" by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido). You can find details about the book on the [O'Reilly website](http://shop.oreilly.com/product/0636920030515.do).

The books requires the current stable version of scikit-learn. Most of the book can also be used with previous versions of scikit-learn, though you need to adjust the import for everything from the ``model_selection`` module, mostly ``cross_val_score``, ``train_test_split`` and ``GridSearchCV``.

This repository provides the notebooks from which the book is created, together with the ``mglearn`` library of helper functions to create figures and datasets.

All datasets are included in the repository, with the exception of the aclImdb dataset, which you can download from the page of [Andrew Maas](http://ai.stanford.edu/~amaas/data/sentiment/). See the book for details.

If you get ``ImportError: No module named mglearn`` you can try to install mglearn into your python environment using the command ``pip install mglearn`` in your terminal or ``!pip install mglearn`` in Jupyter Notebook.

Here are the chapters:

* [Introduction](#introduction)
* [Supervised Learning](#supervised-learning)
* [Unsupervised Learning](#unsupervised-learning)
* [Representing Data and Engineering Features](#representing-data-and-engineering-features)
* [Model Evaluation and Improvement](#model-evaluation-and-improvement)
* [Algorithm Chains and Pipelines](#algorithm-chains-and-pipelines)
* [Wrapping Up](#wrapping-up)

## Introduction

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/01-introduction.ipynb).

* [scikit-learn](#scikitlearn)
* [Jupyter Notebook](#jupyter-notebook)
* [NumPy](#numpy)
* [SciPy](#scipy)
* [Matplotlib](#matplotlib)
* [Pandas](#pandas)

### ScikitLearn

* scikit-learn is an open source project, meaning that it is free to use and distribute, and anyone can easily obtain the source code to see what is going on behind the scenes. The scikit-learn project is constantly being developed and improved, and it has a very active user community. It contains a number of state-of-the-art machine learning algorithms, as well as comprehensive documentation about each algorithm.
* scikit-learn is a very popular tool, and the most prominent Python library for machine learning. It is widely used in industry and academia, and a wealth of tutorials and code snippets are available online. scikit-learn works well with a number of other scientific Python tools.

[back to current section](#introduction)

### Jupyter Notebook

* Jupyter Notebook is an interactive environment for running code in the browser. It is a great tool for exploratory data analysis and is widely used by data scientists.
* While the Jupyter Notebook supports many programming languages, we only need the Python support. The Jupyter Notebook makes it easy to incorporate code, text, and images, and all of this book was in fact written as a Jupyter Notebook.

[back to current section](#introduction)

### NumPy

* NumPy is one of the fundamental packages for scientific computing in Python. It contains functionality for multidimensional arrays, high-level mathematical functions such as linear algebra operations and the Fourier transform, and pseudorandom number generators.
* In scikit-learn, the NumPy array is the fundamental data structure. scikit-learn takes in data in the form of NumPy arrays. Any data you’re using will have to be converted to a NumPy array. The core functionality of NumPy is the `ndarray` class, a multidimensional (n-dimensional) array. All elements of the array must be of the same type.

[back to current section](#introduction)

### SciPy

* SciPy is a collection of functions for scientific computing in Python. It provides, among other functionality, advanced linear algebra routines, mathematical function optimization, signal processing, special mathematical functions, and statistical distributions.
* scikit-learn draws from SciPy’s collection of functions for implementing its algorithms. The most important part of SciPy for us is `scipy.sparse`: this provides sparse matrices, which are another representation that is used for data in scikitlearn.

[back to current section](#introduction)

### Matplotlib

* matplotlib is the primary scientific plotting library in Python. It provides functions for making publication-quality visualizations such as line charts, histograms, scatter plots, and so on. Visualizing your data and different aspects of your analysis can give you important insights.
* When working inside the Jupyter Notebook, you can show figures directly in the browser by using the `%matplotlib notebook` and `%matplotlib inline` commands.

[back to current section](#introduction)

### Pandas

* pandas is a Python library for data wrangling and analysis. It is built around a data structure called the `DataFrame` that is modeled after the R DataFrame. Simply put, a pandas DataFrame is a table, similar to an Excel spreadsheet.
* pandas provides a great range of methods to modify and operate on this table; in particular, it allows SQL-like queries and joins of tables. In contrast to NumPy, which requires that all entries in an array be of the same type, pandas allows each column to have a separate type (for example, integers, dates, floating-point numbers, and strings).
* Another valuable tool provided by pandas is its ability to ingest from a great variety of file formats and databases, like SQL, Excel files, and comma-separated values (CSV) files.

[back to current section](#introduction)

[back to top](#introduction-to-machine-learning-with-python)

## Supervised Learning

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/02-supervised-learning.ipynb).

* [Classification and Regression](#classification-and-regression)
* [Generalization, Overfitting, and Underfitting](#generalization-overfitting-and-underfitting)
* [k-Nearest-Neighbors](#k-Nearest-Neighbors): For small datasets, good as a baseline, easy to explain.
* [Linear Models](#linear-models): Go-to as a first algorithm to try, good for very large datasets, good for very highdimensional data.
* [Naive Bayes](#naive-bayes): Only for classification. Even faster than linear models, good for very large datasets and high-dimensional data. Often less accurate than linear models.
* [Decision Trees](#decision-trees): Very fast, don’t need scaling of the data, can be visualized and easily explained.
* [Random Forests](#random-forests): Nearly always perform better than a single decision tree, very robust and powerful. Don’t need scaling of data. Not good for very high-dimensional sparse data.
* [Gradient Boosting Machines](#gradient-boosting-machines): Often slightly more accurate than random forests. Slower to train but faster to predict than random forests, and smaller in memory. Need more parameter tuning than random forests.
* [Kernelized Support Vector Machines](#kernelized-support-vector-machines): Powerful for medium-sized datasets of features with similar meaning. Require scaling of data, sensitive to parameters.
* [Neural Networks](#neural-networks): Can build very complex models, particularly for large datasets. Sensitive to scaling of the data and to the choice of parameters. Large models need a long time to train.

### Classification and Regression

* In **classification**, the goal is to predict a class label, which is a choice from a predefined list of possibilities. Classification is sometimes separated into binary classification, which is the special case of distinguishing between exactly two classes, and multiclass classification, which is classification between more than two classes.
* For **regression** tasks, the goal is to predict a continuous number, or a floating-point number in programming terms (or real number in mathematical terms). Predicting a person’s annual income from their education, their age, and where they live is an example of a regression task. When predicting income, the predicted value is an amount, and can be any number in a given range. Another example of a regression task is predicting the yield of a corn farm given attributes such as previous yields, weather, and number of employees working on the farm. The yield again can be an arbitrary number.

[back to current section](#supervised-learning)

### Generalization, Overfitting, and Underfitting

* In supervised learning, we want to build a model on the training data and then be able to make accurate predictions on new, unseen data that has the same characteristics as the training set that we used. If a model is able to make accurate predictions on unseen data, we say it is able to **generalize** from the training set to the test set. We want to build a model that is able to generalize as accurately as possible.
* Building a model that is too complex for the amount of information we have is called **overfitting**. Overfitting occurs when you fit a model too closely to the particularities of the training set and obtain a model that works well on the training set but is not able to generalize to new data. On the other hand, if your model is too simple, then you might not be able to capture all the aspects of and variability in the data, and your model will do badly even on the training set. Choosing too simple a model is called **underfitting**.
* The tradeoff between over and under-fitting is illustrated below:

![overfitting-underfitting](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/overfitting_underfitting_cartoon.png)

[back to current section](#supervised-learning)

### k-Nearest-Neighbors

```
from sklearn.neighbors import KNeighborsClassifier/KNeighborsRegressor
clf = KNeighborsClassifier/KNeighborsRegressor(n_neighbors=3)
```

* In principle, there are two important parameters to the KNeighbors classifier: **the number of neighbors** and how you measure **distance between data points**. In practice, using a small number of neighbors like three or five often works well, but you should certainly adjust this parameter. Choosing the right distance measure is somewhat beyond the scope of this book. By default, Euclidean distance is used, which works well in many settings.
* One of the strengths of k-NN is that the model is very easy to understand, and often gives reasonable performance without a lot of adjustments. Using this algorithm is a good baseline method to try before considering more advanced techniques. Building the nearest neighbors model is usually very fast, but when your training set is very large (either in number of features or in number of samples) prediction can be slow.
* When using the k-NN algorithm, it’s important to preprocess your data. This approach often does not perform well on datasets with many features (hundreds or more), and it does particularly badly with datasets where most features are 0 most of the time (so-called sparse datasets).
* So, while the nearest k-neighbors algorithm is easy to understand, it is not often used in practice, due to prediction being slow and its inability to handle many features.

![kNN](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/kNN.png)

[back to current section](#supervised-learning)

### Linear Models

```
# Linear models for regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)

# Linear models for classification
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X_train, y_train)

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
```

* The main parameter of linear models is the regularization parameter, called `alpha` in the regression models and `C` in `LinearSVC` and `LogisticRegression`. Large values for alpha or small values for C mean simple models. In particular for the regression models, tuning these parameters is quite important. Usually C and alpha are searched for on a logarithmic scale.
* The other decision you have to make is whether you want to use L1 regularization or L2 regularization. If you assume that only a few of your features are actually important, you should use L1. Otherwise, you should default to L2.
* L1 can also be useful if interpretability of the model is important. As L1 will use only a few features, it is easier to explain which features are important to the model, and what the effects of these features are.
* Linear models are very fast to train, and also fast to predict. They scale to very large datasets and work well with sparse data. If your data consists of hundreds of thousands or millions of samples, you might want to investigate using the `solver='sag'` option in `LogisticRegression` and `Ridge`, which can be faster than the default on large datasets. Other options are the `SGDClassifier` class and the `SGDRegressor` class, which implement even more scalable versions of the linear models described here.
* Another strength of linear models is that they make it relatively easy to understand how a prediction is made. Unfortunately, it is often not entirely clear why coefficients are the way they are. This is particularly true if your dataset has highly correlated features; in these cases, the coefficients might be hard to interpret.
* Linear models often perform well when the number of features is large compared to the number of samples. They are also often used on very large datasets, simply because it’s not feasible to train other models. However, in lower-dimensional spaces, other models might yield better generalization performance.

![linear-models](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/linear-models.png)

[back to current section](#supervised-learning)

### Naive Bayes

* `MultinomialNB` and `BernoulliNB` have a single parameter, alpha, which controls model complexity. The way alpha works is that the algorithm adds to the data alpha many virtual data points that have positive values for all the features. This results in a “smoothing” of the statistics. A large alpha means more smoothing, resulting in less complex models. The algorithm’s performance is relatively robust to the setting of alpha, meaning that setting alpha is not critical for good performance. However, tuning it usually improves accuracy somewhat.
* `GaussianNB` is mostly used on very high-dimensional data, while the other two variants of naive Bayes are widely used for sparse count data such as text. `MultinomialNB` usually performs better than `BinaryNB`, particularly on datasets with a relatively large number of nonzero features (i.e., large documents).
* The naive Bayes models share many of the strengths and weaknesses of the linear models. They are very fast to train and to predict, and the training procedure is easy to understand. The models work very well with high-dimensional sparse data and are relatively robust to the parameters. Naive Bayes models are great baseline models and are often used on very large datasets, where training even a linear model might take too long.

[back to current section](#supervised-learning)

### Decision Trees

```
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
tree.feature_importances_

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
```

* The parameters that control model complexity in decision trees
are the pre-pruning parameters that stop the building of the tree before it is fully developed. Usually, picking one of the pre-pruning strategies—setting either `max_depth`, `max_leaf_nodes`, or `min_samples_leaf`—is sufficient to prevent overfitting.
* Decision trees have two advantages over many of the algorithms we’ve discussed so far: the resulting model can easily be visualized and understood by nonexperts (at least for smaller trees), and the algorithms are completely invariant to scaling of the data. As each feature is processed separately, and the possible splits of the data don’t depend on scaling, no preprocessing like normalization or standardization of features is needed for decision tree algorithms. In particular, decision trees work well when you have features that are on completely different scales, or a mix of binary and continuous features.
* The main downside of decision trees is that even with the use of pre-pruning, they tend to overfit and provide poor generalization performance. Therefore, in most applications, the ensemble methods we discuss next are usually used in place of a single decision tree.

![decision-trees](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/decision-trees.svg)

[back to current section](#supervised-learning)

### Random Forests

```
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
```

* Random forests for regression and classification are currently among the most widely used machine learning methods. They are very powerful, often work well without heavy tuning of the parameters, and don’t require scaling of the data.
* Essentially, random forests share all of the benefits of decision trees, while making up for some of their deficiencies. One reason to still use decision trees is if you need a compact representation of the decision-making process. It is basically impossible to interpret tens or hundreds of trees in detail, and trees in random forests tend to be deeper than decision trees (because of the use of feature subsets). Therefore, if you need to summarize the prediction making in a visual way to nonexperts, a single decision tree might be a better choice. While building random forests on large datasets might be somewhat time consuming, it can be parallelized across multiple CPU cores within a computer easily. If you are using a multi-core processor (as nearly all modern computers do), you can use the `n_jobs` parameter to adjust the number of cores to use. Using more CPU cores will result in linear speed-ups (using two cores, the training of the random forest will be twice as fast), but specifying `n_jobs` larger than the number of cores will not help. You can set `n_jobs=-1` to use all the cores in your computer.
* You should keep in mind that random forests, by their nature, are random, and setting different random states (or not setting the `random_state` at all) can drastically change the model that is built. The more trees there are in the forest, the more robust it will be against the choice of random state. If you want to have reproducible results, it is important to fix the `random_state`.
* Random forests don’t tend to perform well on very high dimensional, sparse data, such as text data. For this kind of data, linear models might be more appropriate. Random forests usually work well even on very large datasets, and training can easily be parallelized over many CPU cores within a powerful computer. However, random forests require more memory and are slower to train and to predict than linear models. If time and memory are important in an application, it might make sense to use a linear model instead.
* The important parameters to adjust are `n_estimators`, `max_features`, and possibly pre-pruning options like `max_depth`. For n_estimators, larger is always better. Averaging more trees will yield a more robust ensemble by reducing overfitting. However, there are diminishing returns, and more trees need more memory and more time to train. A common rule of thumb is to build “as many as you have time/memory for.”
* `max_features` determines how random each tree is, and a smaller `max_features` reduces overfitting. In general, it’s a good rule of thumb to use the default values: `max_features=sqrt(n_features)` for classification and `max_features=log2(n_features)` for regression. Adding `max_features` or `max_leaf_nodes` might sometimes improve performance. It can also drastically reduce space and time requirements for training and prediction.

![random-forest](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/rf_fi.png)

[back to current section](#supervised-learning)

### Gradient Boosting Machines

```
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.01)
gbrt.fit(X_train, y_train)
```

* Gradient boosted decision trees are among the most powerful and widely used models for supervised learning. Their main drawback is that they require careful tuning of the parameters and may take a long time to train. Similarly to other tree-based models, the algorithm works well without scaling and on a mixture of binary and continuous features. As with other tree-based models, it also often does not work well on high-dimensional sparse data.
* The main parameters of gradient boosted tree models are the number of trees, `n_estimators`, and the `learning_rate`, which controls the degree to which each tree is allowed to correct the mistakes of the previous trees. These two parameters are highly interconnected, as a lower `learning_rate` means that more trees are needed to build a model of similar complexity. In contrast to random forests, where a higher `n_estimators` value is always better, increasing `n_estimators` in gradient boosting leads to a more complex model, which may lead to overfitting. A common practice is to fit `n_estimators` depending on the time and memory budget, and then search over different `learning_rates`.
* Another important parameter is `max_depth` (or alternatively `max_leaf_nodes`), to reduce the complexity of each tree. Usually `max_depth` is set very low for gradient boosted models, often not deeper than five splits.

![gradient-boosting-machines](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/gbm_fi.png)

[back to current section](#supervised-learning)

### Kernelized Support Vector Machines

```
# Linear Models and Non-linear Features
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

# Kernel Trick
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
sv = svm.support_vectors_
```

* Kernelized support vector machines are powerful models and perform well on a variety of datasets. SVMs allow for complex decision boundaries, even if the data has only a few features. They work well on low-dimensional and high-dimensional data (i.e., few and many features), but don’t scale very well with the number of samples. Running an SVM on data with up to 10,000 samples might work well, but working with datasets of size 100,000 or more can become challenging in terms of runtime and memory usage.
* Another downside of SVMs is that they require careful preprocessing of the data and tuning of the parameters. This is why, these days, most people instead use tree-based models such as random forests or gradient boosting (which require little or no preprocessing) in many applications. Furthermore, SVM models are hard to inspect; it can be difficult to understand why a particular prediction was made, and it might be tricky to explain the model to a nonexpert.
* Still, it might be worth trying SVMs, particularly if all of your features represent measurements in similar units (e.g., all are pixel intensities) and they are on similar scales.
* The important parameters in kernel SVMs are the regularization parameter `C`, the choice of the kernel, and the kernel-specific parameters. Although we primarily focused on the RBF kernel, other choices are available in `scikit-learn`. The RBF kernel has only one parameter, `gamma`, which is the inverse of the width of the Gaussian kernel. `gamma` and `C` both control the complexity of the model, with large values in either resulting in a more complex model. Therefore, good settings for the two parameters are usually strongly correlated, and `C` and `gamma` should be adjusted together.

![svm](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/svm.png)

[back to current section](#supervised-learning)

### Neural Networks

```
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', activation='tanh', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
```

* Neural networks have reemerged as state-of-the-art models in many applications of machine learning. One of their main advantages is that they are able to capture information contained in large amounts of data and build incredibly complex models. Given enough computation time, data, and careful tuning of the parameters, neural networks often beat other machine learning algorithms (for classification and regression tasks).
* This brings us to the downsides. Neural networks—particularly the large and powerful ones—often take a long time to train. They also require careful preprocessing of the data, as we saw here. Similarly to SVMs, they work best with “homogeneous” data, where all the features have similar meanings. For data that has very different kinds of features, tree-based models might work better. Tuning neural network parameters is also an art unto itself. In our experiments, we barely scratched the surface of possible ways to adjust neural network models and how to train them.
* The most important parameters are the number of layers and the number of hidden units per layer. You should start with one or two hidden layers, and possibly expand from there. The number of nodes per hidden layer is often similar to the number of input features, but rarely higher than in the low to mid-thousands.
* A helpful measure when thinking about the model complexity of a neural network is the number of weights or coefficients that are learned. If you have a binary classification dataset with 100 features, and you have 100 hidden units, then there are 100 * 100 = 10,000 weights between the input and the first hidden layer. There are also 100 * 1 = 100 weights between the hidden layer and the output layer, for a total of around 10,100 weights. If you add a second hidden layer with 100 hidden units, there will be another 100 * 100 = 10,000 weights from the first hidden layer to the second hidden layer, resulting in a total of 20,100 weights. If instead you use one layer with 1,000 hidden units, you are learning 100 * 1,000 = 100,000 weights from the input to the hidden layer and 1,000 x 1 weights from the hidden layer to the output layer, for a total of 101,000. If you add a second hidden layer you add 1,000 * 1,000 = 1,000,000 weights, for a whopping total of 1,101,000—50 times larger than the model with two hidden layers of size 100.
* A common way to adjust parameters in a neural network is to first create a network that is large enough to overfit, making sure that the task can actually be learned by the network. Then, once you know the training data can be learned, either shrink the network or increase alpha to add regularization, which will improve generalization performance.

![mlp](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/mlp.png)

[back to current section](#supervised-learning)

When working with a new dataset, it is in general a good idea to start with a simple model, such as a linear model or a naive Bayes or nearest neighbors classifier, and see how far you can get. After understanding more about the data, you can consider moving to an algorithm that can build more complex models, such as random forests, gradient boosted decision trees, SVMs, or neural networks.

![comparison](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/classifier_comparison.png)

[back to top](#introduction-to-machine-learning-with-python)

## Unsupervised Learning

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/03-unsupervised-learning.ipynb).

* [Types of Unsupervised Learning](#types-of-unsupervised-learning)
* [Challenges in Unsupervised Learning](#challenges-in-unsupervised-learning)
* [Principal Component Analysis](#principal-component-analysis)
* [Non Negative Matrix Factorization](#non-negative-matrix-factorization)
* [Manifold Learning with tSNE](#manifold-learning-with-tSNE)
* [k Means Clustering](#k-means-clustering)
* [Agglomerative Clustering](#agglomerative-clustering)
* [DBSCAN](#dbscan)

### Types of Unsupervised Learning

* *Unsupervised transformations* of a dataset are algorithms that create a new representation of the data which might be easier for humans or other machine learning algorithms to understand compared to the original representation of the data. A common application of unsupervised transformations is dimensionality reduction, which takes a high-dimensional representation of the data, consisting of many features, and finds a new way to represent this data that summarizes the essential characteristics with fewer features. A common application for dimensionality reduction is reduction to two dimensions for visualization purposes.
* Another application for unsupervised transformations is finding the parts or components that “make up” the data. An example of this is topic extraction on collections of text documents. Here, the task is to find the unknown topics that are talked about in each document, and to learn what topics appear in each document. This can be useful for tracking the discussion of themes like elections, gun control, or pop stars on social media.
* *Clustering algorithms*, on the other hand, partition data into distinct groups of similar items. Consider the example of uploading photos to a social media site. To allow you to organize your pictures, the site might want to group together pictures that show the same person. However, the site doesn’t know which pictures show whom, and it doesn’t know how many different people appear in your photo collection. A sensible approach would be to extract all the faces and divide them into groups of faces that look similar. Hopefully, these correspond to the same person, and the images can be grouped together for you.

[back to current section](#unsupervised-learning)

### Challenges in Unsupervised Learning

* A major challenge in unsupervised learning is evaluating whether the algorithm learned something useful. Unsupervised learning algorithms are usually applied to data that does not contain any label information, so we don’t know what the right output should be. Therefore, it is very hard to say whether a model “did well.” For example, our hypothetical clustering algorithm could have grouped together all the pictures that show faces in profile and all the full-face pictures. This would certainly be a possible way to divide a collection of pictures of people’s faces, but it’s not the one we were looking for. However, there is no way for us to “tell” the algorithm what we are looking for, and often the only way to evaluate the result of an unsupervised algorithm is to inspect it manually.
* As a consequence, unsupervised algorithms are used often in an exploratory setting, when a data scientist wants to understand the data better, rather than as part of a larger automatic system. Another common application for unsupervised algorithms is as a preprocessing step for supervised algorithms. Learning a new representation of the data can sometimes improve the accuracy of supervised algorithms, or can lead to reduced memory and time consumption.

[back to current section](#unsupervised-learning)

### Principal Component Analysis

```
from sklearn.decomposition import PCA
# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to beast cancer data
pca.fit(X_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
```

* Principal component analysis is a method that rotates the dataset in a way such that the rotated features are statistically uncorrelated. This rotation is often followed by selecting only a subset of the new features, according to how important they are for explaining the data.

The following example illustrates the effect of PCA on a synthetic two-dimensional dataset:

![pca](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/pca.png)

* The first plot (top left) shows the original data points, colored to distinguish among them. The algorithm proceeds by first finding the direction of maximum variance, labeled “Component 1.” This is the direction (or vector) in the data that contains most of the information, or in other words, the direction along which the features are most correlated with each other. Then, the algorithm finds the direction that contains the most information while being orthogonal (at a right angle) to the first direction. In two dimensions, there is only one possible orientation that is at a right angle, but in higher-dimensional spaces there would be (infinitely) many orthogonal directions. Although the two components are drawn as arrows, it doesn’t really matter where the head and the tail are; we could have drawn the first component from the center up to the top left instead of down to the bottom right. The directions found using this process are called *principal components*, as they are the main directions of variance in the data. In general, there are as many principal components as original features.
* The second plot (top right) shows the same data, but now rotated so that the first principal component aligns with the x-axis and the second principal component aligns with the y-axis. Before the rotation, the mean was subtracted from the data, so that the transformed data is centered around zero. In the rotated representation found by PCA, the two axes are uncorrelated, meaning that the correlation matrix of the data in this representation is zero except for the diagonal.
* We can use PCA for dimensionality reduction by retaining only some of the principal components. In this example, we might keep only the first principal component, as shown in the third panel in the bottom left. This reduces the data from a two-dimensional dataset to a one-dimensional dataset. Note, however, that instead of keeping only one of the original features, we found the most interesting direction (top left to bottom right in the first panel) and kept this direction, the first principal component.
* Finally, we can undo the rotation and add the mean back to the data. This will result in the data shown in the last panel in the bottom right. These points are in the original feature space, but we kept only the information contained in the first principal component. This transformation is sometimes used to remove noise effects from the data or visualize what part of the information is retained using the principal components.

[back to current section](#unsupervised-learning)

### Non Negative Matrix Factorization

```
from sklearn.decomposition import NMF
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
```

* Non-negative matrix factorization is another unsupervised learning algorithm that aims to extract useful features. It works similarly to PCA and can also be used for dimensionality reduction. As in PCA, we are trying to write each data point as a weighted sum of some components. But whereas in PCA we wanted components that were orthogonal and that explained as much variance of the data as possible, in NMF, we want the components and the coefficients to be nonnegative; that is, we want both the components and the coefficients to be greater than or equal to zero. Consequently, this method can only be applied to data where each feature is non-negative, as a non-negative sum of non-negative components cannot become negative.
* The process of decomposing data into a non-negative weighted sum is particularly helpful for data that is created as the addition (or overlay) of several independent sources, such as an audio track of multiple people speaking, or music with many instruments. In these situations, NMF can identify the original components that make up the combined data. Overall, NMF leads to more interpretable components than PCA, as negative components and coefficients can lead to hard-to-interpret cancellation effects.

The following example shows the results of NMF on the two-dimensional toy data:

![nmf](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/nmf.png)

* For NMF with two components, as shown on the left, it is clear that all points in the data can be written as a positive combination of the two components. If there are enough components to perfectly reconstruct the data (as many components as there are features), the algorithm will choose directions that point toward the extremes of the data.
* If we only use a single component, NMF creates a component that points toward the mean, as pointing there best explains the data. You can see that in contrast with PCA, reducing the number of components not only removes some directions, but creates an entirely different set of components! Components in NMF are also not ordered in any specific way, so there is no “first non-negative component”: all components play an equal part.
* NMF uses a random initialization, which might lead to different results depending on the random seed. In relatively simple cases such as the synthetic data with two components, where all the data can be explained perfectly, the randomness has little effect (though it might change the order or scale of the components). In more complex situations, there might be more drastic changes.

[back to current section](#unsupervised-learning)

### Manifold Learning with tSNE

```
from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# use fit_transform instead of fit, as TSNE has no transform method
digits_tsne = tsne.fit_transform(digits.data)
```

* Manifold learning algorithms are mainly aimed at visualization, and so are rarely used to generate more than two new features. Some of them, including t-SNE, compute a new representation of the training data, but don’t allow transformations of new data. This means these algorithms cannot be applied to a test set: rather, they can only transform the data they were trained for. Manifold learning can be useful for exploratory data analysis, but is rarely used if the final goal is supervised learning.
* The idea behind t-SNE is to find a two-dimensional representation of the data that preserves the distances between points as best as possible. t-SNE starts with a random two-dimensional representation for each data point, and then tries to make points that are close in the original feature space closer, and points that are far apart in the original feature space farther apart.
* t-SNE puts more emphasis on points that are close by, rather than preserving distances between far-apart points. In other words, it tries to preserve the information indicating which points are neighbors to each other.

![t-SNE](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/t-SNE.png)

[back to current section](#unsupervised-learning)

### k Means Clustering

```
from sklearn.cluster import KMeans
# build the clustering model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
kmeans.predict(X)
```

* k-means clustering is one of the simplest and most commonly used clustering algorithms.
* It tries to find cluster centers that are representative of certain regions of the data.
* The algorithm alternates between two steps: assigning each data point to the closest cluster center, and then setting each cluster center as the mean of the data points that are assigned to it.
* The algorithm is finished when the assignment of instances to clusters no longer changes.

The following example illustrates the algorithm on a synthetic dataset:

![k-Means](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/k-Means.png)

* k-means is a very popular algorithm for clustering, not only because it is relatively easy to understand and implement, but also because it runs relatively quickly. k-means scales easily to large datasets, and scikit-learn even includes a more scalable variant in the MiniBatchKMeans class, which can handle very large datasets.
* One of the drawbacks of k-means is that it relies on a random initialization, which means the outcome of the algorithm depends on a random seed. By default, scikit-learn runs the algorithm 10 times with 10 different random initializations, and returns the best result.
* Further downsides of k-means are the relatively restrictive assumptions made on the shape of clusters, and the requirement to specify the number of clusters you are looking for (which might not be known in a real-world application).

[back to current section](#unsupervised-learning)

### Agglomerative Clustering

```
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
```

* Agglomerative clustering refers to a collection of clustering algorithms that all build upon the same principles: the algorithm starts by declaring each point its own cluster, and then merges the two most similar clusters until some stopping criterion is satisfied.
* The stopping criterion implemented in scikit-learn is the number of clusters, so similar clusters are merged until only the specified number of clusters are left.
* There are several linkage criteria that specify how exactly the “most similar cluster” is measured. This measure is always defined between two existing clusters.

The following plot illustrates the progression of agglomerative clustering on a two-dimensional dataset, looking for three clusters:

![agglomerative-clustering](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/agglomerative-clustering.png)

[back to current section](#unsupervised-learning)

### DBSCAN

```
from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
```

* Another very useful clustering algorithm is DBSCAN (which stands for “density-based spatial clustering of applications with noise”). The main benefits of DBSCAN are that it does not require the user to set the number of clusters a priori, it can capture clusters of complex shapes, and it can identify points that are not part of any cluster.
* DBSCAN is somewhat slower than agglomerative clustering and k-means, but still scales to relatively large datasets.
* DBSCAN works by identifying points that are in “crowded” regions of the feature space, where many data points are close together. These regions are referred to as dense regions in feature space. The idea behind DBSCAN is that clusters form dense regions of data, separated by regions that are relatively empty.

![DBSCAN](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/DBSCAN.png)

[back to current section](#unsupervised-learning)

[back to top](#introduction-to-machine-learning-with-python)

## Representing Data and Engineering Features

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/04-representing-data-feature-engineering.ipynb).

* [Categorical Variables](#categorical-variables)
* [Univariate Nonlinear Transformations](#univariate-nonlinear-transformations)
* [Automatic Feature Selection](#automatic-feature-selection)

### Categorical Variables

* By far the most common way to represent categorical variables is using the *one-hot-encoding* or *one-out-of-N encoding*, also known as *dummy variables*. The idea behind dummy variables is to replace a categorical variable with one or more new features that can have the values 0 and 1. The values 0 and 1 make sense in the formula for linear binary classification, and we can represent any number of categories by introducing one new feature per category.
* The `get_dummies` function automatically transform all columns that have object type or are categorical.

```
data_dummies = pd.get_dummies(data)
```

[back to current section](#representing-data-and-engineering-features)

### Univariate Nonlinear Transformations

* Adding squared or cubed features can help linear models for regression. There are other transformations that often prove useful for transforming certain features: in particular, applying mathematical functions like log, exp, or sin. While tree-based models only care about the ordering of the features, linear models and neural networks are very tied to the scale and distribution of each feature, and if there is a nonlinear relation between the feature and the target, that becomes hard to model — particularly in regression. The functions log and exp can help by adjusting the relative scales in the data so that they can be captured better by a linear model or neural network.
* Most models work best when each feature (and in regression also the target) is loosely Gaussian distributed—that is, a histogram of each feature should have something resembling the familiar “bell curve” shape. Using transformations like log and exp is a hacky but simple and efficient way to achieve this. A particularly common case when such a transformation can be helpful is when dealing with integer count data. By count data, we mean features like “how often did user A log in?” Counts are never negative, and often follow particular statistical patterns.

[back to current section](#representing-data-and-engineering-features)

### Automatic Feature Selection

* In *univariate statistics*, we compute whether there is a statistically significant relationship between each feature and the target. Then the features that are related with the highest confidence are selected. In the case of classification, this is also known as *analysis of variance* (ANOVA). A key property of these tests is that they are *univariate*, meaning that they only consider each feature individually. Consequently, a feature will be discarded if it is only informative when combined with another feature. Univariate tests are often very fast to compute, and don’t require building a model. On the other hand, they are completely independent of the model that you might want to apply after the feature selection.

```
from sklearn.feature_selection import SelectPercentile
# use f_classif (the default) and SelectPercentile to select 50% of features
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
# transform training set
X_train_selected = select.transform(X_train)
```

* *Model-based feature selection* uses a supervised machine learning model to judge the importance of each feature, and keeps only the most important ones. The supervised model that is used for feature selection doesn’t need to be the same model that is used for the final supervised modeling. The feature selection model needs to provide some measure of importance for each feature, so that they can be ranked by this measure.

```
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
```

* In *iterative feature selection*, a series of models are built, with varying numbers of features. There are two basic methods: starting with no features and adding features one by one until some stopping criterion is reached, or starting with all features and removing features one by one until some stopping criterion is reached. Because a series of models are built, these methods are much more computationally expensive than the methods we discussed previously. One particular method of this kind is *recursive feature elimination* (RFE), which starts with all features, builds a model, and discards the least important feature according to the model. Then a new model is built using all but the discarded feature, and so on until only a pre-specified number of features are left. For this to work, the model used for selection needs to provide some way to determine feature importance, as was the case for the model-based selection.

```
from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=40)
select.fit(X_train, y_train)
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)
```

[back to current section](#representing-data-and-engineering-features)

[back to top](#introduction-to-machine-learning-with-python)

## Model Evaluation and Improvement

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/05-model-evaluation-and-improvement.ipynb).

* [Cross Validation](#cross-validation)
* [Metrics for Binary Classification](#metrics-for-binary-classification)
* [Metrics for Multiclass Classification](#metrics-for-multiclass-classification)
* [Metrics in Model Selection](#metrics-in-model-selection)
* [Key Takeaways](#key-takeaways)

### Cross Validation

```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, iris.data, iris.target)
```

* Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set. In cross-validation, the data is instead split repeatedly and multiple models are trained.
* The most commonly used version of cross-validation is k-fold cross-validation, where k is a user-specified number, usually 5 or 10.
* When performing five-fold cross-validation, the data is first partitioned into five parts of (approximately) equal size, called folds.
* Next, a sequence of models is trained. The first model is trained using the first fold as the test set, and the remaining folds (2–5) are used as the training set. The model is built using the data in folds 2–5, and then the accuracy is evaluated on fold 1.
* Then another model is built, this time using fold 2 as the test set and the data in folds 1, 3, 4, and 5 as the training set. This process is repeated using folds 3, 4, and 5 as test sets.
* For each of these five splits of the data into training and test sets, we compute the accuracy. In the end, we have collected five accuracy values.

The process is illustrated below:

![cross-validation](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/cross-validation.png)

Benefits of Cross-Validation:

* First, remember that `train_test_split` performs a random split of the data. Imagine that we are “lucky” when randomly splitting the data, and all examples that are hard to classify end up in the training set. In that case, the test set will only contain “easy” examples, and our test set accuracy will be unrealistically high. Conversely, if we are “unlucky,” we might have randomly put all the hard-to-classify examples in the test set and consequently obtain an unrealistically low score. However, when using cross-validation, each example will be in the training set exactly once: each example is in one of the folds, and each fold is the test set once. Therefore, the model needs to generalize well to all of the samples in the dataset for all of the cross-validation scores (and their mean) to be high.
* Another benefit of cross-validation as compared to using a single split of the data is that we use our data more effectively. When using `train_test_split`, we usually use 75% of the data for training and 25% of the data for evaluation. When using five-fold cross-validation, in each iteration we can use four-fifths of the data (80%) to fit the model. When using 10-fold cross-validation, we can use nine-tenths of the data (90%) to fit the model. More data will usually result in more accurate models.
* The main disadvantage of cross-validation is increased computational cost. As we are now training k models instead of a single model, cross-validation will be roughly k times slower than doing a single split of the data.

[back to current section](#model-evaluation-and-improvement)

### Metrics in Model Selection

```
explicit_accuracy =  cross_val_score(SVC(), digits.data, digits.target == 9, scoring="accuracy", cv=5)
roc_auc =  cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc", cv=5)
```

* We often want to use metrics like AUC in model selection using `GridSearchCV` or `cross_val_score`. Luckily `scikit-learn` provides a very simple way to achieve this, via the scoring argument that can be used in both `GridSearchCV` and `cross_val_score`. You can simply provide a string describing the evaluation metric you want to use.
* The most important values for the `scoring` parameter for classification are `accuracy` (the default); `roc_auc` for the area under the ROC curve; `average_precision` for the area under the precision-recall curve; `f1`, `f1_macro`, `f1_micro`, and `f1_weighted` for the binary f1-score and the different weighted variants. For regression, the most commonly used values are `r2` for the R^2 score, `mean_squared_error` for mean squared error, and `mean_absolute_error` for mean absolute error.

[back to current section](#model-evaluation-and-improvement)

### Key Takeaways

* **Cross-validation** or the use of a test set allow us to evaluate a machine learning model as it will perform in the future. However, if we use the test set or cross-validation to select a model or select model parameters, we “use up” the test data, and using the same data to evaluate how well our model will do in the future will lead to overly optimistic estimates. We therefore need to resort to a split into training data for model building, validation data for model and parameter selection, and test data for model evaluation. Instead of a simple split, we can replace each of these splits with cross-validation. The most commonly used form is a training/test split for evaluation, and using cross-validation on the training set for model and parameter selection.
* It is rarely the case that the end goal of a machine learning task is building a model with a high accuracy. Make sure that the **metric** you choose to evaluate and select a model for is a good stand-in for what the model will actually be used for. In reality, classification problems rarely have balanced classes, and often false positives and false negatives have very different  consequences. Make sure you understand what these consequences are, and pick an evaluation metric accordingly.

[back to current section](#model-evaluation-and-improvement)

[back to top](#introduction-to-machine-learning-with-python)

## Algorithm Chains and Pipelines

The code in this chapter can be accessed in [this notebook](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/06-algorithm-chains-and-pipelines.ipynb).

* The *Pipeline* class is a general-purpose tool to chain together multiple processing steps in a machine learning workflow. Using pipelines allows us to encapsulate multiple steps into a single Python object that adheres to the familiar `scikit-learn` interface of `fit`, `predict`, and `transform`.
* In particular when doing model evaluation using cross-validation
and parameter selection using grid search, using the *Pipeline* class to capture all the processing steps is essential for proper evaluation. The *Pipeline* class also allows writing more succinct code, and reduces the likelihood of mistakes that can happen when building processing chains without the pipeline class (like forgetting to apply all transformers on the test set, or not applying them in the right order).

![pipeline](https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/master/EBooks/Intro-To-ML-with-Python/images/pipeline.png)

[back to top](#introduction-to-machine-learning-with-python)

## Wrapping Up

* [Approaching a Machine Learning Problem](#approaching-a-machine-learning-problem)
* [From Prototype to Production](#from-prototype-to-production)
* [Testing Production Systems](#testing-production-systems)
* [Building Your Own Estimator](#building-your-own-estimator)

### Approaching a Machine Learning Problem

* To make effective use of machine learning, we need to take a step back and consider the problem at large. First, you should think about what kind of question you want to answer. Do you want to do exploratory analysis and just see if you find something interesting in the data? Or do you already have a particular goal in mind? If you have such goal, before building a system to achieve it, you should first think about how to define and measure success, and what the impact of a successful solution would be to your overall business or research goals.
* The next steps are usually acquiring the data and building a working prototype. While trying out models, keep in mind that this is only a small part of a larger data science workflow, and model building is often part of a feedback circle of collecting new data, cleaning data, building models, and analyzing the models. Analyzing the mistakes a model makes can often be informative about what is missing in the data, what additional data could be collected, or how the task could be reformulated to make machine learning more effective. Collecting more or different data or changing the task formulation slightly might provide a much higher payoff than running endless grid searches to tune parameters.

### From Prototype to Production

* Many companies have complex infrastructure, and it is not always easy to include Python in these systems. That is not necessarily a problem. In many companies, the data analytics teams work with languages like Python and R that allow the quick testing of ideas, while production teams work with languages like Go, Scala, C++, and Java to build robust, scalable systems. Data analysis has different requirements from building live services, and so using different languages for these tasks makes sense. A relatively common solution is to reimplement the solution that was found by the analytics team inside the larger framework, using a high-performance language. This can be easier than embedding a whole library or programming language and converting from and to the different data formats.
* Regardless of whether you can use scikit-learn in a production system or not, it is important to keep in mind that production systems have different requirements from one-off analysis scripts. If an algorithm is deployed into a larger system, software engineering aspects like reliability, predictability, runtime, and memory requirements gain relevance. Simplicity is key in providing machine learning systems that perform well in these areas. Critically inspect each part of your data processing and prediction pipeline and ask yourself how much complexity each step creates, how robust each component is to changes in the data or compute infrastructure, and if the benefit of each component warrants the complexity.

### Testing Production Systems

* In this book, we covered how to evaluate algorithmic predictions based on a test set that we collected beforehand. This is known as *offline evaluation*. If your machine learning system is user-facing, this is only the first step in evaluating an algorithm, though. The next step is usually *online testing* or *live testing*, where the consequences of employing the algorithm in the overall system are evaluated. Changing the recommendations or search results users are shown by a website can drastically change their behavior and lead to unexpected consequences.
* To protect against these surprises, most user-facing services employ *A/B testing*, a form of blind user study. In A/B testing, without their knowledge a selected portion of users will be provided with a website or service using algorithm A, while the rest of the users will be provided with algorithm B. For both groups, relevant success metrics will be recorded for a set period of time. Then, the metrics of algorithm A and algorithm B will be compared, and a selection between the two approaches will be made according to these metrics. Using A/B testing enables us to evaluate the algorithms “in the wild,” which might help us to discover unexpected consequences when users are interacting with our model. Often A is a new model, while B is the established system.

### Building Your Own Estimator

```
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_paramter=1, second_parameter=2):
        # all parameters must be specified in the __init__ function
        self.first_paramter = 1
        self.second_parameter = 2

    def fit(self, X, y=None):
        # fit should only take X and y as parameters
        # even if your model is unsupervised, you need to accept a y argument!

        # Model fitting code goes here
        print("fitting the model right here")
        # fit returns self
        return self

    def transform(self, X):
        # transform takes as parameter only X

        # apply some transformation to X:
        X_transformed = X + 1
        return X_transformed
```

[back to top](#introduction-to-machine-learning-with-python)

## Setup

To run the code, you need the packages ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas`` and ``pillow``. Some of the visualizations of decision trees and neural networks structures also require ``graphviz``. The chapter on text processing also requires ``nltk`` and ``spacy``.

The easiest way to set up an environment is by installing [Anaconda](https://www.continuum.io/downloads).

### Installing packages with conda:
If you already have a Python environment set up, and you are using the ``conda`` package manager, you can get all packages by running

    conda install numpy scipy scikit-learn matplotlib pandas pillow graphviz python-graphviz

For the chapter on text processing you also need to install ``nltk`` and ``spacy``:

    conda install nltk spacy

### Installing packages with pip
If you already have a Python environment and are using pip to install packages, you need to run

    pip install numpy scipy scikit-learn matplotlib pandas pillow graphviz

You also need to install the graphiz C-library, which is easiest using a package manager. If you are using OS X and homebrew, you can ``brew install graphviz``. If you are on Ubuntu or debian, you can ``apt-get install graphviz``. Installing graphviz on Windows can be tricky and using conda / anaconda is recommended.

For the chapter on text processing you also need to install ``nltk`` and ``spacy``:

    pip install nltk spacy

### Downloading English language model
For the text processing chapter, you need to download the English language model for spacy using

    python -m spacy download en

![cover](cover.jpg)
