# Introduction to Machine Learning with Python

This repository holds the code for the book "Introduction to Machine Learning with Python" by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido). You can find details about the book on the [O'Reilly website](http://shop.oreilly.com/product/0636920030515.do).

The books requires the current stable version of scikit-learn. Most of the book can also be used with previous versions of scikit-learn, though you need to adjust the import for everything from the ``model_selection`` module, mostly ``cross_val_score``, ``train_test_split`` and ``GridSearchCV``.

This repository provides the notebooks from which the book is created, together with the ``mglearn`` library of helper functions to create figures and datasets.

All datasets are included in the repository, with the exception of the aclImdb dataset, which you can download from the page of [Andrew Maas](http://ai.stanford.edu/~amaas/data/sentiment/). See the book for details.

If you get ``ImportError: No module named mglearn`` you can try to install mglearn into your python environment using the command ``pip install mglearn`` in your terminal or ``!pip install mglearn`` in Jupyter Notebook.

Here are the chapters:

* [Introduction](#introduction)
* [Supervised Learning](#supervised-learning)
* [Unsupervised Learning and Preprocessing](#unsupervised-learning-and-preprocessing)
* [Representing Data and Engineering Features](#representing-data-and-engineering-features)
* [Model Evaluation and Improvement](#model-evaluation-and-improvement)
* [Algorithm Chains and Pipelines](#algorithm-chains-and-pipelines)
* [Working with Text Data](#working-with-text-data)
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
* [k-Nearest-Neighbors](#k-Nearest-Neighbors)
* [Linear Models](#linear-models)
* [Naive Bayes](#naive-bayes)
* [Decision Trees](#decision-trees)
* [Random Forests](#random-forests)
* [Gradient Boosting Machines](#gradient-boosting-machines)

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

[back to top](#introduction-to-machine-learning-with-python)

## Unsupervised Learning and Preprocessing

[back to top](#introduction-to-machine-learning-with-python)

## Representing Data and Engineering Features

[back to top](#introduction-to-machine-learning-with-python)

## Model Evaluation and Improvement

[back to top](#introduction-to-machine-learning-with-python)

## Algorithm Chains and Pipelines

[back to top](#introduction-to-machine-learning-with-python)

## Working with Text Data

[back to top](#introduction-to-machine-learning-with-python)

## Wrapping Up

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
