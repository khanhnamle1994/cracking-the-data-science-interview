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
* [Generalization, Overfitting, and Underfitting](#generalization-overfitting-underfitting)
* [k-Nearest-Neighbors](#k-Nearest-Neighbors)
* [Linear Models](#linear-models)
* [Naive Bayes](#naive-bayes)
* [Decision Trees](#decision-trees)
* [Random Forests](#random-forests)
* [Gradient Boosting Machines](#gradient-boosting-machines)

### Classification and Regression

* In classification, the goal is to predict a class label, which is a choice from a predefined list of possibilities. Classification is sometimes separated into binary classification, which is the special case of distinguishing between exactly two classes, and multiclass classification, which is classification between more than two classes.
* For regression tasks, the goal is to predict a continuous number, or a floating-point number in programming terms (or real number in mathematical terms). Predicting a person’s annual income from their education, their age, and where they live is an example of a regression task. When predicting income, the predicted value is an amount, and can be any number in a given range. Another example of a regression task is predicting the yield of a corn farm given attributes such as previous yields, weather, and number of employees working on the farm. The yield again can be an arbitrary number.

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
