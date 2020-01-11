# Introduction to Machine Learning with Python

This repository holds the code for the book "Introduction to Machine Learning with Python" by [Andreas Mueller](http://amueller.io) and [Sarah Guido](https://twitter.com/sarah_guido). You can find details about the book on the [O'Reilly website](http://shop.oreilly.com/product/0636920030515.do).

The books requires the current stable version of scikit-learn. Most of the book can also be used with previous versions of scikit-learn, though you need to adjust the import for everything from the ``model_selection`` module, mostly ``cross_val_score``, ``train_test_split`` and ``GridSearchCV``.


This repository provides the notebooks from which the book is created, together with the ``mglearn`` library of helper functions to create figures and datasets.

All datasets are included in the repository, with the exception of the aclImdb dataset, which you can download from the page of [Andrew Maas](http://ai.stanford.edu/~amaas/data/sentiment/). See the book for details.

If you get ``ImportError: No module named mglearn`` you can try to install mglearn into your python environment using the command ``pip install mglearn`` in your terminal or ``!pip install mglearn`` in Jupyter Notebook.

## Setup

To run the code, you need the packages ``numpy``, ``scipy``, ``scikit-learn``, ``matplotlib``, ``pandas`` and ``pillow``. Some of the visualizations of decision trees and neural networks structures also require ``graphviz``. The chapter on text processing also requirs ``nltk`` and ``spacy``.

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
