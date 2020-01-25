# Practical Statistics For Data Scientists

Code associated with the book "[Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.amazon.com/Practical-Statistics-Data-Scientists-Essential/dp/1491952962)"

The scripts are stored by chapter and replicate most of the figures and code snippets.

**HOW TO GET THE DATA**:
* Run R script:
* The data is not saved on github and you will need to download the data.
* You can do this in R using the sript `src/download_data.r`. This will copy the data into the data directory ~/statistics-for-data-scientists/data.

**Manual download**:
Alternatively, you can manually download the files from https://drive.google.com/drive/folders/0B98qpkK5EJemYnJ1ajA1ZVJwMzg or from https://www.dropbox.com/sh/clb5aiswr7ar0ci/AABBNwTcTNey2ipoSw_kH5gra?dl=0

**IMPORTANT NOTE**:
The scripts all assume that you have cloned the repository into the top level home directory (~/). If you save the repository elsewhere, you will need to edit the line:
```
  PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')
```
to point to the appropriate directory in all of the scripts.


Here are the chapters:

* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Data and Sampling Distributions](#data-and-sampling-distributions)
* [Statistical Experiments and Significance Testing](#statistical-experiments-and-significance-testing)
* [Regression and Prediction](#regression-and-prediction)
* [Classification](#classification)
* [Statistical Machine Learning](#statistical-machine-learning)
* [Unsupervised Learning](#unsupervised-learning)

## Exploratory Data Analysis

Here are the sections:

* [Elements of Structured Data](#elements-of-structured-data)
* [Rectangular Data](#rectangular-data)
* [Estimates of Location](#estimates-of-location)
* [Estimates of Variability](#estimates-of-variability)
* [Exploring the Data Distribution](#exploring-the-data-distribution)
* [Exploring Binary and Categorical Data](#exploring-binary-and-categorical-data)
* [Correlation](#correlation)
* [Exploring Two or More Variables](#exploring-two-or-more-variables)

### Elements of Structured Data

* Data is typically classified in software by type.
* Data types include continuous, discrete, categorical, and ordinal.
* Data typing in software acts as a signal to the software on how to process the data.

### Rectangular Data

* The basic data structure in data science is a rectangular matrix in which rows are records and columns are variables/features.
* Terminology can be confusing; there are a variety of synonyms arising from the different disciplines that contribute to data science (statistics, computer science, and information technology).

### Estimates of Location

* The basic metric for location is the mean, but it can be sensitive to extreme values (outliers).
* Other metrics (median, trimmed mean) are more robust.

### Estimates of Variability

* The variance and standard deviation are the most widespread and routinely reported statistics of variability.
* Both are sensitive to outliers.
* More robust metrics include mean and median absolute deviations from the mean and percentiles (quantiles).

### Exploring the Data Distribution

* A frequency histogram plots frequency counts on the y-axis and variable values on the x-axis; it gives a sense of the distribution of the data at a glance.
* A frequency table is a tabular version of the frequency counts found in a histogram.
* A boxplot - with the top and bottom of the box at the 75th and 25th percentiles, respectively - also gives a quick sense of the distribution of the data; it is often used in side-by-side displays to compare distributions.
* A density plot is a smoothed version of a histogram; it requires a function to estimate a plot based on the data (multiple estimates are possible, of course).

### Exploring Binary and Categorical Data

* Categorical data is typically summed up in proportions, and can be visualized in a bar chart.
* Categories might represent distinct things (apples and oranges, male and female), levels of a factor variable (low, medium, and high), or numeric data that has been binned.
* Expected value is the sum of values times their probability of occurrence, often used to sum up factor variable levels.

### Correlation

* The correlation coefficient measures the extent to which two variables are associated with one another.
* When high values of v1 go with high values of v2, v1 and v2 are positively associated.
* When high values of v1 are associated with low values of v2, v1 and v2 are negatively associated.
* The correlation coefficient is a standardized metric so that it always ranges from -1 to +1.
* A correlation coefficient of 0 indicates no correlation, but be aware that random arrangements of data will produce both positive and negative values for the correlation coefficient just by chance.

### Exploring Two or More Variables

* Hexagonal binning and contour plots are useful tools that permit graphical examination of 2 numeric variables at a time, without being overwhelmed by huge amounts of data.
* Contingency tables are the standard tool for looking at the counts of 2 categorical variables.
* Boxplots and violin plots allow you to plot a numeric variable against a categorical variable.

[back to top](#practical-statistics-for-data-scientists)

## Data and Sampling Distributions

A popular misconception holds that the era of big data means the end of a need for sampling. In fact, the proliferation of data of varying quality and relevance reinforces the need for sampling as a tool to work efficiently with a variety of data and to minimize bias. Even in a big data project, predictive models are typically developed and piloted with samples. Samples are also used in tests of various sorts (e.g., pricing, web treatments).

Figure below shows a schematic that underpins the concepts in this chapter. The lefthand side represents a population that, in statistics, is assumed to follow an underlying but *unknown* distribution. The only thing available is the *sample* data and its empirical distribution, shown on the righthand side. To get from the lefthand side to the righthand side, a *sampling* procedure is used (represented by an arrow). Traditional statistics focused very much on the lefthand side, using theory based on strong assumptions about the population. Modern statistics has moved to the righthand side, where such assumptions are not needed.

![Figure 2-1](psds_0201.png)

Here are the sections:

* [Random Sampling and Sample Bias](#random-sampling-and-sample-bias)
* [Selection Bias](#selection-bias)
* [Sampling Distribution of a Statistic](#sampling-distribution-of-a-statistic)
* [The Bootstrap](#the-bootstrap)
* [Confidence Intervals](#confidence-intervals)
* [Normal Distribution](#normal-distribution)
* [Long Tailed Distribution](#long-tailed-distribution)
* [Student t-Distribution](#student-t-distribution)
* [Binomial Distribution](#binomial-distribution)
* [Poisson and Related Distributions](#poisson-and-related-distributions)

### Random Sampling and Sample Bias

* Even in the era of big data, random sampling remains an important arrow in the data scientist's quiver.
* Bias occurs when measurements or observations are systematically in error because they are not representative of the full population.
* Data quality is often more important than data quantity, and random sampling can reduce bias and facilitate quality improvement that would be prohibitively expensive.

### Selection Bias

* Specifying a hypothesis, then collecting data following randomization and random sampling principles, ensures against bias.
* All other forms of data analysis run the risk of bias resulting from the data collection/analysis process (repeated running of models in data mining, data snooping in research, and after-the-fact selection of interesting events).

### Sampling Distribution of a Statistic

* The frequency distribution of a sample statistic tells us how that metric would turn out differently from sample to sample.
* This sampling distribution can be estimated via the bootstrap, or via formulas that rely on the central limit theorem.
* A key metric that sums up the variability of a sample statistic is its standard error.

### The Bootstrap

* The bootstrap (sampling with replacement from a data set) is a powerful tool for assessing the variability of a sample statistic.
* The bootstrap can be applied in similar fashion in a wide variety of circumstances, without extensive study of mathematical approximations to sampling distributions.
* It also allows us to estimate sampling distributions for statistics where no mathematical approximation has been developed.
* When applied to predictive models, aggregating multiple bootstrap sample predictions (bagging) outperforms the use of a single model.

### Confidence Intervals

* Confidence intervals are the typical way to present estimates as an interval range.
* The more data you have, the less variable a sample estimate will be.
* The lower the level of confidence you can tolerate, the narrower the confidence interval will be.
* The bootstrap is an effective way to construct confidence intervals.

### Normal Distribution

* The normal distribution was essential to the historical development of statistics, as it permitted mathematical approximation of uncertainty and variability.
* While raw data is typically not normally distributed, errors often are, as are averages and totals in large samples.
* To convert data to z-scores, you subtract the mean of the data and divide by the standard deviation; you can then compare the data to a normal distribution.

### Long Tailed Distribution

* Most data is not normally distributed.
* Assuming a normal distribution can lead to under-estimation of extreme events ("black swans").

### Student t-Distribution

* The t-distribution is actually a family of distributions resembling the normal distribution, but with thicker tails.
* It is widely used as a reference basis for the distribution of sample means, differences between two sample means, regression parameters, and more.

### Binomial Distribution

* Binomial outcomes are important to model, since they represent, among other things, fundamental decision (buy or don't buy, click or don't click, survive or die...)
* A binomial trial is an experiment with two possible outcomes: one with probability p and the other with probability 1 - p.
* With large n, and provided p is not too close to 0 or 1, the binomial distribution can be approximated by the normal distribution.

### Poisson and Related Distributions

Many processes produce events randomly at a given overall rate — visitors arriving at a website, cars arriving at a toll plaza (events spread over time), imperfections in a square meter of fabric, or typos per 100 lines of code (events spread over space).

It is useful when addressing queuing questions like “How much capacity do we need to be 95% sure of fully processing the internet traffic that arrives on a server in any 5- second period?”

* **Lambda**: The rate (per unit of time or space) at which events occur. This is also the variance.
* **Poisson distribution**: The frequency distribution of the number of events in sampled units of time or space. For events that occur at a constant rate, the number of events per unit of time or space can be modeled as a Poisson distribution. In this scenario, you can also model the time or distance between one event and the next as an exponential distribution, see below.
* **Exponential distribution**: The frequency distribution of the time or distance from one event to the next event. Using the same parameter that we used in the Poisson distribution, we can also model the distribution of the time between events: time between visits to a website or between cars arriving at a toll plaza. It is also used in engineering to model time to failure, and in process management to model, for example, the time required per service call.
* **Estimating the failure rate**: In many applications, the event rate, , is known or can be estimated from prior data. However, for rare events, this is not necessarily so. Aircraft engine failure for example. If there is some data but not enough to provide a precise, reliable estimate of the rate, a goodness-of-fit test (see “Chi-Square Test”) can be applied to various rates to determine how well they fit the observed data.
* **Weibull distribution**: A generalized version of the exponential, in which the event rate is allowed to shift over time. So a changing event rate over time (e.g., an increasing probability of device failure) can be modeled with the Weibull distribution.

[back to current section](#data-and-sampling-distributions)

[back to top](#practical-statistics-for-data-scientists)

## Statistical Experiments and Significance Testing

Design of experiments is a cornerstone of the practice of statistics, with applications in virtually all areas of research. The goal is to design an experiment in order to confirm or reject a hypothesis. Data scientists are faced with the need to conduct continual experiments, particularly regarding user interface and product marketing. This chapter reviews traditional experimental design and discusses some common challenges in data science. It also covers some oft-cited concepts in statistical inference and explains their meaning and relevance (or lack of relevance) to data science.

Whenever you see references to statistical significance, t-tests, or p-values, it is typically in the context of the classical statistical inference “pipeline” (see Figure below). This process starts with a hypothesis (“drug A is better than the existing standard drug,” “price A is more profitable than the existing price B”). An experiment (it might be an A/B test) is designed to test the hypothesis—designed in such a way that, hopefully, will deliver conclusive results. The data is collected and analyzed, and then a conclusion is drawn. The term *inference* reflects the intention to apply the experiment results, which involve a limited set of data, to a larger process or population.

![Figure 3-1](psds_03in01.png)

Here are the sections:

* [A/B Testing](#ab-testing)
* [Hypothesis Test](#hypothesis-test)
* [Resampling](#resampling)
* [Statistical Significance and P-Values](#statistical-significance-and-p-values)
* [t-Tests](#t-tests)
* [Multiple Testing](#multiple-testing)
* [Degrees of Freedom](#degrees-of-freedom)
* [ANOVA](#ANOVA)
* [Chi-Square Test](#chi-square-test)
* [Multi-Arm Bandit Algorithm](#multi-arm-bandit-algorithm)
* [Power and Sample Size](#power-and-sample-size)

## Regression and Prediction

## Classification

## Statistical Machine Learning

## Unsupervised Learning
