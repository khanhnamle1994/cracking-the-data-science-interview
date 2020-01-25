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

[back to current section](#data-and-sampling-distributions)

### Selection Bias

To paraphrase Yogi Berra, “If you don’t know what you’re looking for, look hard enough and you’ll find it.” Selection bias refers to the practice of selectively choosing data — consciously or unconsciously — in a way that that leads to a conclusion that is misleading or ephemeral.

* **Data snooping** — that is, extensive hunting through the data until something interesting emerges? There is a saying among statisticians: “If you torture the data long enough, sooner or later it will confess.”

Typical forms of selection bias in statistics, in addition to the vast search effect, include nonrandom sampling (see sampling bias), cherry-picking data, selection of time intervals that accentuate a particular statistical effect, and stopping an experiment when the results look “interesting.”

[back to current section](#data-and-sampling-distributions)

### Sampling Distribution of a Statistic

* **Sample statistic**: A metric calculated for a sample of data drawn from a larger population.
* **Data distribution**: The frequency distribution of individual values in a data set.
* **Sampling distribution**: The frequency distribution of a sample statistic over many samples or resamples. It is important to distinguish between the distribution of the individual data points, known as the data distribution, and the distribution of a sample statistic, known as the sampling distribution.
* **Central Limit Theorem**: The tendency of the sampling distribution to take on a normal shape as sample size rises. It says that the means drawn from multiple samples will resemble the familiar bell-shaped normal curve, even if the source population is not normally distributed, provided that the sample size is large enough and the departure of the data from normality is not too great. The central limit theorem allows normal-approximation formulas like the t-distribution to be used in calculating sampling distributions for inference — that is, confidence intervals and hypothesis tests. The central limit theorem receives a lot of attention in traditional statistics texts because it underlies the machinery of hypothesis tests and confidence intervals, which themselves consume half the space in such texts. Data scientists should be aware of this role, but, since formal hypothesis tests and confidence intervals play a small role in data science, and the bootstrap is available in any case, the central limit theorem is not so central in the practice of data science.

[back to current section](#data-and-sampling-distributions)

### The Bootstrap

One easy and effective way to estimate the sampling distribution of a statistic, or of model parameters, is to draw additional samples, with replacement, from the sample itself and recalculate the statistic or model for each resample. This procedure is called the bootstrap, and it does not necessarily involve any assumptions about the data or the sample statistic being normally distributed.

Conceptually, you can imagine the bootstrap as replicating the original sample thousands or millions of times so that you have a hypothetical population that embodies all the knowledge from your original sample (it’s just larger). You can then draw samples from this hypothetical population for the purpose of estimating a sampling distribution. See figure below.

![Figure 2-7](psds_0207.png)

In practice, it is not necessary to actually replicate the sample a huge number of times. We simply replace each observation after each draw; that is, we sample with replacement. In this way we effectively create an infinite population in which the probability of an element being drawn remains unchanged from draw to draw.

The number of iterations of the bootstrap is set somewhat arbitrarily. The more iterations you do, the more accurate the estimate of the standard error, or the confidence interval. The result from this procedure is a bootstrap set of sample statistics or estimated model parameters, which you can then examine to see how variable they are.

The bootstrap can be used with multivariate data, where the rows are sampled as units (see Figure below). A model might then be run on the bootstrapped data, for example, to estimate the stability (variability) of model parameters, or to improve predictive power. With classification and regression trees (also called decision trees), running multiple trees on bootstrap samples and then averaging their predictions (or, with classification, taking a majority vote) generally performs better than using a single tree. This process is called bagging.

![Figure 2-8](psds_0208.png)

[back to current section](#data-and-sampling-distributions)

### Confidence Intervals

Frequency tables, histograms, boxplots, and standard errors are all ways to understand the potential error in a sample estimate. Confidence intervals are another.

Confidence intervals always come with a coverage level, expressed as a (high) percentage, say 90% or 95%. One way to think of a 90% confidence interval is as follows: it is the interval that encloses the central 90% of the bootstrap sampling distribution of a sample statistic (see “The Bootstrap”). More generally, an x% confidence interval around a sample estimate should, on average, contain similar sample estimates x% of the time (when a similar sampling procedure is followed).

The percentage associated with the confidence interval is termed the level of confidence. The higher the level of confidence, the wider the interval. Also, the smaller the sample, the wider the interval (i.e., the more uncertainty). Both make sense: the more confident you want to be, and the less data you have, the wider you must make the confidence interval to be sufficiently assured of capturing the true value.

[back to current section](#data-and-sampling-distributions)

### Normal Distribution

* **Error**: The difference between a data point and a predicted or average value.
* **Standardize**: Subtract the mean and divide by the standard deviation.
* **z-score**: The result of standardizing an individual data point.
* **Standard normal**: A normal distribution with mean = 0 and standard deviation = 1.
* **QQ-Plot**: A plot to visualize how close a sample distribution is to a normal distribution.

A standard normal distribution is one in which the units on the x-axis are expressed in terms of standard deviations away from the mean. To compare data to a standard normal distribution, you subtract the mean then divide by the standard deviation; this is also called normalization or standardization (see “Standardization (Normalization, Z-Scores)”).

Converting data to z-scores (i.e., standardizing or normalizing the data) does not make the data normally distributed. It just puts the data on the same scale as the standard normal distribution, often for comparison purposes.

The normal distribution was essential to the historical development of statistics, as it permitted mathematical approximation of uncertainty and variability. While raw data is typically not normally distributed, errors often are, as are averages and totals in large samples. To convert data to z-scores, you subtract the mean of the data and divide by the standard deviation; you can then compare the data to a normal distribution.

[back to current section](#data-and-sampling-distributions)

### Long Tailed Distribution

* **Skew**: Where one tail of a distribution is longer than the other.

While the normal distribution is often appropriate and useful with respect to the distribution of errors and sample statistics, it typically does not characterize the distribution of raw data. Sometimes, the distribution is highly skewed (asymmetric), such as with income data, or the distribution can be discrete, as with binomial data. Both symmetric and asymmetric distributions may have long tails. The tails of a distribution correspond to the extreme values (small and large). Long tails, and guarding against them, are widely recognized in practical work.

[back to current section](#data-and-sampling-distributions)

### Student t-Distribution

The t-distribution is a normally shaped distribution, but a bit thicker and longer on the tails. It is used extensively in depicting distributions of sample statistics. Distributions of sample means are typically shaped like a t-distribution, and there is a family of t-distributions that differ depending on how large the sample is. The larger the sample, the more normally shaped the t-distribution becomes.

The t-distribution has been used as a reference for the distribution of a sample mean, the difference between two sample means, regression parameters, and other statistics.

[back to current section](#data-and-sampling-distributions)

### Binomial Distribution

* **Trial**: An event with a discrete outcome (e.g., a coin flip).
* **Success**: The outcome of interest for a trial. Synonyms “1” (as opposed to “0”).
* **Binomial**: Having two outcomes. Synonyms yes/no, 0/1, binary.
* **Binomial trial**: A trial with two outcomes. Synonym Bernoulli trial.
* **Binomial distribution**: Distribution of number of successes in x trials. Synonym Bernoulli distribution.

The binomial distribution is the frequency distribution of the number of successes (x) in a given number of trials (n) with specified probability (p) of success in each trial. There is a family of binomial distributions, depending on the values of x, n, and p. The binomial distribution would answer a question like: If the probability of a click converting to a sale is 0.02, what is the probability of observing 0 sales in 200 clicks?

* Mean: `n * p`
* Variance: `n * p(1 - p)`

With large n, and provided p is not too close to 0 or 1, the binomial distribution can be approximated by the normal distribution.

[back to current section](#data-and-sampling-distributions)

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
