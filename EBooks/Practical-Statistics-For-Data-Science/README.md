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

* [AB Testing](#ab-testing)
* [Resampling](#resampling)
* [Statistical Significance and P-Values](#statistical-significance-and-p-values)
* [t-Tests](#t-tests)
* [Multiple Testing](#multiple-testing)
* [Degrees of Freedom](#degrees-of-freedom)
* [ANOVA](#ANOVA)
* [Chi-Square Test](#chi-square-test)
* [Multi-Arm Bandit Algorithm](#multi-arm-bandit-algorithm)
* [Power and Sample Size](#power-and-sample-size)

### AB Testing

* **Treatment**: Something (drug, price, web headline) to which a subject is exposed.
* **Treatment group**: A group of subjects exposed to a specific treatment.
* **Control group**: A group of subjects exposed to no (or standard) treatment.
* **Randomization**: The process of randomly assigning subjects to treatments.
* **Subjects**: The items (web visitors, patients, etc.) that are exposed to treatments.
* **Test statistic**: The metric used to measure the effect of the treatment.

[back to current section](#statistical-experiments-and-significance-testing)

### Resampling

Permutation tests are useful heuristic procedures for exploring the role of random variation. They are relatively easy to code, interpret and explain, and they offer a useful detour around the formalism and “false determinism” of formula-based statistics. One virtue of resampling, in contrast to formula approaches, is that it comes much closer to a “one size fits all” approach to inference. Data can be numeric or binary. Sample sizes can be the same or different. Assumptions about normally distributed data are not needed.

There are two main types of resampling procedures: the bootstrap and permutation tests. The bootstrap is used to assess the reliability of an estimate. Permutation tests are used to test hypotheses, typically involving two or more groups.

[back to current section](#statistical-experiments-and-significance-testing)

### Statistical Significance and P-Values

* **P-value**: Given a chance model that embodies the null hypothesis, the p-value is the probability of obtaining results as unusual or extreme as the observed results.
* **Alpha**: The probability threshold of “unusualness” that chance results must surpass, for actual outcomes to be deemed statistically significant. The probability question being answered is not “what is the probability that this happened by chance?” but rather “given a chance model, what is the probability of a result this extreme?” We then deduce backward about the appropriateness of the chance model, but that judgment does not carry a probability. This point has been the subject of much confusion.
* **Type 1 error**: Mistakenly concluding an effect is real (when it is due to chance).
* **Type 2 error**: Mistakenly concluding an effect is due to chance (when it is real).

[back to current section](#statistical-experiments-and-significance-testing)

### t-Tests

Before the advent of computers, resampling tests were not practical and statisticians used standard reference distributions.  A test statistic could then be standardized and compared to the reference distribution. One such widely used standardized statistic is the t-statistic

[back to current section](#statistical-experiments-and-significance-testing)

### Multiple Testing

Problem of **overfitting** in data mining, or “**fitting the model to the noise.**” The more variables you add, or the more models you run, the greater the probability that something will emerge as “significant” just by chance. Multiplicity in a research study or data mining project (multiple comparisons, many variables, many models, etc.) increases the risk of concluding that something is significant just by chance.

* For predictive modeling, the risk of getting an illusory model whose apparent efficacy is largely a product of random chance is mitigated by cross-validation, and use of a holdout sample.
* For other procedures without a labeled holdout set to check the model, you must rely on: (1) Awareness that the more you query and manipulate the data, the greater the role that chance might play; and (2) Resampling and simulation heuristics to provide random chance benchmarks against which observed results can be compared.

[back to current section](#statistical-experiments-and-significance-testing)

### Degrees of Freedom

The concept is applied to statistics calculated from sample data, and refers to the number of values free to vary. For example, if you know the mean for a sample of 10 values, and you also know 9 of the values, you also know the 10th value. Only 9 are free to vary.

When you use a sample to estimate the variance for a population, you will end up with an estimate that is slightly biased downward if you use n in the denominator. If you use n – 1 in the denominator, the estimate will be free of that bias.

[back to current section](#statistical-experiments-and-significance-testing)

### ANOVA

* **Pairwise comparison**: A hypothesis test (e.g., of means) between two groups among multiple groups.
* **Omnibus test**: A single hypothesis test of the overall variance among multiple group means.
* **Decomposition of variance**: Separation of components. contributing to an individual value (e.g., from the overall average, from a treatment mean, and from a residual error).
* **F-statistic**: A standardized statistic that measures the extent to which differences among group means exceeds what might be expected in a chance model.
* **SS**: “Sum of squares,” referring to deviations from some average value.

[back to current section](#statistical-experiments-and-significance-testing)

### Chi-Square Test

* **Chi-square statistic**: A measure of the extent to which some observed data departs from expectation.
* **Expectation or expected**: How we would expect the data to turn out under some assumption, typically the null hypothesis.

[back to current section](#statistical-experiments-and-significance-testing)

### Multi-Arm Bandit Algorithm

Multi-arm bandits offer an approach to testing, especially web testing, that allows explicit optimization and more rapid decision making than the traditional statistical approach to designing experiments.

**How it works**

Your goal is to win as much money as possible, and more specifically, to identify and settle on the winning arm sooner rather than later. The challenge is that you don’t know at what rate the arms pay out — you only know the results of pulling the arm. Suppose each “win” is for the same amount, no matter which arm. What differs is the probability of a win. Suppose further that you initially try each arm 50 times and get the following results:

* **Arm A**: 10 wins out of 50
* **Arm B**: 2 win out of 50
* **Arm C**: 4 wins out of 50

We start pulling A more often, to take advantage of its apparent superiority, but we don’t abandon B and C. We just pull them less often. If A continues to outperform, we continue to shift resources (pulls) away from B and C and pull A more often. If, on the other hand, C starts to do better, and A starts to do worse, we can shift pulls from A back to C. If one of them turns out to be superior to A and this was hidden in the initial trial due to chance, it now has an opportunity to emerge with further testing.

A more sophisticated algorithm uses “**Thompson’s sampling.**” This procedure “samples” (pulls a bandit arm) at each stage to maximize the probability of choosing the best arm. Of course you don’t know which is the best arm — that’s the whole problem! — but as you observe the payoff with each successive draw, you gain more information. Thompson’s sampling uses a Bayesian approach: some prior distribution of rewards is assumed initially, using what is called a beta distribution (this is a common mechanism for specifying prior information in a Bayesian problem). As information accumulates from each draw, this information can be updated, allowing the selection of the next draw to be better optimized as far as choosing the right arm.

[back to current section](#statistical-experiments-and-significance-testing)

### Power and Sample Size

* **Effect size**: The minimum size of the effect that you hope to be able to detect in a statistical test, such as “a 20% improvement in click rates”.
* **Power**: The probability of detecting a given effect size with a given sample size.
* **Significance level**: The statistical significance level at which the test will be conducted.

[back to current section](#statistical-experiments-and-significance-testing)

[back to top](#practical-statistics-for-data-scientists)

## Regression and Prediction

Here are the sections:

* [Simple Linear Regression](#simple-linear-regression)
* [Multiple Linear Regression](#multiple-linear-regression)
* [Prediction Using Regression](#prediction-using-regression)
* [Factor Variables In Regression](#factor-variables-in-regression)
* [Interpreting The Regression Equation](#interpreting-the-regression-equation)
* [Testing The Assumptions Regression Diagnostics](#testing-the-assumptions-regression-diagnostics)
* [Polynomial And Spline Regression](#polynomial-and-spline-regression)

### Simple Linear Regression
Simple linear regression models the relationship between the magnitude of one variable and that of a second — for example, as X increases, Y also increases. Or as X increases, Y decreases. Correlation is another way to measure how two variables are related. The difference is that while correlation measures the strength of an association between two variables, regression quantifies the nature of the relationship

* **Response**: The variable we are trying to predict. Synonyms: dependent variable, Y-variable, target, outcome
* **Independent**: variable The variable used to predict the response. Synonyms: independent variable, X-variable, feature, attribute
* **Record**: The vector of predictor and outcome values for a specific individual or case. Synonyms: row, case, instance, example
* **Intercept**: The intercept of the regression line — that is, the predicted value when X=0. Synonyms: b0 and Beta0
* **Regression coefficient**: The slope of the regression line. Synonyms: slope, b1, beta1, parameter estimates, weights
* **Fitted values**: The estimates Y-hat_{i} obtained from the regression line. Synonyms: predicted values
* **Residuals**: The difference between the observed values and the fitted values. Synonyms: errors
* **Least squares**: The method of fitting a regression by minimizing the sum of squared residuals. Synonyms: ordinary least squares

[back to current section](#regression-and-prediction)

### Multiple Linear Regression

Instead of a line, we now have a linear model — the relationship between eachcoefficient and its variable (feature) is linear.

* **Root mean squared error**: The square root of the average squared error of the regression (this is the most widely used metric to compare regression models). Synonyms: RMSE
* **Residual standard error**: The same as the root mean squared error, but adjusted for degrees of freedom. Synonyms RSE
* **R-squared**: The proportion of variance explained by the model, from 0 to 1. Synonyms: coefficient of determination,
* **t-statistic**: The coefficient for a predictor, divided by the standard error of the coefficient, giving a metric to compare the importance of variables in the model.
* **Weighted regression**: Regression with the records having different weights

**Cross Validation**

Classic statistical regression metrics (R2, F-statistics, and p-values) are all “in-sample” metrics — they are applied to the same data that was used to fit themodel. Intuitively, you can see that it would make a lot of sense to set aside some of the original data, not use it to fit the model, and then apply the model to the set-aside (holdout) data to see how well it does. Normally, you would use a majority of the data to fit the model, and use a smaller portion to test the model.

Cross-validation extends the idea of a holdout sample to multiple sequential holdout samples. The algorithm for basic k-fold cross-validation is as follows:
1. Set aside 1/k of the data as a holdout sample.
2. Train the model on the remaining data.
3. Apply (score) the model to the 1/k holdout, and record needed model assessment metrics.
4. Restore the first 1/k of the data, and set aside the next 1/k (excluding any records that got picked the first time).
5. Repeat steps 2 and 3.
6. Repeat until each record has been used in the holdout portion.
7. Average or otherwise combine the model assessment metrics.

The division of the data into the training sample and the holdout sample is also called a fold.

**Model Selection and Stepwise Regression**

In some problems, many variables could be used as predictors in a regression. Adding more variables, however, does not necessarily mean we have a better model. Statisticians use the principle of *Occam’s razor* to guide the choice of a model: all things being equal, a simpler model should be used in preference to amore complicated model.

Including additional variables always reduces RMSE and increases R^2. Hence, these are not appropriate to help guide the model choice. In the 1970s, Hirotugu Akaike, the eminent Japanese statistician, developed a metric called AIC (Akaike’s Information Criteria) that penalizes adding terms to a model. In the case of regression, AIC has the form:

```
AIC = 2P + n log(RSS/n)
```

Where P is the number of variables and n is the number of records. The goal is to find the model that minimizes AIC; models with k more extra variables are penalized by 2k.

How do we find the model that minimizes AIC? One approach is to search through all possible models, called *all subset regression*. This is computationally expensive and is not feasible for problems with large data and many variables. An attractive alternative is to use *stepwise regression*, which successively adds and drops predictors to find a model that lowers AIC.

Simpler yet are *forward selection* and *backward selection*. In forward selection, you start with no predictors and add them one-by-one, at each step adding the predictor that has the largest contribution to R^2, stopping when the contribution is no longer statistically significant. In backward selection, or *backward elimination*, you start with the full model and take away predictors that are not statistically significant until you are left with a model in which all predictors are statistically significant.

Stepwise regression and all subset regression are in-sample methods to assess and tune models. This means the model selection is possibly subject to overfitting and may not perform as well when applied to new data. One common approach to avoid this is to use cross-validation to validate the models.

[back to current section](#regression-and-prediction)

### Prediction Using Regression
The primary purpose of regression in data science is prediction. This is useful to keep in mind, since regression, being an old and established statistical method, comes with baggage that is more relevant to its traditional explanatory modeling role than to prediction.

* **Prediction interval**: An uncertainty interval around an individual predicted value.
* **Extrapolation**: Extension of a model beyond the range of the data used to fit it.

**Confidence and Prediction Intervals**
Useful metrics are confidence intervals, which are uncertainty intervals placed around regression coefficients and predictions. An easy way to understand this is via the bootstrap.

The most common regression confidence intervals encountered in software output are those for regression parameters (coefficients). Here is a bootstrap algorithm for generating confidence intervals for regression parameters (coefficients) for a data set with P predictors and n records (rows):
1. Consider each row (including outcome variable) as a single “ticket” and place all the n tickets in a box.
2. Draw a ticket at random, record the values, and replace it in the box.
3. Repeat step 2 n times; you now have one bootstrap resample.
4. Fit a regression to the bootstrap sample, and record the estimated coefficients.
5. Repeat steps 2 through 4, say, 1,000 times.
6. You now have 1,000 bootstrap values for each coefficient; find the appropriate percentiles for each one (e.g., 5th and 95th for a 90% confidence interval).

Of greater interest to data scientists are intervals around predicted y values ( ). The uncertainty around comes from two sources:
* Uncertainty about what the relevant predictor variables and their coefficients
* Additional error inherent in individual data points

The individual data point error can be thought of as follows: even if we knew for certain what the regression equation was (e.g., if we had a huge number of records to fit it), the actual outcome values for a given set of predictor values will vary.

We can model this individual error with the residuals from the fitted values. The bootstrap algorithm for modeling both the regression model error and the individual data point error would look as follows:
1. Take a bootstrap sample from the data (spelled out in greater detail earlier).
2. Fit the regression, and predict the new value.
3. Take a single residual at random from the original regression fit, add it to the predicted value, and record the result.
4. Repeat steps 1 through 3, say, 1,000 times.
5. Find the 2.5th and the 97.5th percentiles of the results.

[back to current section](#regression-and-prediction)

### Factor Variables In Regression
Factor variables, also termed categorical variables, take on a limited number of discrete values. For example, a loan purpose can be “debt consolidation,” “wedding,” “car,” and so on. The binary (yes/no) variable, also called an indicator variable, is a special case of a factor variable. Regression requires numerical inputs, so factor variables need to be recoded to use in the model. The most common approach is to convert a variable into a set of binary dummy variables.

* **Dummy variables**: Binary 0–1 variables derived by recoding factor data for use in regression and other models.
* **Reference coding**: The most common type of coding used by statisticians, in which one level of a factor is used as a reference and other factors are compared to that level. Synonyms: treatment coding
* **One hot encoder**: A common type of coding used in the machine learning community in which all factors levels are retained. While useful for certain machine learning algorithms, this approach is not appropriate for multiple linear regression.
* **Deviation coding**: A type of coding that compares each level against the overall mean as opposed to the reference level. Synonyms: sum contrasts

[back to current section](#regression-and-prediction)

### Interpreting The Regression Equation

* **Correlated variables**: When the predictor variables are highly correlated, it is difficult to interpret the individual coefficients.
* **Multicollinearity**: When the predictor variables have perfect, or near-perfect, correlation, the regression can be unstable or impossible to compute. Synonyms: collinearity
* **Confounding variables**: An important predictor that, when omitted, leads to spurious relationships in a regression equation.
* **Main effects**: The relationship between a predictor and the outcome variable, independent from other variables.
* **Interactions**: An interdependent relationship between two or more predictors and the response.

**Correlated Predictors**
In multiple regression, the predictor variables are often correlated with each other. The coefficient for Bedrooms is negative! This implies that adding a bedroom to a house will reduce its value. How can this be? This is because the predictor variables are correlated: larger houses tend to have more bedrooms, and it is the size that drives house value, not the number of bedrooms. Consider two homes of the exact same size: it is reasonable to expect that a home with more, but smaller, bedrooms would be considered less desirable.

Having correlated predictors can make it difficult to interpret the sign and value of regression coefficients

The update function can be used to add or remove variables from a model. Now the coefficient for bedrooms is positive in our ex — in line with what we would expect (though it is really acting as a proxy for house size, now that those variables have been removed).

Correlated variables are only one issue with interpreting regression coefficients.

**Multicollinearity**
An extreme case of correlated variables produces multicollinearity — a condition in which there is redundance among the predictor variables. Perfect multicollinearity occurs when one predictor variable can be expressed as a linear combination of others. Multicollinearity occurs when:
* A variable is included multiple times by error.
* P dummies, instead of P – 1 dummies, are created from a factor variable (see “Factor Variables in Regression”).
* Two variables are nearly perfectly correlated with one another.

Multicollinearity in regression must be addressed — variables should be removed until the multicollinearity is gone. A regression does not have a well-defined solution in the presence of perfect multicollinearity.

**Confounding Variables**
With correlated variables, the problem is one of commission: including different variables that have a similar predictive relationship with the response. With confounding variables, the problem is one of omission: an important variable is not included in the regression equation. Naive interpretation of the equation coefficients can lead to invalid conclusions.

**Model Selection with Interaction Terms**
In problems involving many variables, it can be challenging to decide which interaction terms should be included in the model. Several different approaches are commonly taken:
* In some problems, prior knowledge and intuition can guide the choice of which interaction terms to include in the model.
* Stepwise selection (see “Model Selection and Stepwise Regression”) can be used to sift through the various models.
* Penalized regression can automatically fit to a large set of possible interaction terms.
* Perhaps the most common approach is the use tree models, as well as their descendents, random forest and gradient boosted trees. This class of models automatically searches for optimal interaction terms; see “Tree Models”.

[back to current section](#regression-and-prediction)

### Testing The Assumptions Regression Diagnostics

* **Standardized residuals**: Residuals divided by the standard error of the residuals.
* **Outliers**: Records (or outcome values) that are distant from the rest of the data (or the predicted outcome).
* **Influential value**: A value or record whose presence or absence makes a big difference in the regression equation.
* **Leverage**: The degree of influence that a single record has on a regression equation. Synonyms: hat-value
* **Non-normal residuals**: Non-normally distributed residuals can invalidate some technical requirements of regression, but are usually not a concern in data science.
* **Heteroskedasticity**: When some ranges of the outcome experience residuals with higher variance (may indicate a predictor missing from the equation).
* **Partial residual plots**: A diagnostic plot to illuminate the relationship between the outcome variable and a single predictor. Synonyms: added variables plot.

[back to current section](#regression-and-prediction)

### Polynomial And Spline Regression

* **Polynomial regression**: Adds polynomial terms (squares, cubes, etc.) to a regression.
* **Spline regression**: Fitting a smooth curve with a series of polynomial segments.
* **Knots**: Values that separate spline segments.
* **Generalized additive models**: Spline models with automated selection of knots. Synonyms: GAM.

[back to current section](#regression-and-prediction)

[back to top](#practical-statistics-for-data-scientists)

## Classification

* [Naive Bayes](#naive-bayes)
* [Discriminant Analysis](#discriminant-analysis)
* [Logistic Regression](#logistic-regression)
* [Evaluating Classification Models](#evaluating-classification-models)
* [Strategies for Imbalanced Data](#strategies-for-imbalanced-data)

## Statistical Machine Learning

* [K Nearest Neighbors](#k-nearest-neighbors)
* [Tree Models](#tree-models)
* [Bagging and Random Forest](#bagging-and-random-forest)
* [Boosting](#boosting)

### K Nearest Neighbors

- kNN classifies a record by assigning it to the class that similar records belong to.
- Similarity (distance) is determined by Euclidean distance or Manhattan distance.
  - Euclidean distance corresponds to the straight-line distance between two points.
  - Manhattan distance is the distance between two points traversed in a single direction at a time.
- The number of nearest neighbors to compare a record to, K, is determined by how well the algorithm performs on training data, using different values for K.
- Typically, the predictor variables are standardized so that variables of large scale do not dominate the distance metric.
  - Standardization / normalization puts all variables on similar scales by subtracting the mean and dividing by the standard deviation.
  - These are called z-scores, stated in terms of "standard deviations away from the mean."
- kNN is often used as a first stage in predictive modeling, and the predicted value is added back into the data as a predictor for second-stage modeling.

[back to current section](#statistical-machine-learning)

### Tree Models

- Decision trees produce a set of rules to classify or predict an outcome.
- The rules correspond to successive partitioning of the data into subpartitions.
- Each partition, or split, references a specific value of a predictor variable and divides the data into records where that predictor value is above or below that split value.
- At each stage, the tree algorithm chooses the split that minimizes the outcome impurity within each subpartition.
- When no further splits can be made, the tree is fully grown and each terminal node, or leaf, has records of a single class; new cases following that rule (split) path would be assigned that class.
- A fully grown tree overfits the data and must be pruned back so that it captures signal and not noise.
- Multiple-tree algorithms like random forests and boostred trees yield better predictive performance, but lose the rule-based communicative power of single trees.

[back to current section](#statistical-machine-learning)

### Bagging and Random Forest

- Ensemble models improve model accuracy by combining the results from many models. The simple version is as follows:
  - Develop a predictive model and record the predictions for a given dataset.
  -  Repeat for multiple models, on the same data.
  - For each record to be predicted, take an average (or a weighted average, or a majority vote) of the predictions.
- Bagging is a particular type of ensemble model based on fitting many models to bootstrapped samples of the data and averaging the models.
- Random forest is a special type of bagging applied to decision trees. In addition to resampling the data, the random forest algorithm samples the predictor variables when splitting the trees.
- A useful output from the random forest is a measure of variable importance that ranks the predictors in terms of their contribution to model accuracy.
- The random forest has a set of hyper-parameters that should be tuned using cross-validation to avoid overfitting.
  - *nodesize*: The minimum size for terminal nodes (leaves in the tree). The default is 1 for classification and 5 for regression.
  - *maxnode*: The maximum number of nodes in each decision tree. By default, there is no limit and the largest tree will be fit subject to the constraints of *nodesize*.

[back to current section](#statistical-machine-learning)

### Boosting

- Boosting is a class of ensemble models based on fitting a sequence of models, with more weight given to records with large errors in successive rounds.
- Stochastic gradient boosting is the most general type of boosting and offers the best performance. The most common form of stochastic gradient boosting uses tree models.
- XGBoost is a popular and computationally efficient software package for stochastic gradient boosting: it is available in all common languages used in data science.
- Boosting is prone to overfitting the data, and the hyperparameters need to be tuned to avoid this.
  - *eta*: (Learning Rate) The shrinkage factor between 0 and 1 applied to the weight values in the boosting algorithm.
  - *nrounds*: The number of boosting rounds. If *eta* is set to a small value, it is important to increase the number of rounds since the algorithm learns more slowly.
  - *max_depth*: The maximum depth of the tree. In contrast to the random forest, which fits very deep trees, boosting usually fits the shallow trees. This has the advantage of avoiding spurious complex interactions in the model that can arise from noisy data.
  - *subsample* and *colsample_bytree*: Fraction of the records to sample without replacement and the fraction of predictors to sample for use in fitting the trees.
  - *lambda* and *alpha*: The regularization parameters to help control overfitting.
- Regularization is one way to avoid overfitting by including a penalty term on the number of parameters in a model.
  - *alpha* is the Manhattan distance.
  - *lambda* is the squared Euclidean distance.
- Cross-validation is especially important for boosting due to the large number of hyperparameters that need to be set.
  - Cross-validation randomly splits up the data into K different groups, also called *folds*.
  - For each fold, a model is trained on the data not in the fold and then evaluated on the data in the field.
  - This yields a measure of accuracy of the model on out-of-sample data.
  - The best set of hyperparameters is the one given by the model with the lowest overall error as computed by averaging the errors from each of the folds.

[back to current section](#statistical-machine-learning)

## Unsupervised Learning

* [Principal Component Analysis](#principal-component-analysis)
* [K Means Clustering](#k-means-clustering)
* [Hierarchical Clustering](#hierarchical-clustering)
* [Model Based Clustering](#model-based-clustering)
* [Scaling and Categorical Variables](#scaling-and-categorical-variables)

### Principal Component Analysis

- Principal components are linear combinations of the predictor variables (numeric data only).
- They are calculated so as to minimize correlation between components, reducing redundancy.
- A limited number of components will typically explain most of the variance in the outcome variable.
- The limited set of principal components can then be used in place of the (more numerous) original predictors, reducing dimensionality.

[back to current section](#unsupervised-learning)

### K Means Clustering

- The number of desired clusters, K, is chosen by the user.
- The algorithm develops clusters by iteratively assigning records to the nearest cluster mean until cluster assignments do not change.
- Practical considerations usually dominate the choice of K: there is no statistically determined optimal number of clusters.

[back to current section](#unsupervised-learning)

### Hierarchical Clustering

- Start with every record in its own cluster.
- Progressively, clusters are joined to nearby clusters until all records belong to a single cluster  (the agglomerative algorithm).
- The agglomerative history is retained and plotted, and the user (without specifying the number of clusters beforehand) can visualize the number and structure of clusters at different stages.
- Inter-cluster distances are computed in different ways, all relying on the set of all inter-record distances.

[back to current section](#unsupervised-learning)

### Model Based Clustering

- Clusters are assumed to derive from different data-generating processes with different probability distributions.
- Different models are fit, assuming different numbers of (typically normal) distributions.
- The method chooses the model (and the associated number of clusters) that fits the data well without using too many parameters (i.e., overfitting).

[back to current section](#unsupervised-learning)

### Scaling and Categorical Variables

- Variables measured on different scales need to be transformed to similar scales, so that their impact on algorithms is not determined mainly by their scale.
- A common scaling method is normalization (standardization) - subtracting the mean and dividing by the standard deviation.
- Another method is Gower's distance, which scales all variables to the 0-1 range (it is often used with mixed numeric and categorical data).

[back to current section](#unsupervised-learning)
