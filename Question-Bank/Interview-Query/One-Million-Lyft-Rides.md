## Question
This question was asked by: Lyft

Let's say we have 1 million Lyft rider journey trips in the city of Seattle. We want to build a model to predict ETA after a rider makes a Lyft request.

How would we know if we have enough data to create an accurate enough model?

<!-- ## Solution
Collecting data can be costly. This question assesses the candidate’s skill in being able to practically figure out if a model needs more data. There are a couple of factors to look into:

- Look at the feature set size to training data size ratio. If we have an extremely high number of features compared to training data, then the model inaccuracy will be prone to overfitting.

- Create an existing model off a portion of the data, the training set, and measure performance of the model on the validation sets, otherwise known as using a holdout set. We hold back some subset of the data from the training of the model, and then use this holdout set to check the model performance to get a baseline level.

- Learning curves. Learning curves help us calculate our accuracy rate by testing data on subsequently larger subsets of data. If we fit our model on 20%, 40%, 60%, 80% of our data size and then cross-validate to determine model accuracy, we can then determine how much more data we need to achieve a certain accuracy level.

For example. If we reach 75% accuracy with 500K datapoints but then only 77% accuracy with 1 million datapoints, then we’ll realize that our model is not predicting well enough with it’s existing features since doubling the training data size did not significantly increase the accuracy rate. -->
