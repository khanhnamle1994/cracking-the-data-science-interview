require(ggplot2)

# load and view the diamonds data
data(diamonds)
head(diamonds)

# plot a historgram with a line marking $12,000
ggplot(diamonds) + geom_histogram(aes(x = price)) + geom_vline(xintercept = 12000)

# build a T/F variable indicating if the price is above our threshold
diamonds$Expensive <- ifelse(diamonds$price >= 12000, 1, 0)
head(diamonds)

# get rid of the price column
diamonds$price <- NULL

require(glmnet)

# build the predictor matrix, we are leaving out the last column which is our response
x <- model.matrix(~., diamonds[, -ncol(diamonds)])
# build the response vector
y <- as.matrix(diamonds$Expensive)
# run the glmnet
system.time(modGlmnet <- glmnet(x=x, y=y, family = "binomial"))
# plot the coefficient path
plot(modGlmnet, label=TRUE)

# This illustrates that setting a seed allows you to recreate random results, run them both a few times
set.seed(48872)
sample(1:10)

# Decision tree
require(rpart)
# fire a simple decision tree
modTree <- rpart(Expensive ~ ., data = diamonds)
# plot the splits
plot(modTree)
text(modTree)

# Bagging (Boostrap aggregating)
require(boot)
mean(diamonds$carat)
sd(diamonds$carat)
# function for bootstrapping the mean
boot.mean <- function(x, i)
{
  mean(x[i])
}
# allows us to find the variability of the mean
boot(data = diamonds$carat, statistic = boot.mean, R = 120)

require(adabag)
modBag <- bagging(formula = Species ~ ., iris, mfinal = 10)

# boosting
require(mboost)
system.time(modglmBoost <- glmboost(as.factor(Expensive) ~ .,
                                    data = diamonds,
                                    family = Binomial(link = "logit")))
summary(modglmBoost)
?blackboost

# Random forests
require(randomForest)
system.time(modForest <- randomForest(Species ~ ., data = iris, 
                                      importance = TRUE, proximity = TRUE))