file <- "binary-class-dataset.txt"
set <- read.table(file, header = TRUE, sep = "\t")
names(set)

split <- .65
set['rand'] <- runif(nrow(set))
train <- set[(set$rand <= split),]
test <- set[(set$rand > split),]
set$y <- set$y_buy

## R Functions
library(mgcv)

# GAM Smoothed plot
plotrel <- function(x, y, b, title) {
  # Produce a GAM smoothed representation of the data
  g <- gam(as.formula("y ~ x"), family = "binomial", data = set)
  xs <- seq(min(x), max(x), length = 200)
  p <- predict(g, newdata = data.frame(x = x), type = "response")
  
  # Now get empirical estimates (and discretize if non-discrete)
  if (length(unique(x)) > b) {
    div <- floor(max(x) / b)
    x_b <- floor(x / div) * div
    c <- table(x_b, y)
  }
  else { c <- table(x, y) }
  pact <- c[ , 2] / (c[ , 1] + c[ , 2])
  cnt <- c[ , 1] + c[ , 2]
  xd <- as.integer(rownames(c))
  plot(xs, p, type = "l", main = title,
       ylab = "P(Conversion | Ad), X", xlab = "X")
  points(xd, pact, type = "p", col = "red")
  rug(x + runif(length(x)))
}

library(plyr)

# wMAE plot and calculation
getmae <- function(p, y, b, title, doplot) {
  # Normalize to interval [0, 1]
  max_p <- max(p)
  p_norm <- p / max_p
  # break up to b bins and rescale
  bin <- max_p * floor(p_norm * b) / b
  d <- data.frame(bin, p, y)
  t <- table(bin)
  summ <- ddply(d, .(bin), summarise, mean_p = mean(p), mean_y = mean(y))
  fin <- data.frame(bin = summ$bin, mean_p = summ$mean_p, mean_y = summ$mean_y, t)
  # Get wMAE
  num = 0
  den = 0
  for (i in c(1:nrow(fin))) {
    num <- num + fin$Freq[i] * abs(fin$mean_p[i] - fin$mean_y[i])
    den <- den + fin$Freq[q]
  }
  wmae <- num / den
  if (doplot == 1) {
    plot(summ$bin, summ$mean_p, type = "p",
         main = paste(title, " MAE =", wmae),
         col = "blue", ylab = "P(C | AD, X)",
         xlab = "P(C | AD, X)")
    points(summ$bin, summ$mean_y, type = "p", col = "red")
    rug(p)
  }
  return(wmae)
}

library(ROCR)
get_auc <- function(ind, y) {
  pred <- prediction(ind, y)
  perf <- performance(pred, 'auc', fpr.stop = 1)
  auc <- as.numeric(substr(slot(perf, "y.values"), 1, 8), double)
  return(auc)
}

# Get X-validated performance metrics for a given feature set
getxval <- function(vars, data, folds, mae_bins) {
  # assign each observation to a fold
  data["fold"] <- floor(runif(nrow(data)) * folds) + 1
  auc <- c()
  wmae <- c()
  fold <- c()
  # make a formula object
  f = as.formula(paste("y", "~", paste(vars, collapse = "+")))
  for (i in c(1:folds)) {
    train <- data[(data$fold != i), ]
    test <- data[(data$fold == i), ]
    mod_x <- glm(f, data = train, family = binomial(logit))
    p <- predict(mod_x, newdata = test, type = "response")
    # Get wMAE
    wmae <- c(wmae, getmae(p, test$Y, mae_bins, "dummy", 0))
    fold <- c(fold, i)
    auc <- c(auc, get_auc(p, test$Y))
  }
  return(data.frame(fold, wmae, auc))
}

## Main: Models and Plots
# Now build a model on all variables and look at coefficients and model fit
vlist <- c("at_buy_boolean", "at_freq_buy", "at_freq_last24_buy", "at_freq_last24_sv",
           "at_freq_sv", "expected_time_buy", "expected_time_sv", "last_buy",
           "last_sv", "num_checkins")
f = as.formula(paste("y_buy", "~", paste(vlist, collapse = "+")))
fit <- glm(f, data = train, family = binomial(logit))
summary(fit)

# Create empty vectors to store the performance/evaluation metrics
auc_mu <- c()
auc_sig <- c()
mae_mu <- c()
mae_sig <- c()

for (i in c(1:length(vlist))) {
  a <- getxval(c(vlist[i]), set, 10, 100)
  auc_mu <- c(auc_mu, mean(a$auc))
  auc_sig <- c(auc_sig, sd(a$auc))
  mae_mu <- c(mae_mu, mean(a$wmae))
  mae_mu <- c(mae_sig, sd(a$wmae))
}

univar <- data.frame(vlist, auc_mu, auc_sig, mae_mu, mae_sig)