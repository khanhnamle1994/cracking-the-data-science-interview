library(MASS)
library(dplyr)
library(ggplot2)
library(FNN)
library(mgcv)
library(rpart)
library(klaR)

options(width=80)


## Import datasets needed for chapter 5
PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')

loan3000 <- read.csv(file.path(PSDS_PATH, 'data', 'loan3000.csv'))
loan_data <- read.csv(file.path(PSDS_PATH, 'data', 'loan_data.csv'))
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))
full_train_set <- read.csv(file.path(PSDS_PATH, 'data', 'full_train_set.csv'))
full_train_set$outcome <- ordered(full_train_set$outcome, levels=c('paid off', 'default'))

## Naive Bayes
naive_model <- NaiveBayes(outcome ~ purpose_ + home_ + emp_len_, 
                          data = na.omit(loan_data))
naive_model$table

new_loan <- loan_data[147, c('purpose_', 'home_', 'emp_len_')]
row.names(new_loan) <- NULL
new_loan

predict(naive_model, new_loan)

## example not in book
less_naive <- NaiveBayes(outcome ~ borrower_score + payment_inc_ratio + 
                           purpose_ + home_ + emp_len_, data = loan_data)
less_naive$table[1:2]

png(filename=file.path(PSDS_PATH, 'figures', 'psds_naive_bayes.png'),  width = 4, height=3, units='in', res=300)

stats <- less_naive$table[[1]]
ggplot(data.frame(borrower_score=c(0,1)), aes(borrower_score)) +
  stat_function(fun = dnorm, color='blue', linetype=1, 
                arg=list(mean=stats[1, 1], sd=stats[1, 2])) +
  stat_function(fun = dnorm, color='red', linetype=2, 
                arg=list(mean=stats[2, 1], sd=stats[2, 2])) +
  labs(y='probability')
dev.off()

#
## Code for LDA
loan_lda <- lda(outcome ~ borrower_score + payment_inc_ratio,
                data=loan3000)
loan_lda$scaling

## Code snippet 4.2
pred <- predict(loan_lda)
head(pred$posterior)


## LDA
## Code for Figure 5-1
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0501.png'),  width = 4, height=3, units='in', res=300)

pred <- predict(loan_lda)
lda_df <- cbind(loan3000, prob_default=pred$posterior[,'default'])

x <- seq(from=.33, to=.73, length=100)
y <- seq(from=0, to=20, length=100)
newdata <- data.frame(borrower_score=x, payment_inc_ratio=y)
pred <- predict(loan_lda, newdata=newdata)
lda_df0 <- cbind(newdata, outcome=pred$class)

ggplot(data=lda_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
  geom_point(alpha=.6) +
  scale_color_gradient2(low='white', high='blue') +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  geom_line(data=lda_df0, col='green', size=2, alpha=.8) +
  theme_bw()

dev.off()


## Logistic regression
logistic_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                        home_ + emp_len_ + borrower_score,
                      data=loan_data, family='binomial')
logistic_model
summary(logistic_model)

p <- seq(from=0.01, to=.99, by=.01)
df <- data.frame(p = p ,
                 logit = log(p/(1-p)),
                 odds = p/(1-p))
## Figure 5-2

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0502.png'),  width = 5, height=4, units='in', res=300)
ggplot(data=df, aes(x=p, y=logit)) +
  geom_line() +
  labs(x = 'p', y='logit(p)') +
  theme_bw()
dev.off()

## Figure 5-3
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0503.png'),  width = 5, height=4, units='in', res=300)
ggplot(data=df, aes(x=logit, y=odds)) +
  geom_line() +
  labs(x = 'log(odds ratio)', y='odds ratio') +
  ylim(1, 100) +
  xlim(0, 5) +
  theme_bw()
dev.off()


pred <- predict(logistic_model)
summary(pred)

prob <- 1/(1 + exp(-pred))
summary(prob)
#

logistic_gam <- gam(outcome ~ s(payment_inc_ratio) + purpose_ + 
                      home_ + emp_len_ + s(borrower_score),
                    data=loan_data, family='binomial')
logistic_gam

terms <- predict(logistic_gam, type='terms')
partial_resid <- resid(logistic_gam) + terms
df <- data.frame(payment_inc_ratio = loan_data[, 'payment_inc_ratio'],
                 terms = terms[, 's(payment_inc_ratio)'],
                 partial_resid = partial_resid[, 's(payment_inc_ratio)'])
  

## Code for Figure 5-4
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0504.png'),  width = 5, height=4, units='in', res=300)

ggplot(df, aes(x=payment_inc_ratio, y=partial_resid, solid = FALSE)) +
  geom_point(shape=46, alpha=.4) +
  geom_line(aes(x=payment_inc_ratio, y=terms), 
            color='red', alpha=.5, size=1.5) +
  labs(y='Partial Residual') +
  xlim(0, 25) +
  theme_bw()

dev.off()


# Confusion matrix
pred <- predict(logistic_gam, newdata=loan_data)
pred_y <- as.numeric(pred > 0)
true_y <- as.numeric(loan_data$outcome=='default')
true_pos <- (true_y==1) & (pred_y==1)
true_neg <- (true_y==0) & (pred_y==0)
false_pos <- (true_y==0) & (pred_y==1)
false_neg <- (true_y==1) & (pred_y==0)
conf_mat <- matrix(c(sum(true_pos), sum(false_pos),
                     sum(false_neg), sum(true_neg)), 2, 2)
colnames(conf_mat) <- c('Yhat = 1', 'Yhat = 0')
rownames(conf_mat) <- c('Y = 1', 'Y = 0')
conf_mat

# precision
conf_mat[1,1]/sum(conf_mat[,1])
# recall
conf_mat[1,1]/sum(conf_mat[1,])
# specificity
conf_mat[2,2]/sum(conf_mat[2,])

## Code for Figure 5-6
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0506.png'),  width = 4, height=4, units='in', res=300)

idx <- order(-pred)
recall <- cumsum(true_y[idx]==1)/sum(true_y==1)
specificity <- (sum(true_y==0) - cumsum(true_y[idx]==0))/sum(true_y==0)
roc_df <- data.frame(recall = recall, specificity = specificity)
ggplot(roc_df, aes(x=specificity, y=recall)) +
  geom_line(color='blue') + 
  scale_x_reverse(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) + 
  geom_line(data=data.frame(x=(0:100)/100), aes(x=x, y=1-x),
            linetype='dotted', color='red') +
  theme_bw()

dev.off()

## Code for Figure 5-7
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0507.png'),  width = 4, height=4, units='in', res=300)

ggplot(roc_df, aes(specificity)) +
  geom_ribbon(aes(ymin=0, ymax=recall), fill='blue', alpha=.3) +
  scale_x_reverse(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) +
  labs(y='recall') +
  theme_bw()

dev.off()

## AUC calculation
sum(roc_df$recall[-1] * diff(1-roc_df$specificity))
head(roc_df)


## Code for Undersampling
mean(full_train_set$outcome=='default')

full_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                    home_ + emp_len_+ dti + revol_bal + revol_util,
                  data=full_train_set, family='binomial')
pred <- predict(full_model)
mean(pred > 0)

## Code for oversampling/up weighting
wt <- ifelse(full_train_set$outcome=='default', 1/mean(full_train_set$outcome == 'default'), 1)
full_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                    home_ + emp_len_+ dti + revol_bal + revol_util,
                  data=full_train_set, weight=wt, family='quasibinomial')
pred <- predict(full_model)
mean(pred > 0)

## SMOTE example (data generation); code not in book. This no longer works for some reason
library(unbalanced)
loan_data_samp <- sample_frac(full_train_set, .05)
smote_data <- ubSMOTE(loan_data_samp, loan_data_samp$outcome, 
                      perc.over = 2000, k = 5, perc.under = 100)

# Code for Figure 5-8: comparison of methods
loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, 
                   control = rpart.control(cp=.005))

lda_pred <- lda_df0[, c('borrower_score', 'payment_inc_ratio')]
lda_pred$method = 'LDA'

tree_pred <- data.frame(borrower_score = c(0.375, 0.375, 0.525, 0.525, 0.625, 0.625),
                        payment_inc_ratio = c(0, 9.732,  9.732, 8.772, 8.772, 20),
                        method = rep('Tree', 6))

glm0 <- glm(outcome ~ (payment_inc_ratio) +  (borrower_score),
            data=loan3000, family='binomial')
y <- seq(from=0, to=20, length=100)
x <- (-glm0$coefficients[1] - glm0$coefficients[2]*y)/glm0$coefficients[3]
glm0_pred <- data.frame(borrower_score=x, payment_inc_ratio=y, method='Logistic')

gam1 <- gam(outcome ~ s(payment_inc_ratio) +  s(borrower_score),
            data=loan3000, family='binomial')
# newdata = gam0_pred

gam_fun <- function(x){
  rss <- sum(predict(gam1, newdata=data.frame(borrower_score=x, payment_inc_ratio=y))^2)
}
est_x <- nlminb(newdata$borrower_score, gam_fun )
gam1_pred <- data.frame(borrower_score=est_x$par, payment_inc_ratio=y, method="GAM")

loan_fits <- rbind(lda_pred,
                   tree_pred,
                   glm0_pred,
                   gam1_pred)


## Code for Figure 5-8
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0508.png'),  width = 6, height=4, units='in', res=300)
ggplot(data=loan_fits, aes(x=borrower_score, y=payment_inc_ratio, color=method, linetype=method)) +
  geom_line(size=1.5) +
  theme(legend.key.width = unit(2,"cm")) +
  guides(linetype = guide_legend(override.aes = list(size = 1)))
dev.off()

###############################################################
# JUNK 
ggplot(roc_df, aes(specificity)) +
  geom_ribbon(aes(ymin=0, ymax=recall), fill='blue', alpha=.3) +
  scale_x_reverse(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) +
  labs(y='recall') +
  theme_bw()

dev.off()
##



loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, 
                   control = rpart.control(cp=.005))

loan_all_data <- loans[loans$status %in% c('Charged Off', 'Current', 'Fully Paid'),]
loan_all_data$outcome <- ifelse(loan_all_data$outcome=='default', 1, 0)
loan_all_data$outcome <- factor(loan_all_data$outcome, levels=0:1, labels=c('good', 'default'))

wt <- mean(loan_all_data$outcome == 'default')
wt

loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, 
                   control = rpart.control(cp=.005))

                  #control = tree.control(nrow(loan_data), minsize=500))
#plot(loan_tree, type='uniform')

png(filename="/Users/andrewbruce1/book/loan_tree_model.png", width = 400, height = 300)

plot(loan_tree, uniform=TRUE, margin=.05)
text(loan_tree, cex=.75)

dev.off()

loan_tree

r_tree <- data_frame(x1 = c(0.525, 0,     0.375, 0.525, 0.625),
                     x2 = c(0.525, 0.525, 0.375, 1,     0.625),
                     y1 = c(0,     9.732, 0,     8.772, 8.772),
                     y2 = c(25,    9.732, 9.732, 8.772, 25),
                     rule_number = factor(c(1, 2, 4, 3, 5)))
r_tree <- as.data.frame(r_tree)

labs <- data.frame(x=c(.375/2, .45, 1.525/2, 1.625/2, .575, .525/2),
                   y=c(8.772/2, 8.772/2, 9.732/2, 
                      9.732 + (25 - 9.732)/2, 9.732 + (25 - 9.732)/2, 9.732 + (25 - 8.772)/2),
                   decision = factor(c('default', 'paid off', 'paid off', 'paid off', 'default', 'default')))


png(filename="/Users/andrewbruce1/book/loan_rpart.png", width = 300, height = 400)

ggplot(data=loan3000, aes(x=borrower_score, y=payment_inc_ratio)) +
  geom_point( aes(color=outcome, shape=outcome), alpha=.5) +
  scale_color_manual(values=c('blue', 'red')) +
  scale_shape_discrete(solid=FALSE) +
  geom_segment(data=r_tree, aes(x=x1, y=y1, xend=x2, yend=y2, linetype=rule_number), size=1.5, alpha=.7) +
  guides(colour = guide_legend(override.aes = list(size=1.5)),
         linetype = guide_legend(keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), limits=c(0, 25)) + 
  geom_label(data=labs, aes(x=x, y=y, label=decision)) +
  theme(legend.position='bottom')

dev.off()

info <- function(x){
  info <- ifelse(x==0, 0, -x * log2(x) - (1-x) * log2(1-x))
  return(info)
}
x <- 0:50/100
plot(x, info(x) + info(1-x))

gini <- function(x){
  return(x * (1-x))
}
plot(x, gini(x))

impure <- data.frame(p = rep(x, 3),
                     impurity = c(2*x,
                                  gini(x)/gini(.5)*info(.5),
                                  info(x)),
                     type = rep(c('Accuracy', 'Gini', 'Entropy'), rep(51,3)))

png(filename="/Users/andrewbruce1/book/impurity.png", width = 300, height = 300)

ggplot(data=impure, aes(x=p, y=impurity, linetype=type, color=type)) + 
  geom_line(size=1.5) +
  guides( linetype = guide_legend( keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0.01)) + 
  scale_y_continuous(expand=c(0,0.01)) + 
  theme(legend.position='bottom', legend.title=element_blank())

dev.off()


###################################################
# ensemble models: random forest

set.seed(50505021)
loan3000 <- loan_data[sample(nrow(loan_data), 3000),  
                      c("outcome", "purpose_", "dti", "borrower_score", "payment_inc_ratio")]
loan_data$purpose <- factor(loan_data$purpose_)

save(loan3000, file="/Users/andrewbruce1/book/loan300.rdata")


library(randomForest)

rf <- randomForest(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000)
#control = tree.control(nrow(loan_data), minsize=500))
#plot(loan_tree, type='uniform')

png(filename="/Users/andrewbruce1/book/rf_accuracy.png", width = 300, height = 250)

error_df = data.frame(error_rate = rf$err.rate[,'OOB'],
                      num_trees = 1:rf$ntree)
ggplot(error_df, aes(x=num_trees, y=error_rate)) +
  geom_line()

dev.off()


png(filename="/Users/andrewbruce1/book/loan_random_forest.png", width = 300, height = 340)

pred <- predict(loan_lda)
rf_df <- cbind(loan3000, pred_default=pred[,'default']>.5, prob_default=pred[, 'default'])

ggplot(data=rf_df, aes(x=borrower_score, y=payment_inc_ratio, 
                       color=pred_default, shape=pred_default)) +
  geom_point(alpha=.6, size=2) +
  scale_shape_manual( values=c( 46, 4)) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme(legend.position='bottom') 

dev.off()

# A nice plot showing a gradient of predictions but not as illustrative as the prior plot
ggplot(data=rf_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
  geom_point(alpha=.6) +
  scale_color_gradient2(low='blue', mid='white', high='red', midpoint=.5) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme(legend.position='bottom') +
  geom_line(data=lda_df0, col='green', size=2, alpha=.8)

nrow(nan.omit(loan_data))

rf_all <- randomForest(outcome ~ ., data=loan_data, importance=TRUE)

varImpPlot(rf_all, type=1)

imp1 <- importance(rf_all, type=1)
imp2 <- importance(rf_all, type=2)
idx <- order(imp1[,1])
nms <- factor(row.names(imp1)[idx], levels=row.names(imp1)[idx])
imp <- data.frame(Predictor = rep(nms, 2),
                  Importance = c(imp1[idx, 1], imp2[idx, 1]),
                  Type = rep( c('Accuracy Decrease', 'Gini Decrease'), rep(nrow(imp1), 2)))

png(filename="/Users/andrewbruce1/book/loan_var_imp.png", width = 300, height = 450)

ggplot(imp) + 
  geom_point(aes(y=Predictor, x=Importance), size=2, stat="identity") + 
  facet_wrap(~Type, ncol=1, scales="free_x") + 
  theme(
    panel.grid.major.x = element_blank() ,
    panel.grid.major.y = element_line(linetype=3, color="darkgray") ) 

dev.off()

loan_data1 <- loan_data0[,-which(names(loan_data0) %in% 'emp_length')]
loan_data1$term = factor(loan_data1$term)
loan_data1$emp_length = factor(loan_data1$emp_length>1)

params <- data.frame(nodesize = c(5, 15, 25, 5, 10, 25),
                     mtry = c(3, 3, 3, 5, 5, 5))
rf_list <- vector('list', 6)
for(i in 1:nrow(params)){
  rf_list[[i]] <- randomForest(outcome ~ ., data=loan_data, mtry=params[i, 'mtry'],
                               nodesize = params[i,'nodesize'], ntree=100)
}

rf_list[[1]]$confusion

varImpPlot(rf_all, type=1)

library(xgboost)
predictors <- data.matrix(loan3000[, c('borrower_score', 'payment_inc_ratio')])
label <- as.numeric(loan3000[,'outcome'])-1
xgb <- xgboost(data=predictors, label=label, objective = "binary:logistic", 
               params=list(subsample=.63, eta=0.1), nrounds=100)


pred <- predict(xgb, newdata=predictors)
xgb_df <- cbind(loan3000, pred_default=pred>.5, prob_default=pred)

png(filename="/Users/andrewbruce1/book/loan_xgboost.png", width = 300, height = 340)

ggplot(data=xgb_df, aes(x=borrower_score, y=payment_inc_ratio, 
                       color=pred_default, shape=pred_default)) +
  geom_point(alpha=.6, size=2) +
  scale_shape_manual( values=c( 46, 4)) +
  scale_x_continuous(expand=c(.03, 0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme(legend.position='bottom') 

dev.off()

seed <- 400820
predictors <- data.matrix(loan_data[,-which(names(loan_data) %in% 'outcome')])
label <- as.numeric(loan_data$outcome)-1
test_idx <- sample(nrow(loan_data), 10000)

test_set <- loan_data[idx,]
train_set <- loan_data[-idx,]


xgb_default <- xgboost(data=predictors[-test_idx,], label=label[-test_idx], 
                   objective = "binary:logistic", nrounds=250)
pred_default <- predict(xgb_default, predictors[test_idx,])
error_default <- abs(label[test_idx] - pred_default) > 0.5
xgb_default$evaluation_log[250,]
mean(error_default)

xgb_penalty <- xgboost(data=predictors[-test_idx,], 
                       label=label[-test_idx], 
                       params=list(eta=.1, subsample=.63, lambda=1000),
                       objective = "binary:logistic", nrounds=250)
pred_penalty <- predict(xgb_penalty, predictors[test_idx,])
error_penalty <- abs(label[test_idx] - pred_penalty) > 0.5
xgb_penalty$evaluation_log[250,]
mean(error_penalty)

error_default <- rep(0, 250)
error_penalty <- rep(0, 250)
for(i in 1:250)
{
  pred_default <- predict(xgb_default, predictors[test_idx,], ntreelimit = i)
  error_default[i] <- mean(abs(label[test_idx] - pred_default) > 0.5)
  pred_penalty <- predict(xgb_penalty, predictors[test_idx,], ntreelimit = i)
  error_penalty[i] <- mean(abs(label[test_idx] - pred_penalty) > 0.5)
}

errors <- rbind(xgb_default$evaluation_log,
                xgb_penalty$evaluation_log,
                data.frame(iter=1:250, train_error=error_default),
                data.frame(iter=1:250, train_error=error_penalty))
errors$type <- rep(c('default train', 'penalty train', 
                     'default test', 'penalty test'), rep(250, 4))
                

png(filename="/Users/andrewbruce1/book/xgboost_error.png", width = 300, height = 340)

ggplot(errors, aes(x=iter, y=train_error, group=type)) +
  geom_line(aes(linetype=type, color=type), size=1.5) +
  theme(legend.position='bottom', legend.key.width = unit(1.5,"cm")) +
  labs(x="Iterations", y="Error") +
  guides(colour = guide_legend(override.aes = list(size=1)))

dev.off()
dev.set(2)



N <- nrow(loan_data)
fold_number <- sample(1:5, N, replace = TRUE)
params <- data.frame(eta = rep(c(.1, .5, .9), 3),
                     max_depth = rep(c(3, 6, 12), rep(3,3)))
rf_list <- vector('list', 9)
error <- matrix(0, nrow=9, ncol=5)
for(i in 1:nrow(params)){
  for(k in 1:5){
    cat('Fold', k, 'for model', i, '\n')
    fold_idx <- (1:N)[fold_number == k]
    xgb <- xgboost(data=predictors[-fold_idx,], label=label[-fold_idx], 
                   params = list(eta = params[i, 'eta'], 
                                 max_depth = params[i, 'max_depth']),
                   objective = "binary:logistic", nrounds=100, verbose=0)
    pred <- predict(xgb, predictors[fold_idx,])
    error[i, k] <- mean(abs(label[fold_idx] - pred) >= 0.5)
  }
}

avg_error <- 100 * round(rowMeans(error), 4)
cbind(params, avg_error)

