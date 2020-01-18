library(dplyr)
library(ggplot2)
library(FNN)
library(rpart)
library(randomForest)
library(xgboost)

###############################################################

## Import datasets needed for chapter 6
PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')

loan200 <- read.csv(file.path(PSDS_PATH, 'data', 'loan200.csv'))
loan200$outcome <- ordered(loan200$outcome, levels=c('paid off', 'default'))

loan3000 <- read.csv(file.path(PSDS_PATH, 'data', 'loan3000.csv'))
loan3000$outcome <- ordered(loan3000$outcome, levels=c('paid off', 'default'))

loan_data <- read.csv(file.path(PSDS_PATH, 'data', 'loan_data.csv'))
loan_data <- select(loan_data, -X, -status)

## KNN
## the first row of loan200 is the target data
newloan <- loan200[1, 2:3, drop=FALSE]
knn_pred <- knn(train=loan200[-1,2:3], test=newloan, cl=loan200[-1,1], k=20)
knn_pred == 'paid off'

## look at the nearest 20 records
loan200[attr(knn_pred, 'nn.index')-1, ]
dist <- attr(knn_pred, 'nn.dist')

circleFun <- function(center = c(0,0), r = 1, npoints = 100){
  tt <- seq(0, 2*pi, length.out = npoints-1)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x = c(xx, xx[1]), y = c(yy, yy[1])))
}

circle_df <- circleFun(center=unlist(newloan), r=max(dist), npoints=201)
loan200_df <- bind_cols(loan200, circle_df)

## Code for figure 6-2: small KNN example
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0602.png'), width = 5.5, height=4, units='in', res=300)

ggplot(data=loan200_df, aes(x=payment_inc_ratio, dti, color=outcome, shape=outcome)) +
  geom_point(size=2) +
  scale_shape_manual(values = c(1, 4, 15)) +
  geom_path(aes(x=x, y=y), color='black') +
  xlim(3, 15) + 
  ylim(17, 29) +
  theme_bw() 

dev.off()

## Standardization
loan_df <- model.matrix(~ -1 + payment_inc_ratio + dti + revol_bal + revol_util, data=loan_data)
newloan = loan_df[1,, drop=FALSE]
loan_df = loan_df[-1,]
outcome <- loan_data[-1,1]
knn_pred <- knn(train=loan_df, test=newloan, cl=outcome, k=5)
knn_pred
loan_df[attr(knn_pred,"nn.index"),]

loan_df <- model.matrix(~ -1 + payment_inc_ratio + dti + revol_bal + revol_util, data=loan_data)
loan_std <- scale(loan_df)
target_std = loan_std[1,, drop=FALSE]
loan_std = loan_std[-1,]
outcome <- loan_data[-1,1]
knn_pred <- knn(train=loan_std, test=target_std, cl=outcome, k=5)
knn_pred
loan_df[attr(knn_pred,"nn.index"),]

## Create a feature for borrowers
borrow_df <- model.matrix(~ -1 + dti + revol_bal + revol_util + open_acc +
                            delinq_2yrs_zero + pub_rec_zero, data=loan_data)
borrow_knn <- knn(borrow_df, test=borrow_df, cl=loan_data[, 'outcome'], prob=TRUE, k=20)
prob <- attr(borrow_knn, "prob")
borrow_feature <- ifelse(borrow_knn=='default', 1-prob, prob)
summary(borrow_feature)

loan_data$borrower_score <- borrow_feature

################################################################
# Decision trees

loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, 
                   control = rpart.control(cp=.005))

## Figure 6-3: Rules for simple tree model (not same as in book)
png(filename=file.path(PSDS_PATH, 'figures', 'psds_rpart_tree.png'),  width = 6, height=4, units='in', res=300)

par(mar=c(0,0,0,0)+.1)
plot(loan_tree, uniform=TRUE, margin=.05)
text(loan_tree, cex=.75)

dev.off()

## Figure 6-4: View of partition rules
r_tree <- data_frame(x1 = c(0.575, 0.375, 0.375, 0.375, 0.475),
                     x2 = c(0.575, 0.375, 0.575, 0.575, 0.475),
                     y1 = c(0,         0, 10.42, 4.426, 4.426),
                     y2 = c(25,       25, 10.42, 4.426, 10.42),
                     rule_number = factor(c(1, 2, 3, 4, 5)))
r_tree <- as.data.frame(r_tree)

labs <- data.frame(x=c(.575 + (1-.575)/2, 
                       .375/2, 
                       (.375 + .575)/2,
                       (.375 + .575)/2, 
                       (.475 + .575)/2, 
                       (.375 + .475)/2
                       ),
                   y=c(12.5, 
                       12.5,
                       10.42 + (25-10.42)/2,
                       4.426/2, 
                       4.426 + (10.42-4.426)/2,
                       4.426 + (10.42-4.426)/2
                       ),
                   decision = factor(c('paid off', 'default', 'default', 'paid off', 'paid off', 'default')))


png(filename=file.path(PSDS_PATH, 'figures', 'psds_0604.png'), width = 6, height=4, units='in', res=300)

ggplot(data=loan3000, aes(x=borrower_score, y=payment_inc_ratio)) +
  geom_point( aes(color=outcome, shape=outcome), alpha=.5) +
  scale_color_manual(values=c('blue', 'red')) +
  scale_shape_manual(values = c(1, 46)) +
  # scale_shape_discrete(solid=FALSE) +
  geom_segment(data=r_tree, aes(x=x1, y=y1, xend=x2, yend=y2, linetype=rule_number), size=1.5, alpha=.7) +
  guides(colour = guide_legend(override.aes = list(size=1.5)),
         linetype = guide_legend(keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), limits=c(0, 25)) + 
  geom_label(data=labs, aes(x=x, y=y, label=decision)) +
  #theme(legend.position='bottom') +
  theme_bw()

dev.off()


## Gini coefficient and impurity

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

## Figure 06-05: comparison of impurity measures
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0605.png'), width = 5, height=4, units='in', res=300)

ggplot(data=impure, aes(x=p, y=impurity, linetype=type, color=type)) + 
  geom_line(size=1.5) +
  guides( linetype = guide_legend( keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0.01)) + 
  scale_y_continuous(expand=c(0,0.01)) + 
  theme_bw() +
  theme( legend.title=element_blank()) 
  
dev.off()


###############################################################################
# ensemble models: random forest

rf <- randomForest(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000)
rf

## Figure 6-6: error rate of random forest
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0606.png'), width = 5, height=4, units='in', res=300)

error_df = data.frame(error_rate = rf$err.rate[,'OOB'],
                      num_trees = 1:rf$ntree)
ggplot(error_df, aes(x=num_trees, y=error_rate)) +
  geom_line()  +
  theme_bw()

dev.off()

## Figure 6-7: plot of random forest predictions
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0607.png'),  width = 5, height=4, units='in', res=300)

pred <- predict(rf, prob=TRUE)
rf_df <- cbind(loan3000, pred = pred)

ggplot(data=rf_df, aes(x=borrower_score, y=payment_inc_ratio, 
                       shape=pred, color=pred)) +
  geom_point(alpha=.6, size=2) +
  scale_shape_manual( values=c( 46, 4)) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme_bw()
dev.off()



# A nice plot showing a gradient of predictions but not as illustrative as the prior plot
## not in book
png(filename=file.path(PSDS_PATH, 'figures', 'psds_rf_gradient.png'),  width = 5, height=4, units='in', res=300)
ggplot(data=rf_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
  geom_point(alpha=.6) +
  scale_color_gradient2(low='blue', mid='white', high='red', midpoint=.5) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme(legend.position='bottom') +
  geom_line(data=lda_df0, col='green', size=2, alpha=.8)
dev.off()

## Fit random forest to all the data; this can take a while and
## may cause problems if you don't have enough RAM
rf_all <- randomForest(outcome ~ ., data=loan_data, importance=TRUE)
rf_all

varImpPlot(rf_all, type=1)

imp1 <- importance(rf_all, type=1)
imp2 <- importance(rf_all, type=2)
idx <- order(imp1[,1])
nms <- factor(row.names(imp1)[idx], levels=row.names(imp1)[idx])
imp <- data.frame(Predictor = rep(nms, 2),
                  Importance = c(imp1[idx, 1], imp2[idx, 1]),
                  Type = rep( c('Accuracy Decrease', 'Gini Decrease'), rep(nrow(imp1), 2)))

## Figure 6-8: Variable importance plot for random forest
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0608.png'),  width = 4, height=6, units='in', res=300)

ggplot(imp) + 
  geom_point(aes(y=Predictor, x=Importance), size=2, stat="identity") + 
  facet_wrap(~Type, ncol=1, scales="free_x") + 
  theme(
    panel.grid.major.x = element_blank() ,
    panel.grid.major.y = element_line(linetype=3, color="darkgray") ) +
  theme_bw()

dev.off()

## search over hyperparameter space (not in book); this takes a while
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


###############################################################################
# ensemble models: xgboost

predictors <- data.matrix(loan3000[, c('borrower_score', 'payment_inc_ratio')])
label <- as.numeric(loan3000[,'outcome'])-1
xgb <- xgboost(data=predictors, label=label, objective = "binary:logistic", 
               params=list(subsample=.63, eta=0.1), nrounds=100)


pred <- predict(xgb, newdata=predictors)
xgb_df <- cbind(loan3000, pred_default=pred>.5, prob_default=pred)

## Figure 6-9: prediction from xgboost
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0609.png'), width = 5, height=4, units='in', res=300)

ggplot(data=xgb_df, aes(x=borrower_score, y=payment_inc_ratio, 
                        color=pred_default, shape=pred_default)) +
  geom_point(alpha=.6, size=2) +
  scale_shape_manual( values=c( 46, 4)) +
  scale_x_continuous(expand=c(.03, 0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  theme_bw()

dev.off()

## Create a test and training set and compare the learning rates under different hyperparameter choices
seed <- 400820
predictors <- data.matrix(loan_data[,-which(names(loan_data) %in% 'outcome')])
label <- as.numeric(loan_data$outcome)-1
test_idx <- sample(nrow(loan_data), 10000)

xgb_default <- xgboost(data=predictors[-test_idx,], label=label[-test_idx], 
                       objective = "binary:logistic", nrounds=250, verbose=0)
pred_default <- predict(xgb_default, predictors[test_idx,])
error_default <- abs(label[test_idx] - pred_default) > 0.5
xgb_default$evaluation_log[250,]
mean(error_default)

xgb_penalty <- xgboost(data=predictors[-test_idx,], 
                       label=label[-test_idx], 
                       params=list(eta=.1, subsample=.63, lambda=1000),
                       objective = "binary:logistic", nrounds=250, verbose=0)
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

## Figure 6-10: learning rates for different choices of hyperparameters
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0610.png'), width = 6, height=4, units='in', res=300)

ggplot(errors, aes(x=iter, y=train_error, group=type)) +
  geom_line(aes(linetype=type, color=type), size=1) +
  scale_linetype_manual(values=c('solid', 'dashed', 'dotted', 'longdash')) +
  theme_bw() +
  theme(legend.key.width = unit(1.5,"cm")) +
  labs(x="Iterations", y="Error") +
  guides(colour = guide_legend(override.aes = list(size=1)))

dev.off()

## Cross validation
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

