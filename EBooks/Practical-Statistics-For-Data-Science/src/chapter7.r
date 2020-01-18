library(dplyr)
library(tidyr)
library(ggplot2)
library(ascii)
library(lubridate)
library(ellipse)
library(mclust)
library(cluster)


###############################################################
## Import datasets needed for chapter 7
PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')

sp500_px <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_px.csv'), row.names = 1)
sp500_sym <- read.csv(file.path(PSDS_PATH, 'data', 'sp500_sym.csv'), stringsAsFactors = FALSE)
loan_data <- read.csv(file.path(PSDS_PATH, 'data', 'loan_data.csv'))
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))

###############################################################
## PCA for oil data
oil_px = as.data.frame(scale(oil_px, scale=FALSE))
oil_px <- sp500_px[, c('CVX', 'XOM')]
pca <- princomp(oil_px)
pca$loadings

## Figure 7-1: principal components for oil stock data
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0701.png'), width = 4, height=4, units='in', res=300)
loadings <- pca$loadings
ggplot(data=oil_px, aes(x=CVX, y=XOM)) +
  geom_point(alpha=.3) +
  scale_shape_manual(values=c(46)) +
  stat_ellipse(type='norm', level=.99, color='grey25') +
  geom_abline(intercept = 0, slope = loadings[2,1]/loadings[1,1], color='grey25', linetype=2) +
  geom_abline(intercept = 0, slope = loadings[2,2]/loadings[1,2],  color='grey25', linetype=2) +
  scale_x_continuous(expand=c(0,0), lim=c(-3, 3)) + 
  scale_y_continuous(expand=c(0,0), lim=c(-3, 3)) +
  theme_bw()

dev.off()



## Figure 7-2: screeplot 
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0702.png'), width = 4, height=4, units='in', res=300)

syms <- c( 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
           'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
top_cons <- sp500_px[row.names(sp500_px)>='2011-01-01', syms]
sp_pca <- princomp(top_cons)
par(mar=c(6,3,0,0)+.1, las=2)
screeplot(sp_pca, main='')

dev.off()

## Loadings for stock data
loadings = sp_pca$loadings[,1:5]
loadings <- as.data.frame(loadings)
loadings$Symbol <- row.names(loadings)
loadings <- gather(loadings, "Component", "Weight", -Symbol)
head(loadings)

## Figure 7-3: Plot of component loadings
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0703.png'), width = 4, height=4, units='in', res=300)

loadings$Color = loadings$Weight > 0
ggplot(loadings, aes(x=Symbol, y=Weight, fill=Color)) +
  geom_bar(stat='identity', position = "identity", width=.75) + 
  facet_grid(Component ~ ., scales='free_y') +
  guides(fill=FALSE)  +
  ylab('Component Loading') +
  theme_bw() +
  theme(axis.title.x = element_blank(),
        axis.text.x  = element_text(angle=90, vjust=0.5))

dev.off()


###############################################################
## K-means chapter

set.seed(1010103)
df <- sp500_px[row.names(sp500_px)>='2011-01-01', c('XOM', 'CVX')]
km <- kmeans(df, centers=4, nstart=1)

df$cluster <- factor(km$cluster)
head(df)

centers <- data.frame(cluster=factor(1:4), km$centers)
centers

## Figure 7-4: K-means clusters for two stocks
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0704.png'), width = 4, height=3, units='in', res=300)

ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point(alpha=.3) +
  scale_shape_manual(values = 1:4,
                     guide = guide_legend(override.aes=aes(size=1))) + 
  geom_point(data=centers,  aes(x=XOM, y=CVX), size=2, stroke=2)  +
  theme_bw() +
  scale_x_continuous(expand=c(0,0), lim=c(-2, 2)) + 
  scale_y_continuous(expand=c(0,0), lim=c(-2.5, 2.5)) 

dev.off()


## cluster means algorithm
syms <- c( 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 'SLB', 'COP',
           'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
df <- sp500_px[row.names(sp500_px)>='2011-01-01', syms]

set.seed(10010)
km <- kmeans(df, centers=5, nstart=10)
km$size
centers <- km$centers

#centers <- scale(scale(centers, center=FALSE, scale=1/attr(df, 'scaled:scale')),
#                 center=-attr(df, 'scaled:center'), scale=FALSE)

## Figure 7-5 interpreting the clusters
centers <- as.data.frame(t(centers))
names(centers) <- paste("Cluster", 1:5)
centers$Symbol <- row.names(centers)
centers <- gather(centers, "Cluster", "Mean", -Symbol)

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0705.png'), width = 4, height=5, units='in', res=300)

centers$Color = centers$Mean > 0
ggplot(centers, aes(x=Symbol, y=Mean, fill=Color)) +
  geom_bar(stat='identity', position = "identity", width=.75) + 
  facet_grid(Cluster ~ ., scales='free_y') +
  guides(fill=FALSE)  +
  ylab('Component Loading') +
  theme_bw() +
  theme(axis.title.x = element_blank(),
        axis.text.x  = element_text(angle=90, vjust=0.5))

dev.off()

## Figure 7-6: selecting the number of clusters (elbow plot)
pct_var <- data.frame(pct_var = 0,
                      num_clusters=2:14)
totalss <- kmeans(df, centers=14, nstart=50, iter.max = 100)$totss
for(i in 2:14){
  pct_var[i-1, 'pct_var'] <- kmeans(df, centers=i, nstart=50, iter.max = 100)$betweenss/totalss
}

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0706.png'), width = 4, height=3, units='in', res=300)

ggplot(pct_var, aes(x=num_clusters, y=pct_var)) +
  geom_line() +
  geom_point() +
  labs(y='% Variance Explained', x='Number of Clusters') +
  scale_x_continuous(breaks=seq(2, 14, by=2))   +
  theme_bw()
dev.off()


################################################################
## hclust chapter

syms1 <- c('GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 
           'XOM', 'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP',
           'WMT', 'TGT', 'HD', 'COST')

df <- sp500_px[row.names(sp500_px)>='2011-01-01', syms1]
d <- dist(t(df))
hcl <- hclust(d)

## Figure 7-7: dendograme of stock data
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0707.png'), width = 4, height=4, units='in', res=300)

par(cex=.75, mar=c(0, 5, 0, 0)+.1)
plot(hcl, ylab='distance', xlab='', sub='', main='')

dev.off()

## Figure 7-8: comparison of the different measuresof dissimilarity
cluster_fun <- function(df, method)
{
  d <- dist(df)
  hcl <- hclust(d, method=method)
  tree <- cutree(hcl, k=4)
  df$cluster <- factor(tree)
  df$method <- method
  return(df)
}

df0 <- sp500_px[row.names(sp500_px)>='2011-01-01', c('XOM', 'CVX')]
df <- rbind(cluster_fun(df0, method='single'),
            cluster_fun(df0, method='average'),
            cluster_fun(df0, method='complete'),
            cluster_fun(df0, method='ward.D'))
df$method <- ordered(df$method, c('single', 'average', 'complete', 'ward.D'))

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0708.png'), width = 5.5, height=4, units='in', res=300)

ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point(alpha=.3) +
  scale_shape_manual(values = c(46, 3, 1,  4),
                     guide = guide_legend(override.aes=aes(size=2))) +
  facet_wrap( ~ method) +
  theme_bw()

dev.off()



###############################################################
# Model-based clusting
# Multivariate normal

mu <- c(.5, -.5)
sigma <- matrix(c(1, 1, 1, 2), nrow=2)
prob <- c(.5, .75, .95, .99) ## or whatever you want
names(prob) <- prob ## to get id column in result
x <- NULL
for (p in prob){
  x <- rbind(x,  ellipse(x=sigma, centre=mu, level=p))
}
df <- data.frame(x, prob=factor(rep(prob, rep(100, length(prob)))))
names(df) <- c("X", "Y", "Prob")

## Figure 7-9: Multivariate normal ellipses
dfmu <- data.frame(X=mu[1], Y=mu[2])
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0709.png'), width = 4, height=4, units='in', res=300)

ggplot(df, aes(X, Y)) + 
  geom_path(aes(linetype=Prob)) +
  geom_point(data=dfmu, aes(X, Y), size=3) +
  theme_bw()

dev.off()

## Figure 7-10 mclust applied XOM and CVX

df <- sp500_px[row.names(sp500_px)>='2011-01-01', c('XOM', 'CVX')]
mcl <- Mclust(df)
summary(mcl)

cluster <- factor(predict(mcl)$classification)
png(filename=file.path(PSDS_PATH, 'figures', 'psds_0710.png'), width = 5, height=4, units='in', res=300)

ggplot(data=df, aes(x=XOM, y=CVX, color=cluster, shape=cluster)) +
  geom_point(alpha=.8) +
  theme_bw() +
  scale_shape_manual(values = c(46, 3),
                     guide = guide_legend(override.aes=aes(size=2))) 

dev.off()

summary(mcl, parameters=TRUE)$mean
summary(mcl, parameters=TRUE)$variance

## Figure 7-11: BIC scores for the different models fit by mclust

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0711.png'), width = 4, height=4, units='in', res=300)

par(mar=c(4, 5, 0, 0)+.1)
plot(mcl, what='BIC', ask=FALSE, cex=.75)

dev.off()
#

#######################################################################
# Scaling chapter

defaults <- loan_data[loan_data$outcome=='default',]
df <- defaults[, c('loan_amnt', 'annual_inc', 'revol_bal', 'open_acc', 'dti', 'revol_util')]
km <- kmeans(df, centers=4, nstart=10)
centers <- data.frame(size=km$size, km$centers) 
round(centers, digits=2)

df0 <- scale(df)
km0 <- kmeans(df0, centers=4, nstart=10)
centers0 <- scale(km0$centers, center=FALSE, scale=1/attr(df0, 'scaled:scale'))
centers0 <- scale(centers0, center=-attr(df0, 'scaled:center'), scale=FALSE)
centers0 <- data.frame(size=km0$size, centers0) 
round(centers0, digits=2)

km <- kmeans(df, centers=4, nstart=10)
centers <- data.frame(size=km$size, km$centers) 
round(centers, digits=2)


## Figure 7-12: screeplot for data with dominant variables

syms <- c('GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM', 
          'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST')
top_15 <- sp500_px[row.names(sp500_px)>='2011-01-01', syms]
sp_pca1 <- princomp(top_15)

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0712.png'), width = 4, height=4, units='in', res=300)

par(mar=c(6,3,0,0)+.1, las=2)
screeplot(sp_pca1, main='')

dev.off()

round(sp_pca1$loadings[,1:2], 3)

#
###############################################################
## Figure 7-13: Categorical data and Gower's distance

x <- loan_data[1:5, c('dti', 'payment_inc_ratio', 'home_', 'purpose_')]
x

daisy(x, metric='gower')

set.seed(301)
df <- loan_data[sample(nrow(loan_data), 250),
                c('dti', 'payment_inc_ratio', 'home_', 'purpose_')]
d = daisy(df, metric='gower')
hcl <- hclust(d)
dnd <- as.dendrogram(hcl)

png(filename=file.path(PSDS_PATH, 'figures', 'psds_0713.png'), width = 4, height=4, units='in', res=300)
par(mar=c(0,5,0,0)+.1)
plot(dnd, leaflab='none', ylab='distance')
dev.off()

dnd_cut <- cut(dnd, h=.5)
df[labels(dnd_cut$lower[[1]]),]


## Problems in clustering with mixed data types
df <- model.matrix(~ -1 + dti + payment_inc_ratio + home_ + pub_rec_zero, data=defaults)
df0 <- scale(df)
km0 <- kmeans(df0, centers=4, nstart=10)
centers0 <- scale(km0$centers, center=FALSE, scale=1/attr(df0, 'scaled:scale'))
round(scale(centers0, center=-attr(df0, 'scaled:center'), scale=FALSE), 2)
