# Author: James Le
data1 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt1.csv"))

# Categorize
head(data1)
data1$agecat <- cut(data1$Age, c(-Inf, 0, 18, 24, 34, 44, 54, 64, Inf))

# View
summary(data1)

# Brackets
install.packages("doBy")
library("doBy")
siterange <- function(x){c(length(x), min(x), mean(x), max(x))}
summaryBy(Age ~ agecat, data = data1, FUN = siterange)

# So only signed in users have ages and genders
summaryBy(Gender + Signed_In + Impressions + Clicks ~ agecat, data = data1)

# Plot
library(ggplot2)
ggplot(data1, aes(x=Impressions, fill=agecat)) + geom_histogram(binwidth = 1)
ggplot(data1, aes(x=agecat, y=Impressions, fill=agecat)) + geom_boxplot()

# Create Click-Through Rate
data1$hasimps <- cut(data1$Impressions, c(-Inf, 0, Inf))
summaryBy(Clicks ~ hasimps, data = data1, FUN = siterange)
ggplot(subset(data1, Impressions > 0), aes(x = Clicks / Impressions, colour = agecat)) + geom_density()
ggplot(subset(data1, Clicks > 0), aes(x = Clicks / Impressions, colour = agecat)) + geom_density()
ggplot(subset(data1, Clicks > 0), aes(x = agecat, y = Clicks, fill = agecat)) + geom_boxplot()
ggplot(subset(data1, Clicks > 0), aes(x = Clicks, colour = agecat)) + geom_density()

# Create Categories
data1$scode[data1$Impressions==0] <- 'NoImps'
data1$scode[data1$Impressions>0] <- 'Imps'
data1$scode[data1$Clicks>0] <- 'Clicks'

# Convery the column to a factor
data1$scode <- factor(data1$scode)
head(data1)

# Look at levels
clen <- function(x){c(length(x))}
etable <- summaryBy(Impressions ~ scode + Gender + agecat, data = data1, FUN = clen)