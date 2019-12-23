setwd('Desktop/Cracking-The-DS-Interview/Doing-Data-Science-Straight-Talk-From-The-Front-Line')

library(plyr)

require(gdata)
bk <- read.xls("rollingsales_brooklyn.xls", pattern="BOROUGH")
head(bk)
summary(bk)

bk$SALE.PRICE.N <- as.numeric(gsub("[^[:digit:]]", "", bk$SALE.PRICE))
count(is.na(bk$SALE.PRICE.N))

## clean/format the data with regexs
bk$gross.sqft <- as.numeric(gsub("[^[:digit:]]", "", bk$GROSS.SQUARE.FEET))
bk$land.sqft <- as.numeric(gsub("[^[:digit:]]", "", bk$LAND.SQUARE.FEET))

## do a bit of exploration to make sure there's not thing weired going on with sale prices
attach(bk)

hist(SALE.PRICE.N)
hist(SALE.PRICE.N[SALE.PRICE.N > 0])
hist(gross.sqft[SALE.PRICE.N == 0])

detach(bk)

## keep only the actual sales
bk.sale <- bk[bk$SALE.PRICE.N != 0,]

plot(bk.sale$gross.sqft, bk.sale$SALE.PRICE.N)
plot(log(bk.sale$gross.sqft), log(bk.sale$SALE.PRICE.N))

## for now, let's look at 1-, 2-, and 3-family homes
bk.homes <- bk.sale[which(grepl("FAMILY",
                                bk.sale$BUILDING.CLASS.CATEGORY)),]
plot(log(bk.homes$gross.sqft), log(bk.homes$SALE.PRICE.N))

# Linear Regression on the housing dataset
model1 <- lm(log(SALE.PRICE.N) ~ log(gross.sqft), data = bk.homes)

## what's going on here?
bk.homes[which(bk.homes$gross.sqft==0),]

bk.homes <- bk.homes[which(bk.homes$gross.sqft>0 & bk.homes$land.sqft>0),]
model1 <- lm(log(SALE.PRICE.N) ~ log(gross.sqft), data = bk.homes)
summary(model1)

plot(log(bk.homes$gross.sqft), log(bk.homes$SALE.PRICE.N))
abline(model1, col="red", lwd=2)
plot(resid(model1))

model2 <- lm(log(SALE.PRICE.N) ~ log(gross.sqft) + 
               log(land.sqft) + factor(NEIGHBORHOOD), data = bk.homes)
summary(model2)
plot(resid(model2))

## leave out intercept for ease of interpretability
model2a <- lm(log(SALE.PRICE.N) ~ 0 + log(gross.sqft) + 
                log(land.sqft) + factor(NEIGHBORHOOD), data = bk.homes)
summary(model2a)
plot(resid(model2a))

## add building type
model3 <- lm(log(SALE.PRICE.N) ~ log(gross.sqft) + 
               log(land.sqft) + factor(NEIGHBORHOOD) + 
               factor(BUILDING.CLASS.CATEGORY), data = bk.homes)
summary(model3)
plot(resid(model3))

## interact neighborhood and building type
model4 <- lm(log(SALE.PRICE.N) ~ log(gross.sqft) + log(land.sqft) + 
               factor(NEIGHBORHOOD) * factor(BUILDING.CLASS.CATEGORY), data = bk.homes)
summary(model4)
plot(resid(model4))