library(dplyr)
library(tidyr)
library(ggplot2)
library(ascii)
library(lubridate)
library(readr)

PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')
## source code to create the data used for the book; this is not ready for public consumption and
## is not needed to run any example in the book
## The data for the book can be downloaded using the script in download_data.t

#################################################################################################
## Create state data  
state_pop <- read.csv("/Users/andrewbruce1/book/state_populations.csv")
murder_rate <- read.csv("/Users/andrewbruce1/book/murder_rate.csv")
state <- merge(state_pop[,1:2], murder_rate[,1:2], by="State")
names(state) <- c("State", "Population", "Murder.Rate" )
state["Abbreviation"] <- state.abb
write.csv(state, file="/Users/andrewbruce1/book/state.csv", row.names = FALSE)



#################################################################################################
## Create airline data
airplanes <- read.csv(download_from_google_drive('0B98qpkK5EJemc3YzUTBoelpjaUU'), stringsAsFactors = FALSE)
names(airplanes)
sort(table(airplanes$airport))
head(airplanes)
dfw <- airplanes %>%
  filter(airport=="DFW", year>2009) %>%
  summarize(Carrier = sum(carrier_ct),
            ATC = sum(nas_ct),
            Weather = sum(X.weather_ct),
            Security = sum(security_ct),
            Inbound = sum(late_aircraft_ct))

write.csv(dfw, file=file.path(path, 'dfw_airline.csv'), row.names = FALSE)


airline_stats <- airplanes %>%
  filter(year>2009,  carrier %in% c("AA", "AS", "B6", "DL", "UA", "WN")) %>%
  transmute(pct_carrier_delay = 100*carrier_ct/arr_flights, 
            pct_atc_delay = 100*nas_ct/arr_flights,
            pct_weather_delay = 100*X.weather_ct/arr_flights,
            airline=factor(carrier, levels = c("AA", "AS", "B6", "DL", "UA", "WN"),
                           labels = c("American", "Alaska", "Jet Blue", "Delta", "United", "Southwest")))
write.csv(airline_stats, file=file.path(PSDS_PATH, 'data', 'airline_stats.csv'), row.names = FALSE)

airport <- airplanes %>%
  filter(airport %in% c("ATL", "BOS", "DFW", "JFK", "SEA", "SFO"),  year>2009) %>%
  group_by(airport) %>%
  summarize(Total = sum(arr_del15),
            Carrier = sum(carrier_ct),
            ATC = sum(nas_ct),
            Weather = sum(X.weather_ct),
            Wecurity = sum(security_ct),
            Inbound = sum(late_aircraft_ct))

#################################################################################################
## Prep the housing tax assessment data

tax_data <- read_delim("/Users/andrewbruce1/local/housing/assessor/EXTR_RealPropApplHist_V.csv", delim=',')
bld_data <- read_delim("/Users/andrewbruce1/local/housing/assessor/EXTR_ResBldg.csv", delim=',')

tax_data0 <- tax_data %>%
  na.omit() %>%
  filter(RollYr >= 2015) %>%
  select(Major, Minor, RollYr, LandVal, ImpsVal) %>%
  mutate(TaxAssessedValue = LandVal + ImpsVal) %>%
  group_by(Major, Minor) %>%
  summarize_each(funs(first))


kc_tax <- bld_data %>%
  left_join(tax_data0, on=c('Major', 'Minor')) %>%
  mutate(ZipCode = factor(ZipCode))

kc_tax <- kc_tax %>%
  select(TaxAssessedValue, SqFtTotLiving, ZipCode)
nrow(kc_tax)

write.csv(kc_tax, file=file.path(PSDS_PATH, 'data', 'kc_tax.csv'), row.names = FALSE)
# save(kc_tax, file= '/Users/andrewbruce1/book/kctax.rdata')

            
#################################################################################################
## Prep the housing sales data

## Import zhvi
zhvi <- read.csv("/Users/andrewbruce1/book/County_Zhvi_AllHomes.csv")
zhvi <- unlist(zhvi[13,-(1:5)])
dates <- parse_date_time(paste(substr(names(zhvi), start=2, stop=8), "01", sep="."), "Ymd")
zhvi <- data.frame(ym=dates, zhvi_px=zhvi, row.names = NULL) %>%
  mutate(zhvi_idx=zhvi/last(zhvi))

# Import sales data
load("/Users/andrewbruce1/book/sales.rdata")
PropertyType <- unlist(sales$PropertyType)
sales$PropertyType <- PropertyType
sales$DocumentDate <- as.character(sales$DocumentDate)
sales <- sales %>%
  select(DocumentDate, SalePrice, PropertyID, PropertyType) %>%
  filter(DocumentDate > "2006-01-01") %>%
  mutate(DocumentDate = parse_date_time(DocumentDate, "Ymd"),
         ym = floor_date(DocumentDate, "month")) %>%
  left_join(zhvi, by="ym") %>%
  mutate(AdjSalePrice = round(SalePrice/zhvi_idx))

# Import condo data
condo <- read.delim("/Users/andrewbruce1/local/housing/assessor/EXTR_CondoUnit2.csv", sep=',')

# Recode usage data
present_use_recode <- function(x)
{
  old <- c("Duplex", "Triplex", "4-Plex", "Vacant(Multi-family)" ,
           "Mobile Home",  "Townhouse Plat",
           "Single Family(Res Use/Zone)",
           "Single Family(C/I Zone)", "Single Family(C/I Use)", 
           "Vacant(Single-family)" )
  new <- c("Multiplex", "Multiplex", "Multiplex", "Multiplex", 
           "Mobile Home",  "Townhouse",
           "Single Family", "Single Family", "Single Family", "Single Family"
           )
  return(new[match(x, old)])
}
# HBUAsIfVacant <- unlist(prop$HBUAsIfVacant)
# prop$HBUAsIfVacant <- HBUAsIfVacant
PresentUse <- unlist(prop$PresentUse)
prop$PresentUse <- PresentUse
prop0 <- prop %>%
  select(PropertyID, NbrLivingUnits,  PresentUse,
         SqFtLot, SqFtTotLiving, SqFtFinBasement, Bathrooms, Bedrooms, BldgGrade, 
         YrBuilt, YrRenovated, TrafficNoise,
         LandVal, ImpsVal, ZipCode)
sales0 <- sales %>%
  left_join(prop0, by='PropertyID') %>%
  mutate(NewConstruction = PropertyType=="Land with new building",
         PropertyType = present_use_recode(PresentUse)
         ) %>%
  select(-PresentUse)

sales0$ZipCode[is.na(sales0$ZipCode)] <- -1
sales0 <- na.omit(sales0)

# export house sales data
write.table(sales0, file='/Users/andrewbruce1/book/house_sales.csv', sep='\t')


#################################################################################################
## Prep the loan data

library(FNN)


## Create loan data


loans0 <- read_csv("/Users/andrewbruce1/book/loanStats.csv")

key_vars <- c("status", "inactive_loans", "bad_loans",
              "grade_num", "sub_grade_num", 
              "loan_amnt", "term", "annual_inc", 
              "dti",  "payment_inc_ratio", "revol_bal", "revol_util",
              "purpose", "home_ownership", "emp_length_num", 
              "delinq_2yrs_zero", "pub_rec_zero", "open_acc")

loans0 <- na.omit(loans0[key_vars])

loans <- loans0 %>%
  mutate(inactive_loans = inactive_loans | bad_loans,
         grade = grade_num + sub_grade_num,
         outcome = factor(bad_loans * inactive_loans + 2*!inactive_loans, 
                          levels=0:2, labels=c( 'paid off', 'default', 'target')),
         emp_length = emp_length_num
  ) %>%
  select(-grade_num, -sub_grade_num, -bad_loans, -inactive_loans, -emp_length_num) %>%
  mutate(payment_inc_ratio = as.numeric(payment_inc_ratio)) %>%
  na.omit()

lc_loans <- loans %>%
  select(status, grade) %>%
  na.omit()

lc_loans[lc_loans$status=="Default", "status"] <- "Charged Off"
lc_loans[lc_loans$status=="In Grace Period", "status"] <- "Late"
lc_loans[lc_loans$status=="Late (16-30 days)", "status"] <- "Late"
lc_loans[lc_loans$status=="Late (31-120 days)", "status"] <- "Late"
lc_loans$status <- ordered(lc_loans$status, levels=c("Fully Paid", "Current", "Late", "Charged Off"))
lc_loans <- na.omit(lc_loans)

grade <- cut(lc_loans$grade, c(-.1, 1, 2, 3, 4, 5, 6, 7.1), labels=c('G', 'F', 'E', 'D', 'C', 'B', 'A'))
lc_loans$grade <- ordered(grade, rev(levels(grade)))

write_csv(lc_loans, file.path(PSDS_PATH, 'data', 'lc_loans.csv'))

###

purpose <- loans$purpose
purpose <- ifelse(purpose %in% c('moving', 'vacation', 'wedding'), 'other', 
                  ifelse(purpose == 'house', 'home_improvement',
                         ifelse(purpose == 'car', 'major_purchase', purpose)))
loans$purpose_ <- purpose

home = loans$home_ownership
home = ifelse(home == 'OTHER', 'RENT', home)
loans$home_ = home

emp_len_ = factor(loans$emp_length < 0.5, labels=c(' > 1 Year', ' < 1 Year'))
loans$emp_len_ = emp_len_

full_train_set <- filter(loans, outcome!='target') %>%
  mutate(outcome = droplevels(outcome))
n_train <- nrow(full_train_set)
full_train_set$outcome <- ordered(full_train_set$outcome, levels=c('paid off', 'default'))

write_csv(full_train_set, file.path(PSDS_PATH, 'data', 'full_train_set.csv'))


default_df <- full_train_set[full_train_set$outcome=='default',]
good_df <- full_train_set[full_train_set$outcome=='paid off', ]
m <- nrow(default_df)

#seed <- .Random.seed
#.Random.seed <- seed
loan_data <- na.omit(bind_rows(default_df, 
                               good_df[sample(nrow(good_df), m),]))

loan_data <- loan_data %>%
  mutate(term = as.numeric(substr(term, 0, 2)),
         addr_state = factor(addr_state),
         purpose_ = factor(purpose_),
         home_ = factor(home_)) %>%
  select(-home_ownership, -status, -purpose, -addr_state, -emp_length)

# Create a feature for borrowers

loan_data <- as.data.frame((loan_data))
borrow_df <- model.matrix(~ -1 + dti + revol_bal + revol_util + open_acc +
                            delinq_2yrs_zero + pub_rec_zero, data=loan_data)

borrow_knn <- knn(borrow_df, test=borrow_df, cl=loan_data[, 'outcome'], prob=TRUE, k=20)
prob <- attr(borrow_knn, "prob")
loan_data$borrower_score <- ifelse(borrow_knn=='default', 1-prob, prob)

write.csv(loan_data, file="/Users/andrewbruce1/book/loan_data.csv")


## create loan3000 dataset
##
seed <- 10101
loan3000 <- loan_data[sample(nrow(loan_data), 3000),  
                      c("outcome", "purpose_", "dti", "borrower_score", "payment_inc_ratio")]
loan_data$purpose <- factor(loan_data$purpose_)

save(loan3000, file="/Users/andrewbruce1/book/loan3000.rdata")
load(file="/Users/andrewbruce1/book/loan3000.rdata")
write.csv(loan3000, file="/Users/andrewbruce1/book/loan3000.csv")

N <- nrow(loans)

idx <- sample(N, 6)
x <- loans[idx, c('loan_amnt', 'annual_inc', 'purpose', 'emp_length_num', 'home_ownership', 'addr_state')]
x

loan_dummies <- model.matrix(~ home_ownership -1, data=loans)[idx[c(1:2, 4:5, 3, 6)],]
head(loan_dummies)


## Create loan200 data for KNN example
seed <- 10101
loan200 <- loan_data[sample(nrow(loan_data), 200),  c("outcome", "payment_inc_ratio", "dti")]
loan200$outcome = as.numeric(loan200$outcome == "paid off")
loan200 <- bind_rows(data.frame(outcome = 2, payment_inc_ratio=9, dti=22.5), loan200)
loan200$outcome <- factor(loan200$outcome, levels=0:2, labels=c('default', 'paid off', 'target'))
write_csv(loan200, "/Users/andrewbruce1/book/loan200.csv")

#########################################################################

set.seed(50505021)
loan3000 <- loan_data[sample(nrow(loan_data), 3000),  
                      c("outcome", "purpose_", "dti", "borrower_score", "payment_inc_ratio")]
loan_data$purpose <- factor(loan_data$purpose_)

save(loan3000, file="/Users/andrewbruce1/book/loan300.rdata")


