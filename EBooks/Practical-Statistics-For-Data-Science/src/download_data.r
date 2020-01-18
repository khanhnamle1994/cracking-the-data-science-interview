## googledrive requires installation of numerous packages; if you encounter errors, you should update all packages.
## an alternative approach to download is included below

library(googledrive)
PSDS_PATH <- file.path('~', 'statistics-for-data-scientists')

## Import state data
drive_download(as_id("0B98qpkK5EJembFc5RmVKVVJPdGc"), path=file.path(PSDS_PATH, 'data', 'state.csv'), overwrite=TRUE)

## Airline data
drive_download(as_id("0B98qpkK5EJemcmZYX2VhMHBXelE"), path=file.path(PSDS_PATH, 'data', 'dfw_airline.csv'), overwrite=TRUE)
drive_download(as_id("0B98qpkK5EJemMzZYZHZJaF9va0U"), path=file.path(PSDS_PATH, 'data', 'airline_stats.csv'), overwrite=TRUE)

## Import stock data
drive_download(as_id('0B98qpkK5EJemV2htZWdhVFRMNlU'), path=file.path(PSDS_PATH, 'data', 'sp500_px.csv'), overwrite=TRUE)
drive_download(as_id('0B98qpkK5EJemY0U0N1N6a21lUzA'), path=file.path(PSDS_PATH, 'data', 'sp500_sym.csv'), overwrite=TRUE)

## Import KC housing tax data
drive_download(as_id('0B98qpkK5EJemck5VWkszN3F3RGM'), path=file.path(PSDS_PATH, 'data', 'kc_tax.csv'), overwrite=TRUE)

## Import lending club loan data
drive_download(as_id('0B98qpkK5EJemRXpfa2lONlFRSms'), path=file.path(PSDS_PATH, 'data', 'lc_loans.csv'), overwrite=TRUE)
drive_download(as_id('1J96vAqyh92VIeh7kBFm1NBfZcvx8wp2s'), path=file.path(PSDS_PATH, 'data', 'full_train_set.csv'), overwrite=TRUE)

## Import a sample of 200 records from lending club loan data
drive_download(as_id('0B98qpkK5EJemd0JnQUtjb051dTA'), path=file.path(PSDS_PATH, 'data', 'loan200.csv'), overwrite=TRUE)

## Import a sample of 3000 records from lending club loan data
drive_download(as_id('0B98qpkK5EJemQXYtYmJUVkdsN1U'), path=file.path(PSDS_PATH, 'data', 'loan3000.csv'), overwrite=TRUE)

## Import a complete set of records from lending club loan data
drive_download(as_id('0B98qpkK5EJemZzdoQ2I3SWlBYzg'), path=file.path(PSDS_PATH, 'data', 'loan_data.csv'), overwrite=TRUE)

## Import loans income data
drive_download(as_id('0B98qpkK5EJemRXVld0NSbWhYNVU'), path=file.path(PSDS_PATH, 'data', 'loans_income.csv'), overwrite=TRUE)

## Import session_times data
drive_download(as_id('0B98qpkK5EJemOC0xMHBTTEowYzg'), path=file.path(PSDS_PATH, 'data', 'web_page_data.csv'), overwrite=TRUE)

## Import four_sessions data
drive_download(as_id('0B98qpkK5EJemOFdZM1JsaEF0Mnc'), path=file.path(PSDS_PATH, 'data', 'four_sessions.csv'), overwrite=TRUE)

## Import click_rate data
drive_download(as_id('0B98qpkK5EJemVHB0ZzdtUG9SeTg'), path=file.path(PSDS_PATH, 'data', 'click_rates.csv'), overwrite=TRUE)

## Import imanishi data
drive_download(as_id('0B98qpkK5EJemZTJnUDd5Ri1vRDA'), path=file.path(PSDS_PATH, 'data', 'imanishi_data.csv'), overwrite=TRUE)

## Import lung disease data
drive_download(as_id('0B98qpkK5EJemb25YYUFJZnZVSnM'), path=file.path(PSDS_PATH, 'data', 'LungDisease.csv'), overwrite=TRUE)

## Import Zillow's county level ZHVI 
drive_download(as_id('0B98qpkK5EJemWGRWOEhYN1RabVk'), path=file.path(PSDS_PATH, 'data', 'County_Zhvi_AllHomes.csv'), overwrite=TRUE)

## Import King county house sales data
drive_download(as_id('0B98qpkK5EJemVTRRN0dLakxwTmM'), path=file.path(PSDS_PATH, 'data', 'house_sales.csv'), overwrite=TRUE)

if(FALSE){
  
  library(RCurl)
  download_from_google_drive <- function(id, fname, path)
  {
    url <- sprintf("https://drive.google.com/uc?export=download&id=%s", id)
    data <- getBinaryURL(url, followlocation = TRUE, ssl.verifypeer = FALSE)
    dest <- file.path(path, 'data', fname)
    writeBin(data, dest, useBytes = TRUE)
  }
  ## Import state data
  download_from_google_drive(id="0B98qpkK5EJembFc5RmVKVVJPdGc", fname='state.csv', path=PSDS_PATH)
  
  ## Airline data
  download_from_google_drive(id="0B98qpkK5EJemcmZYX2VhMHBXelE", fname='dfw_airline.csv', path=PSDS_PATH)
  download_from_google_drive(id="0B98qpkK5EJemMzZYZHZJaF9va0U", fname='airline_stats.csv', path=PSDS_PATH)
  
  ## Import stock data
  download_from_google_drive('0B98qpkK5EJemV2htZWdhVFRMNlU', fname='sp500_px.csv', path=PSDS_PATH)
  download_from_google_drive('0B98qpkK5EJemY0U0N1N6a21lUzA', fname='sp500_sym.csv', path=PSDS_PATH)
  
  ## Import KC housing tax data
  download_from_google_drive('0B98qpkK5EJemck5VWkszN3F3RGM', fname='kc_tax.csv', path=PSDS_PATH)
  
  ## Import lending club loan data
  download_from_google_drive('0B98qpkK5EJemRXpfa2lONlFRSms', fname='lc_loans.csv', path=PSDS_PATH)
  download_from_google_drive('1J96vAqyh92VIeh7kBFm1NBfZcvx8wp2s', fname='full_train_set.csv', path=PSDS_PATH)
  
  ## Import a sample of 200 records from lending club loan data
  download_from_google_drive('0B98qpkK5EJemd0JnQUtjb051dTA', fname='loan200.csv', path=PSDS_PATH)
  
  ## Import a sample of 3000 records from lending club loan data
  download_from_google_drive('0B98qpkK5EJemQXYtYmJUVkdsN1U', fname='loan3000.csv', path=PSDS_PATH)
  
  
  ## Import a complete set of records from lending club loan data
  download_from_google_drive('0B98qpkK5EJemZzdoQ2I3SWlBYzg', fname='loan_data.csv', path=PSDS_PATH)
  
  ## Import loans income data
  download_from_google_drive('0B98qpkK5EJemRXVld0NSbWhYNVU', fname='loans_income.csv', path=PSDS_PATH)
  
  ## Import session_times data
  download_from_google_drive('0B98qpkK5EJemOC0xMHBTTEowYzg', fname='web_page_data.csv', path=PSDS_PATH)
  
  ## Import four_sessions data
  download_from_google_drive('0B98qpkK5EJemOFdZM1JsaEF0Mnc', fname='four_sessions.csv', path=PSDS_PATH)
  
  ## Import click_rate data
  download_from_google_drive('0B98qpkK5EJemVHB0ZzdtUG9SeTg', fname='click_rates.csv', path=PSDS_PATH)
  
  ## Import imanishi data
  download_from_google_drive('0B98qpkK5EJemZTJnUDd5Ri1vRDA', fname='imanishi_data.csv', path=PSDS_PATH)
  
  ## Import lung disease data
  download_from_google_drive('0B98qpkK5EJemb25YYUFJZnZVSnM', fname='LungDisease.csv', path=PSDS_PATH)
  
  ## Import Zillow's county level ZHVI 
  download_from_google_drive('0B98qpkK5EJemWGRWOEhYN1RabVk', fname='County_Zhvi_AllHomes.csv', path=PSDS_PATH)
  
  ## Import King county house sales data
  download_from_google_drive('0B98qpkK5EJemVTRRN0dLakxwTmM', fname='house_sales.csv', path=PSDS_PATH)
  
}
