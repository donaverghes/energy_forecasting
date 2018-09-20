rm(list = ls())

getwd()
ls()
list.files(getwd())

library(googleVis)
library(quantmod)
library(shiny)
library(ggplot2)
library(forecast)
library(TTR)
library(zoo)
library(shinydashboard)
library(DT)
library(tm)
library(ggmap)

library(RCurl)

## RNN
hRNN <- getURL('https://raw.githubusercontent.com/sarah-ewing/energy_forecasting/master/discovery/rnnn.csv')
RNN <- read.csv(text = hRNN, header=F)
colnames(RNN) = c("Dttm", "price","P_price")
RNN$Dttm = as.POSIXct(RNN$Dttm,
                        tz = "UTC",
                        origin = "1970-01-01",
                        format="%Y-%m-%d %H:%M:%S")
RNN_P_price = c(rep(0,5554),RNN$P_price)

## DNN
hDNN <- getURL('https://raw.githubusercontent.com/sarah-ewing/energy_forecasting/master/discovery/dnn.csv')
DNN <- read.csv(text = hDNN, header=F)
colnames(DNN) = c("Dttm", "price","P_price")
DNN$Dttm = as.POSIXct(DNN$Dttm,
                      tz = "UTC",
                      origin = "1970-01-01",
                      format="%Y-%m-%d %H:%M:%S")
DNN_P_price = c(rep(0,5554),DNN$P_price)


## CNN
hCNN <- getURL('https://raw.githubusercontent.com/sarah-ewing/energy_forecasting/master/discovery/cnn.csv')
CNN <- read.csv(text = hCNN, header=F)
colnames(CNN) = c("Dttm", "price","P_price")
CNN$Dttm = as.POSIXct(CNN$Dttm,
                      tz = "UTC",
                      origin = "1970-01-01",
                      format="%Y-%m-%d %H:%M:%S")
CNN_P_price = c(rep(0,5554),CNN$P_price)


### Autoregressive Informed Moving Average
ARIMA = read.csv('ARIMA.csv')
colnames(ARIMA) = c("ID", "Dttm", "price","P_price")
ARIMA$Dttm = as.POSIXct(ARIMA$Dttm,
                        tz = "UTC",
                        origin = "1970-01-01",
                        format="%Y-%m-%d %H:%M:%S")
ARIMA$Date = as.Date(ARIMA$Dttm)
ARIMA = ARIMA[order(ARIMA$Dttm),]
rownames(ARIMA) = NULL

raw = as.data.frame(cbind(ARIMA[,2:5],DNN_P_price, CNN_P_price, RNN_P_price))
colnames(raw) = c("predicted_dttm","price", "p_price",
                  "predicted_date", "DNN", "CNN", "RNN")
raw$day_week = weekdays(as.Date(raw$predicted_date))
rm(RNN, ARIMA, hRNN, RNN_P_price)

t(aggregate(raw$price, by=list(raw$day_week), FUN=mean, na.rm=TRUE))

ID = 1:18
lon = c(-8,-6,-4,-2,-8,-6,-4,-2,-8,-6,-4,-2, 0,-8,-6,-4,-2, 0)
lat = c(37, 37, 37, 37, 39, 39, 39, 39, 41, 41, 41, 41, 41, 43, 43, 43, 43, 43)
df = as.data.frame(cbind(ID, lat, lon))
rm(ID, lon, lat)

#library(googleCloudStorageR)
#gcs_auth(new_user = TRUE)

## set bucket via environment
#Sys.setenv("GCS_DEFAULT_BUCKET" = "sarah-bucket")
#gcs_get_global_bucket()
#gcs_auth()
