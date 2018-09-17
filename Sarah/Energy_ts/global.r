
setwd("/Users/sarahewing/Documents/OnBoarding/energy_forecasting/Energy_ts")
ls()
list.files(getwd())

library(shiny)
library(ggplot2)
library(zoo)

raw = read.csv("/Users/sarahewing/Documents/OnBoarding/energy_forecasting/Energy_ts/data/sarah_data.csv", header = F)

## the names of the variables
colnames(raw) = c("prediction_date", "wind_speed_100m", 
                  "wind_direction_100m", 
                  "temperature", 
                  "air_density",
                  "pressure", "precipitation", "wind_gust", "radiation", 
                  "wind_speed", "wind_direction", "price")
## format the date
raw$prediction_dttm = as.POSIXct(raw$prediction_date, tz = "UTC", 
                                 origin = "1970-01-01", format = "%Y-%m-%d %H:%M:%S")
raw$prediction_date = as.Date(substr(raw$prediction_date,1,11))
raw = raw[order(raw$prediction_date),]

source("ui.r")
source("server.r")
# Start Shiny app ----
shinyApp(ui = ui, server = server)
