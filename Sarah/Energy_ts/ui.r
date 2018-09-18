##########################################
#
#  File Name: ui.r
#
#  Author: Sarah Ewing
#
#  Description: This file is the layout of the
#           Shiny app
#
##########################################

# Define UI for app that draws a histogram ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("Energy Forecast"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
       sliderInput(inputId = "prediction_date_range",
                  label = "Date:",
                  min = as.Date('2015-01-01', origin = "1970-01-01"),
                  max = as.Date('2015-10-20', origin = "1970-01-01"),
                  value = c(as.Date('2015-01-01', origin = "1970-01-01"),
                            as.Date('2015-10-20', origin = "1970-01-01")),
                  step = 5)
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Histogram ----
      plotOutput(outputId = "distPlot")
      
    )
  )
)