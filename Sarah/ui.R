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
shinyUI(
  dashboardPage (
    ###########################################
    dashboardHeader(title = 'Energy Price Forecast'),
    dashboardSidebar(
      # sidebarUserPanel('ATOS_LOGO',
      #                  image = 'https://github.com/sarah-ewing/energy_forecasting/blob/master/Sarah/Energy_ts/data/ATOS_LOGO.png'),
      # 
      #########################################
      sidebarMenu(
        menuItem('Data', tabName = 'dt', icon = icon('database')),
        
        
        
        #########################################
        ## MAP
        menuItem('Map', tabName = 'tm', icon = icon('map')),
        ## MAP
        
        #########################################
        menuItem('TimeSeries', tabName = 'ts', icon = icon('line-chart'))
      ),###sidebarMenu
      
      ####### date range for time series
      sliderInput(inputId = "prediction_date_range",
                  label = "Date:",
                  min = as.Date('2015-09-01', origin = "1970-01-01"),
                  max = as.Date('2015-10-20', origin = "1970-01-01"),
                  value = c(as.Date('2015-09-20', origin = "1970-01-01"),
                            as.Date('2015-10-20', origin = "1970-01-01")),
                  step = 5),
      
      # the chunk below makes a group of checkboxes
      checkboxGroupInput(inputId = "ModelTypes", 
                         label = h3("Models"), 
                         choices = list("ARIMA" = 1, "DNN" = 3, "CNN" = 4, "RNN" = 5),
                         selected = NULL),
      
      hr(),
      fluidRow(column(3, verbatimTextOutput("value")))
    
      
      ), ###dashboardSidebar
    #########################################
    dashboardBody(tabItems(
      tabItem(tabName = 'dt',
              tabBox(
                title = tagList(shiny::icon("line-chart"), "Time Series"),
                width = 9,
                
                tabPanel('Price',
                         height = "700px",
                         width = "900px",
                         plotOutput('price_ts', 
                                    height = "700px",
                                    width = "1000px"))
              
                )),## tabItem(tabName = 'dt'
      #########################################
      tabItem(tabName = 'tm',
              tabBox(
                title = tagList(shiny::icon("map"), "Map"),
                
                tabPanel('Weather Stations in Spain & Portugal',
                         plotOutput('MAP', 
                                    height = "700px",
                                    width = "700px"))
              )), ## tabItem(tabName = 'tm'
      #########################################
      tabItem(tabName = 'ts',
              tabBox(
                title = tagList(shiny::icon("line-chart"), "Time Series"),
                height = "700px",
                width = "1500px",
                tabPanel('Forecast Price',
                         height = "700px",
                         width = "1100px",
                         plotOutput('fore', 
                                    height = "700px",
                                    width = "1100px"))
              )) ## tabItem(tabName = 'ts')
      #########################################
    ))
  )) ## shinyUI & dashboardPage
