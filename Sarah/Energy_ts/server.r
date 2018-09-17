##########################################
#
#  File Name: server.r
#
#  Author: Sarah Ewing
#
#  Description: This file is the interactive calcs
#         of the Shiny app
#
##########################################

# Define server logic required to draw a histogram ----
server <- function(input, output) {
  
  output$distPlot <- renderPlot({
   
    plotty = raw[which(raw$prediction_date > input$prediction_date_range[1]
                       & raw$prediction_date < input$prediction_date_range[2]
                       ),]

    ggplot(plotty, aes(x = prediction_dttm, y = price)) +
      geom_line(size = 1) +
      scale_color_manual(values = c("#00AFBB")) +
      theme_minimal() +
      stat_smooth(color = "#FC4E07", method = "loess")
    
  })
  
}