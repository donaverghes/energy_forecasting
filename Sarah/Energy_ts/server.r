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
shinyServer(function(input, output, session) {
  #####################
  ## map
  output$MAP = renderPlot({
    m <- get_map(
      location = c(lon = -3, lat = 40),
      zoom = 6,
      maptype = "terrain",
      source = "google"
    )
    
    ggmap(m) + geom_point(
      data = df,
      aes(lon, lat),
      size = 3,
      alpha = 0.7,
      color = "red"
    ) + labs(x = "Longitude",
             y = "Latitude",
             title = "")+
      theme(text = element_text(size=20))
  })
  
  ##########################
  ## auto regressive model
  output$price_ts = renderPlot({
    plotty = raw[which(
      raw$predicted_date > input$prediction_date_range[1]
      &
        raw$predicted_date < input$prediction_date_range[2]
    ),]
    
    ggplot(plotty, aes(x = predicted_dttm, y = price)) +
      geom_line(size = 1) +
      scale_color_manual(values = c("#00AFBB")) +
      theme_minimal() +
      stat_smooth(color = "#FC4E07", method = "loess")+
      xlab("Date")+
      ylab("Price")+
      theme(text = element_text(size=20))
  })
  
  ###########################
  ### Time series > Forecast > Forecast > ARIMA model
  output$fore = renderPlot({
    plotty = raw[which(
      raw$predicted_date > input$prediction_date_range[1]
      &
        raw$predicted_date < input$prediction_date_range[2]
    ),]
    cols <- isolate(input$ModelTypes)
    plot(
      x = plotty$predicted_dttm,
      y = plotty$price,
      type = "l",
      ylim = c(0, 100),
      xlab = "Date",
      ylab = "Price",
      cex=3 ,
      lwd=2
    )
    legend(
      "topright",
      legend = c("Price", "ARIMA",
                 #"Linear", 
                 "DNN",
                 "CNN", "RNN"),
      col = c("black", "red", 
              #"green", 
              "orange", "purple",
              "blue"),
      lty = 1,
      cex = 1.5,
      lwd=2
    )
    if (is.null(input$ModelTypes) == TRUE |
        length(input$ModelTypes) < 5) {
      lines(
        x = plotty$predicted_dttm,
        y = rep(-100, length(plotty$predicted_dttm)),
        col = "white",
        lwd=2
      )
      ModelTypes1 = c(input$ModelTypes, rep(0, (5 - length(input$ModelTypes))  ))
    }
    if(length(input$ModelTypes)==5){ModelTypes1 = input$ModelTypes}
    if (is.null(input$ModelTypes) == FALSE) {
      if (min(input$ModelTypes) == 1) {
        ### ARIMA
        lines(
          x = plotty$predicted_dttm,
          y = plotty$p_price,
          col = "red",
          lwd=2
        )
      }
      # if (ModelTypes1[1] == 2 | ModelTypes1[2] == 2 | ModelTypes1[3] == 2 |
      #     ModelTypes1[4] == 2 | ModelTypes1[5] == 2) {
      #   ### Linear
      #   lines(
      #     x = plotty$predicted_dttm,
      #     y = rep(-100, length(plotty$predicted_dttm)),
      #     col = "green",
      #     lwd=2
      #   )
      # }
      if (ModelTypes1[1] == 3 | ModelTypes1[2] == 3 | ModelTypes1[3] == 3 |
          ModelTypes1[4] == 3 | ModelTypes1[5] == 3) {
        ### DNN
        lines(x = plotty$predicted_dttm,
              y = plotty$DNN,
              col = "orange")
      }
      if (ModelTypes1[1] == 4 | ModelTypes1[2] == 4 | ModelTypes1[3] == 4 |
          ModelTypes1[4] == 4 | ModelTypes1[5] == 4) {
        ### CNN
        lines(
          x = plotty$predicted_dttm,
          y = plotty$CNN,
          col = "purple",
          lwd=2
        )
      }
      if (max(input$ModelTypes) == 5) {
        ### RNN
        lines(x = plotty$predicted_dttm,
              y = plotty$RNN,
              col = "blue",
              lwd=2)
      }
    }
    
  })
  
})