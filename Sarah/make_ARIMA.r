x <- getURL('https://raw.githubusercontent.com/sarah-ewing/energy_forecasting/master/Sarah/Energy_ts/data/sarah_data.csv')
y <- read.csv(text = x, header=F)

mu_price = mean(y$V12)
sd_price = sd(y$V12)
standard_price = (y$V12 - mu_price)/sd_price

predicted_price = as.vector(0)
for(i in 0:(length(y$V12))){
  out = forecast(Arima(y = as.ts(y$V12[(0+i):(23+i)], method="ML"), 
                       include.drift = TRUE,
                       order = c(0, 1, 1)))
  predicted_price = c(predicted_price, out$mean[1])
  #print(paste(i, "-", y$V12[(0+i):(23+i)]))
}

#predicted_price2 = (predicted_price * sd_price) + mu_price

xx=as.vector(1:6646)
plotty = as.data.frame(cbind(xx,y[,c("V1","V12")], 
                             predicted_price))

write.csv(plotty, 'ARIMA.csv', row.names = FALSE)


plot(x = plotty$xx, y = plotty$V12, type = "l",
     ylim = c(0,100), xlim = c(6400,6466))
lines(x = plotty$xx, y = plotty$predicted_price, type = "l", col = "red")


