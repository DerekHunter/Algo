from pandas_datareader import data, wb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import os

from algo import Model

os.system('cls' if os.name == 'nt' else 'clear')

timePeriod = [datetime.date(2014, 5, 1), datetime.date(2014, 5, 31)]
tickerFile = open("tickers.csv");
tickers = []
for line in tickerFile:
	tickers.append(line.split(' ')[-1].strip('\n'))
tickerFile.close();

print "Downloading from Yahoo"
stockData = data.DataReader(tickers, 'yahoo', timePeriod[0], timePeriod[1])

model = Model()
os.system('cls' if os.name == 'nt' else 'clear')

capital = 10000
capitalTime = []
for currentDay in range(0, len(stockData['Close'])):
	today = stockData['Close'].iloc[currentDay].name.strftime('%Y-%m-%d').split('-')
	today = datetime.date(int(today[0]),int(today[1]),int(today[2]))
	print "Day#: ", currentDay
	print "DATE: ", today
	print "Capital: ", capital
	print "-"*196
	print "|     TICKER\t|\tOPEN\t|\tCLOSE\t|\tCHANGE\t|\tPREDICTION\t|\tERROR\t|\tPCT ERR\t|".expandtabs(15)
	print "-"*196
	buyList = []
	for ticker in tickers:
		priceChange = stockData['Close'].iloc[currentDay][ticker] - stockData['Open'].iloc[currentDay][ticker]
		prediction = model.Predict(ticker, today)
		if prediction > stockData['Open'].iloc[currentDay][ticker]:
			buyList.append(ticker)
		outputLine = "|      " + ticker + "\t|\t"
		outputLine += str(round(stockData['Open'].iloc[currentDay][ticker]*100)/100) + "\t|\t"
		outputLine += str(round(stockData['Close'].iloc[currentDay][ticker]*100)/100) + "\t|\t"
		outputLine += str(round(priceChange*100)/100) + "\t|\t"
		outputLine += str(round(prediction*100)/100) + "\t|\t"
		outputLine += str(round((prediction-stockData['Close'].iloc[currentDay][ticker])*1000)/1000) + "\t|\t"
		outputLine += str(round((prediction-stockData['Close'].iloc[currentDay][ticker])/stockData['Close'].iloc[currentDay][ticker]*1000)/10) + "\t|"
		print outputLine.expandtabs(15)
		print "-"*196

	if buyList:
		allocation = min([capital/len(buyList), .1*capital])
		for ticker in buyList:
			print "Buying ", ticker, "for ", stockData['Open'].iloc[currentDay][ticker]
			print "Selling ", ticker, "for ", stockData['Close'].iloc[currentDay][ticker]
			print "Profit ", stockData['Close'].iloc[currentDay][ticker] - stockData['Open'].iloc[currentDay][ticker]
			print "\n"
			capital += math.floor(allocation / stockData['Open'].iloc[currentDay][ticker])*(stockData['Close'].iloc[currentDay][ticker] - stockData['Open'].iloc[currentDay][ticker])
	else:
		"No Good Stocks"
	capitalTime.append(capital)
	os.system('cls' if os.name == 'nt' else 'clear')

print "Plotting"
plt.plot(capitalTime)
plt.show()
