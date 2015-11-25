from pandas_datareader import data, wb
import numpy as np
import matplotlib.pyplot as plt
import datetime

end = datetime.date.today()
start = end - datetime.timedelta(100)

## API FOR CURRENT OPEN PRICE FOR THE date
## http://dev.markitondemand.com/MODApis/Api/v2/Quote?symbol=ge

stockData = data.DataReader("nflx", 'google', start, end)
print stockData

trainingData = []
windowLength = 10

# Current Day is the day in which prediction in desired
# window is the data for the previous 'windowLength' days
for currentDay in range(windowLength, len(stockData['Close'])):
	print stockData.iloc[currentDay]
	window = stockData[currentDay-windowLength:currentDay]['Close']
	change = stockData.iloc[currentDay]['Close'] - stockData.iloc[currentDay]['Open']
	if change > 0:
		response = 1
	else:
		response  = 0
	#
	# trainingData.append([[feat_1, feat_2, ... , feat_n],[response]])
	#

# sklearnModel.train([features], [responses])
