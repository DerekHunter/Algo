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
	tickers.append(line.strip('\n'))
tickerFile.close();

model = Model()
os.system('cls' if os.name == 'nt' else 'clear')

day = datetime.date.today()-datetime.timedelta(1)
print (""+"-"*61).expandtabs(15)
print ("|\tPredictions for " + str(day)+"\t\t|").expandtabs(15)
print (""+"-"*61).expandtabs(15)
print "|\tTICKER\t|\tPREDICTION\t|".expandtabs(15)
print (""+"-"*61).expandtabs(15)
for ticker in tickers:
	prediction = model.Predict(ticker, day)
	print ("|\t"+ticker+"\t|\t"+str(round(prediction*100)/100)+"\t|").expandtabs(15)
	print "-"*61
