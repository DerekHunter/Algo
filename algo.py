from pandas_datareader import data, wb
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import talib

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError

from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer


class Model():

	def __init__(self):
		self.windowLength = 10
		self.trainingPeriod = 50

	def Predict(self, ticker, day):
		endDay = day-datetime.timedelta(1)
		startDay = endDay - datetime.timedelta(self.trainingPeriod)
		try:
			stockData = data.DataReader(ticker, 'yahoo', startDay, endDay)
		except:
			return [0]

		rawTrainFeatures = []
		rawTrainResponses = []
		for currentDay in range(self.windowLength, len(stockData)):
			window = stockData[currentDay-self.windowLength:currentDay]
			currentPrice = stockData.iloc[currentDay]['Open']
			response = stockData.iloc[currentDay]['Close']
			rawTrainFeatures.append(self.GetFeature(window))
			rawTrainResponses.append(response)

		rawTestFeatures = self.GetFeature(stockData[len(stockData)-self.windowLength:len(stockData)])

		# normalTrainFeatures, normalTestFeatures = self.NormalizeFeatures(rawTrainFeatures, rawTestFeatures)
		alldata = SupervisedDataSet(len(rawTrainFeatures[0]), 1)
		for index in range(0, len(rawTrainFeatures)):
			alldata.addSample(rawTrainFeatures[index],[rawTrainResponses[index]])

		self.network = buildNetwork(alldata.indim, (alldata.indim+alldata.outdim)/2, alldata.outdim, hiddenclass=SigmoidLayer, outclass=LinearLayer)
		trainer = BackpropTrainer(self.network, dataset=alldata)
		activations = []
		for i in range(50):
			for x in range(5):
				trainer.train()
		return float(self.network.activate(rawTestFeatures))


	def GetFeature(self, data):
		features = []
		closePrice = np.asarray(data['Close'])
		openPrice = np.asarray(data['Open'])
		highPrice = np.asarray(data['High'])
		lowPrice = np.asarray(data['Low'])
		volume = np.asarray(data['Volume'])

		features.append(closePrice[-1])
		features.append(closePrice[-2])
		features.append(closePrice[-3])
		features.append(closePrice[-4])
		features.append(closePrice[-5])

		############## MOMENTUM INDICATORS #################
		# features.append(talib.ADX(highPrice,lowPrice,closePrice)[-1])
		# features.append(talib.APO(closePrice)[-1])
		# features.append(talib.AROONOSC(highPrice, lowPrice)[-1])
		# features.append(talib.BOP(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CCI(highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CMO(closePrice)[-1])
		# features.append(talib.DX(highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CMO(closePrice)[-1])
		# features.append(talib.MOM(closePrice)[-1])
		# features.append(talib.ROCR(closePrice)[-1])
		# features.append(talib.TRIX(closePrice)[-1])
		# features.append(talib.ULTOSC(highPrice, lowPrice, closePrice)[-1])
		#
		# ############## VOLITILITY INDICATORS #################
		# features.append(talib.ATR(highPrice, lowPrice, closePrice)[-1])

		# ############## PATTERN INDICATORS #################
		# features.append(talib.CDL2CROWS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3BLACKCROWS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3INSIDE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3LINESTRIKE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3OUTSIDE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3STARSINSOUTH(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDL3WHITESOLDIERS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLABANDONEDBABY(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLADVANCEBLOCK(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLBELTHOLD(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLBREAKAWAY(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLCLOSINGMARUBOZU(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLCONCEALBABYSWALL(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLCOUNTERATTACK(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLDARKCLOUDCOVER(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLDOJI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLDOJISTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLDRAGONFLYDOJI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLENGULFING(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLEVENINGDOJISTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLEVENINGSTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLGAPSIDESIDEWHITE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLGRAVESTONEDOJI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHAMMER(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHANGINGMAN(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHARAMI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHARAMICROSS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHIGHWAVE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHIKKAKE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHIKKAKEMOD(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLHOMINGPIGEON(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLIDENTICAL3CROWS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLINNECK(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLINVERTEDHAMMER(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLKICKING(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLKICKINGBYLENGTH(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLLADDERBOTTOM(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLLONGLEGGEDDOJI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLLONGLINE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLMARUBOZU(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLMATCHINGLOW(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLMATHOLD(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLMORNINGDOJISTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLMORNINGSTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLONNECK(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLPIERCING(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLRICKSHAWMAN(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLRISEFALL3METHODS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSEPARATINGLINES(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSHOOTINGSTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSHORTLINE(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSPINNINGTOP(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSTALLEDPATTERN(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLSTICKSANDWICH(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLTAKURI(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLTASUKIGAP(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLTHRUSTING(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLTRISTAR(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLUNIQUE3RIVER(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLUPSIDEGAP2CROWS(openPrice, highPrice, lowPrice, closePrice)[-1])
		# features.append(talib.CDLXSIDEGAP3METHODS(openPrice, highPrice, lowPrice, closePrice)[-1])
		return features


	def NormalizeFeatures(self, train, test):
		normal = []
		dataTranspose = map(list, zip(*train))
		i = 0;
		for feature in dataTranspose:
			mean = np.mean(np.asarray(feature))
			std = np.std(np.asarray(feature))
			test[i] = (test[i]-mean)/std
			normal.append(map(lambda x: (x-mean)/std, feature))
			i+=1
		return map(list, zip(*normal)), test
