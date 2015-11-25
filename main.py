from zipline.api import order_target, record, symbol, history, add_history
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from random import random
import numpy as np
import talib
from datetime import datetime
import pytz
from zipline.algorithm import TradingAlgorithm
from zipline.utils.factory import load_bars_from_yahoo
import random
from features import *
import math
import matplotlib.pyplot as plt

plotData = {}
revenue = []


error = {}

class StockPredictor:

	def __init__(self):
		self.windowData = {'open':[], 'features':[]}
		self.windowLength = 50
		self.featureWindowLength = 25
		self.amtOwned = 0
		self.featureCount = 16

		self.network = RecurrentNetwork()

		self.network.addInputModule(LinearLayer(self.featureCount, name='in'))
		self.network.addModule(TanhLayer(25, name='hidden_1'))
		self.network.addModule(TanhLayer(25, name='hidden_2'))
		self.network.addModule(TanhLayer(25, name='hidden_3'))
		self.network.addOutputModule(TanhLayer(1, name='out'))

		self.network.addConnection(FullConnection(self.network['in'], self.network['hidden_1'], name='c_1'))
		self.network.addConnection(FullConnection(self.network['hidden_1'], self.network['hidden_2'], name='c_2'))
		self.network.addConnection(FullConnection(self.network['hidden_2'], self.network['hidden_3'], name='c_3'))
		self.network.addConnection(FullConnection(self.network['hidden_3'], self.network['out'], name='c_4'))

		self.network.sortModules();



	def AddData(self, data):
		self.windowData['open'].append(data['open'])
		if len(self.windowData['open']) > self.windowLength:
			self.windowData['open'].pop(0);

		if len(self.windowData['open']) == self.windowLength:
			self.windowData['features'].append(self.CalculateFeatures())
			if len(self.windowData['features']) > self.featureWindowLength:
				self.windowData['features'].pop(0);
		return

	def IsReady(self):
		return len(self.windowData['features']) == self.featureWindowLength

	#This will take the current days feature and train against the value
	def Train(self, value):
		features = self.GetFeatures();
		ds = SupervisedDataSet(self.featureCount,1)
		ds.addSample(features,(value))
		trainer = BackpropTrainer(self.network, ds)
		return trainer.train()

	def CalculateFeatures(self):
		features = []
		#print "WindowData: ", self.windowData['open']
		features.append(self.windowData['open'][-1])
		features.append(sum(self.windowData['open'])/len(self.windowData['open']))
		features.append(np.var(np.asarray(self.windowData['open'])))
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-1])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-2])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-3])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-4])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-5])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-6])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-7])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 5)[-8])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 10)[-1])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 10)[-2])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 10)[-3])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 10)[-4])
		features.append(talib.SMA(np.asarray(self.windowData['open']), timeperiod = 25)[-1])
		differences  = 0;
		for x in range(1, len(self.windowData['open'])):
			differences += (self.windowData['open'][x-1]-self.windowData['open'][x])
		features.append(differences/(len(self.windowData['open'])-1))
		return features

	def GetFeatures(self):
		features = []
		for y in range(0, self.featureCount):
			feature = []
			for x in range(0, len(self.windowData['features'])):
				feature.append(self.windowData['features'][x][y])
			if not feature:
				featureValue = 0;
			else:
				featureValue = (feature[-1]-np.mean(np.asarray(feature)))/np.std(np.asarray(feature))
			#featureValue = (feature[-1]-min(feature))/(max(feature)-min(feature))
			features.append(featureValue)
		return features

	def Activation(self):
		features = self.GetFeatures()
		activation = self.network.activate(features)
		#print "Activation: ", activation
		return activation


def initialize(context):
	# context.tickerList = ['NX','QTM','QTWW','QRHC','QUIK',
	# 'ZQK','QNST','QUMU','RLGT','ROIAK','RSYS','RDNT','RLOG',
	# 'RPTP','RAVE','RICK','RCMT','RLOC','RNWK']

	context.tickerList = ['ACTA', 'AKAO']

	for ticker in context.tickerList:
		plotData[ticker] = []
		error[ticker] = []

	context.predictors = {}
	for ticker in context.tickerList:
		context.predictors[ticker] = StockPredictor()
	context.currentDay = 0;
	context.trainingLength = 100;
	context.capital = 0;


def handle_data(context, data):
	context.currentDay += 1
	for ticker in context.tickerList:
		plotData[ticker].append(data[ticker]['close'])
		if data[ticker]['open'] == 0:
			continue
		context.predictors[ticker].AddData(data[ticker])
		change = data[ticker]['close'] - data[ticker]['open']
		if context.currentDay > context.trainingLength:
			activation = context.predictors[ticker].Activation();
			print ticker,": ",activation
			if activation > .25:
				#print "Buy: ", data[ticker]['open'], ". Sell: ",data[ticker]['close']
				context.capital += (math.floor(((activation-.25)/.75)*10))*change
			elif activation < -.25:
				#print "Sell: ", data[ticker]['open'], ". Buy: ",data[ticker]['close']
				context.capital += (math.floor(((activation-.25)/.75)*10))*-1*change

		if context.predictors[ticker].IsReady():
			if change > 0:
				error[ticker].append(context.predictors[ticker].Train(1))
			elif change < 0:
				error[ticker].append(context.predictors[ticker].Train(-1))
	revenue.append(context.capital)
	print "Day ",context.currentDay," profits: ", context.capital


if __name__ == '__main__':
	tickerFile = open("data/russel_microcap.csv");
	tickerList = [];
	for line in tickerFile:
		tickerList.append(line.split(' ')[-1].strip('\n'))
	tickerFile.close();
	print len(tickerList)," ticker symbols loaded"

	# Set the simulation start and end dates.
	start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
	end = datetime(2015, 8, 1, 0, 0, 0, 0, pytz.utc)

	# Load price data from yahoo.
	data = load_bars_from_yahoo(stocks=tickerList, indexes={}, start=start,end=end)
	#
	print "Shape of the data: ",data.shape
	print "Cleaning Data"
	data.fillna(0, inplace=True)
	print 'Data Cleaned'

	# Create and run the algorithm.
	algo = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
							 identifiers=tickerList)
	results = algo.run(data)

	 # Plot the portfolio and asset data.
	 # analyze(results=results)
	plt.figure(1)
	plt.title('Algorithm performance')
	plt.plot(revenue)
	plt.ylabel('Profit')
	plt.xlabel('Day')

	plt.figure(2)
	plt.title('Historic Open Pricing')
	for ticker in plotData.iterkeys():
		plt.plot(plotData[ticker], label=ticker)
	plt.ylabel('Price')
	plt.xlabel('Day')
	plt.legend(loc='upper right')

	plt.figure(3)
	for ticker in error.iterkeys():
		plt.plot(error[ticker], label=ticker)

	plt.show()
