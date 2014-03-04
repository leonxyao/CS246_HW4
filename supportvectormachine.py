#!/usr/bin/python
import pdb
import numpy as np 
import math
import time

numfeatures=122
numvectors=6414#6000
learnrate=0.0001
epsilon=0.25
regparameter=100
entirefeaturesfile="features.txt"
entiretargetfile="target.txt"
trainingfeaturesfile="features.train.txt"
trainingtargetfile="target.train.txt"
testfeaturesfile="features.test.txt"
testtargetfile="target.test.txt"
regparvalues=[1,10,50,100,200,300,400,500]
batchsize=20

featuredata=None
testfeaturedata=None
outputdata=None

def calculatePercentCost(weight, bias,costfuncvalues,numiterations):
	"""Calculates percent cost for batch gradient descent."""
	if numiterations==0: #to handle initial case when starting algorithm
		return 10
	else:
		iterdiff=costfuncvalues[numiterations]-costfuncvalues[numiterations-1]
		return math.fabs(iterdiff)*100/costfuncvalues[numiterations-1]

def calculateCostFunction(weight,bias):
	"""Calculates cost function for given weight and bias."""
	errorvals=0.0
	numvectors=featuredata.shape[0]
	for i in range(numvectors):
		confidence=1-outputdata[i]*(sum(featuredata[i]*weight)+bias)
		errorvals+=max(0,confidence)
	return regparameter*errorvals+0.5*sum(weight**2)

def calculateCostFunctionStoch(weight,bias):
	"""Calculates cost function for given weight and bias."""
	errorvals=0.0
	numvectors=featuredata.shape[0]
	for i in range(numvectors):
		confidence=1-featuredata[i][numfeatures]*(sum(featuredata[i][0:numfeatures]*weight)+bias)
		errorvals+=max(0,confidence)
	return regparameter*errorvals+0.5*sum(weight**2)

def calculateError(weight, bias):
	numincorrect=0
	for i in range(testfeaturedata.shape[0]):
		output=0
		if (sum(testfeaturedata[i,0:numfeatures]*weight)+bias)<0:
			output=-1.0
		else:
			output=1.0
		if output!=testfeaturedata[i][numfeatures]:
			numincorrect+=1
	print 'percent error', float(numincorrect)/testfeaturedata.shape[0]*100
	
###################################################################################
#Batch Method

def calculateBatchWeightGradient(weight, bias,index):
	numvectors=featuredata.shape[0]
	gradientval=0
	for i in range(numvectors):
		confidence=outputdata[i]*(sum(featuredata[i,:]*weight)+bias)
		if confidence<1:
			gradientval+=-1*outputdata[i]*featuredata[i][index]
	return float(regparameter)*gradientval

def calculateBatchBiasGradient(weight,bias):
	gradientval=0
	numvectors=featuredata.shape[0]
	for i in range(numvectors):
		confidence=outputdata[i]*(sum(featuredata[i,:]*weight)+bias)
		if confidence<1:
			gradientval+=-1*outputdata[i]
	return float(regparameter)*gradientval

def runBatchGD():
	weight=np.zeros(numfeatures)
	bias=0
	numiterations=0
	
	initialcost=calculateCostFunction(weight,bias) #first cost function value
	costfuncvalues=np.zeros(numfeatures)
	costfuncvalues[0]=initialcost

	while calculatePercentCost(weight,bias,costfuncvalues, numiterations)>epsilon:
		for index in range(numfeatures):
			weightgrad=calculateBatchWeightGradient(weight,bias,index)
			weight[index]+=-1*learnrate*weightgrad
		biasgrad=calculateBatchBiasGradient(weight,bias)
		bias+=-1*learnrate*biasgrad
		numiterations+=1
		costval=calculateCostFunction(weight,bias)
		costfuncvalues[numiterations]=costval
		print numiterations, ' ', costval

###################################################################################

###################################################################################
#Stochastic Method


def calculateStochasticConvergence( deltacost, numiterations):
	if numiterations==0:
		return 5
	else:
		return deltacost[numiterations]

def calculateStochWeightGrad(weight,bias, vecindex, index):
	gradientval=0
	confidence=featuredata[vecindex][numfeatures]*(sum(featuredata[vecindex,0:numfeatures]*weight)+bias)
	if confidence<1:
		gradientval+=-1*featuredata[vecindex][numfeatures]*featuredata[vecindex][index]
	return float(regparameter)*gradientval
	
def calculateStochBiasGrad(weight,bias,vecindex):
	gradientval=0
	confidence=featuredata[vecindex][numfeatures]*(sum(featuredata[vecindex,0:numfeatures]*weight)+bias)
	if confidence<1:
		gradientval+=-1*featuredata[vecindex][numfeatures]
	return float(regparameter)*gradientval
	
def runStochasticGD():
	np.random.shuffle(featuredata)
	weight=np.zeros(numfeatures)
	bias=0
	numiterations=0
	vecindex=0
	numvectors=featuredata.shape[0]

	initialcost=calculateCostFunctionStoch(weight,bias) #first cost function value
	costfuncvalues=np.zeros(5000)
	costfuncvalues[0]=initialcost

	deltacost=np.zeros(5000) #initialize values to be really big

	while calculateStochasticConvergence(deltacost,numiterations)>epsilon:
		for index in range(numfeatures):
			weightgrad=calculateStochWeightGrad(weight,bias,vecindex,index)
			weight[index]+=-1*learnrate*weightgrad
		biasgrad=calculateStochBiasGrad(weight,bias,vecindex)
		bias+=-1*learnrate*biasgrad
		numiterations+=1
		vecindex=vecindex%numvectors+1
		costval=calculateCostFunctionStoch(weight,bias)
		costfuncvalues[numiterations]=costval
		deltacost[numiterations]=0.5*deltacost[numiterations-1]+0.5*calculatePercentCost(weight, bias, costfuncvalues,numiterations)
		print costval
	return (weight,bias)

###################################################################################

###################################################################################
#Mini Batch Method

def calculateMiniBatchConvergence( deltacost, numiterations):
	if numiterations==0:
		return 5
	else:
		return deltacost[numiterations]

def calculateMiniBatchWeightGrad(weight,bias, vecindex, index):
	gradientval=0
	for i in range(vecindex*batchsize+1,min(numvectors, (vecindex+1)*batchsize)):
		confidence=featuredata[i][numfeatures]*(sum(featuredata[i,0:numfeatures]*weight)+bias)
		if confidence<1:
			gradientval+=-1*featuredata[i][numfeatures]*featuredata[i][index]
	return float(regparameter)*gradientval
	
def calculateMiniBatchBiasGrad(weight,bias,vecindex):
	gradientval=0
	for i in range(vecindex*batchsize+1,min(numvectors, (vecindex+1)*batchsize)):
		confidence=featuredata[i][numfeatures]*(sum(featuredata[i,0:numfeatures]*weight)+bias)
		if confidence<1:
			gradientval+=-1*featuredata[i][numfeatures]
	return float(regparameter)*gradientval

def runMiniBatchGD():
	np.random.shuffle(featuredata)
	weight=np.zeros(numfeatures)
	bias=0
	numiterations=0
	vecindex=0
	numvectors=featuredata.shape[0]

	initialcost=calculateCostFunctionStoch(weight,bias) #first cost function value
	costfuncvalues=np.zeros(5000)
	costfuncvalues[0]=initialcost

	deltacost=np.zeros(5000) #initialize values to be really big

	while calculateMiniBatchConvergence(deltacost,numiterations)>epsilon:
		for index in range(numfeatures):
			weightgrad=calculateMiniBatchWeightGrad(weight,bias,vecindex,index)
			weight[index]+=-1*learnrate*weightgrad
		biasgrad=calculateMiniBatchBiasGrad(weight,bias,vecindex)
		bias+=-1*learnrate*biasgrad
		numiterations+=1
		vecindex=(vecindex+1)%((numvectors+batchsize-1)/batchsize)
		costval=calculateCostFunctionStoch(weight,bias)
		costfuncvalues[numiterations]=costval
		deltacost[numiterations]=0.5*deltacost[numiterations-1]+0.5*calculatePercentCost(weight, bias, costfuncvalues,numiterations)
		print costval

################################################################################################
runBatchGD()
# starttime=time.time()
# if __name__=="__main__":
# 	data=[]
# 	with open(trainingfeaturesfile,'r') as f:
# 		for line in f:	
# 			linestrip=line.strip().split(',')
# 			data.append([float(x) for x in linestrip])
# 	#featuredata=np.array(data) #stores feature vectors

# 	classdata=[]
# 	with open(trainingtargetfile,'r') as f:
# 		i=0
# 		for line in f:
# 			linestrip=line.strip()
# 			#classdata.append(float(linestrip))
# 			data[i].append(float(linestrip))
# 			i+=1

# 	#outputdata=np.array(classdata)
# 	featuredata=np.array(data)

# 	testdata=[]
# 	with open(testfeaturesfile,'r') as f:
# 		for line in f:	
# 			linestrip=line.strip().split(',')
# 			testdata.append([float(x) for x in linestrip])
# 	#featuredata=np.array(data) #stores feature vectors
# 	with open(testtargetfile,'r') as f:
# 		i=0
# 		for line in f:
# 			linestrip=line.strip()
# 			#classdata.append(float(linestrip))
# 			testdata[i].append(float(linestrip))
# 			i+=1
# 	testfeaturedata=np.array(testdata)
	
# 	weight,bias=runBatchGD()#runStochasticGD()
# 	calculateError(weight,bias)


# print 'totaltime:', time.time()-starttime