from math import log

def createDataSet():
	dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataset, labels	

# Shannon name comes from the father of Information Theory 'Claude Shannon'
def calcShannonEntropy(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key] / numEntries)
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt
# The higher the entropy the more mixed up the data is.	

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1 : ])
			retDataSet.append(reducedFeatVec)
	return retDataSet 
