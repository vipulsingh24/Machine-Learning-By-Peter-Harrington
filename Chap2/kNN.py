from numpy import *
import operator
from os import listdir

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances =  sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

# Make sure to pass 'datingTestSet2.txt' file 
def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataset):
	minVals = dataset.min(0)
	maxVals = dataset.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataset))
	m = dataset.shape[0]
	normDataSet = dataset - tile(minVals, (m,1))
	normDataSet = normDataSet / tile(ranges, (m,1))
	return normDataSet, ranges, minVals, maxVals


def datingClassTest():
	testRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals, maxVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m * testRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify(normMat[i, :], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print('The classifier came back with: %d, the real answer is: %d' % (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print('The total error rate is: %f' % (errorCount / float(numTestVecs)))

def classifyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(input('Percentage of time spent playing video games? '))
	ffMiles = float(input('Frequent flier miles earned per year? '))
	iceCream = float(input('Liters of ice cream consumed per year? '))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals, maxVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	inArrNorm = (inArr - minVals)/ranges
	classifierResult = classify(inArrNorm, normMat, datingLabels, 3)
	print("You'll probably like this person: ", resultList[classifierResult-1])

def img2vector(filename):
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('digits/trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
	testFileList = listdir('digits/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
		classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
		print('The classifier came back with: %d, the real answer is: %d' % (classifierResult, classNumStr))
		if (classifierResult != classNumStr):
			errorCount += 1.0
	print('\nThe total number of error is: %d' % errorCount)
	print('\nThe total error rate is: %f' % (errorCount/float(mTest)))
