'''
Naive Bayes Classifier
'''
from numpy import *

def loadDataSet():
	'''
	Example data to experiment with.
	'''
	postingList = [['my', 'dog', 'has', 'flea',\
					'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him',\
					'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute',\
					'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how',\
					'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList, classVec

def createVocabList(dataSet):
	'''
	Creates a list of all the unique words in all of our documents.
	'''
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)	# Creates the union of two set
	return list(vocabSet)
	
def setOfWords2Vec(vocabList, inputSet):
	'''
	Outputs a vector of 0s and 1s to represent whether a word from vocabulary list
	is present or not in the given document.
	'''
	returnVec = [0] * len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print('The word: %s is not in my Vocabulary!' % word)
	return returnVec

def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory) / float(numTrainDocs)
	p0Num = zeros(numWords)
	p1Num = zeros(numWords)
	p0Denom = 0.0
	p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num / p1Denom
	p0Vect = p0Num / p0Denom
	return p0Vect, p1Vect, pAbusive
