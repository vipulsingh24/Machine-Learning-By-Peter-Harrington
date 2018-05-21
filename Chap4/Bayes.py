'''
Naive Bayes Classifier
'''


def loadDataSet():
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
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)	# Creates the union of two set
	return list(vocabSet)
	
