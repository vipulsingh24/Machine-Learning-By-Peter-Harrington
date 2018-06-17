import numpy as np

def loadDataSet():
	dataMat = []; label = []
	fr = open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		label.append(int(lineArr[2]))
	return dataMat, label

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def gradientAscent(dataMatIn, classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	m, n =  np.shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

def stocGradAscent(dataMatrix, classLabels):
	m, n = np.shape(dataMatrix)
	alpha = 0.01
	weights = np.ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i] * weights))
		error = classLabels[i] - h
		weights = weights  + alpha * error * dataMatrix[i]
	return weights

def plotBestFit(wei):
	import matplotlib.pyplot as plt
	#weights = wei.getA()
	weights = wei
	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)
	n = np.shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i][1])
			ycord1.append(dataArr[i][2])
		else:
			xcord2.append(dataArr[i][1])
			ycord2.append(dataArr[i][2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c ='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c = 'green')
	x = np.arange(-3.0, 3.0, 1.0)
	y = (-weights[0] - weights[1] * x) / weights[2]
	ax.plot(x, y)
	plt.xlabel('X1')
	plt.ylabel('Y1')
	plt.show()
	
