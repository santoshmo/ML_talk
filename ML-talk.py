import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
import pandas as pd
import csv

covariances = ['spherical', 'tied', 'diag', 'full']

def normalGMMmodelSelect(data, max_components):
	bicVals = [[None for x in range(0, len(covariances))] for y in range(0, max_components)]
	bestCov = ''
	minBIC = sys.float_info.max
	bestNumComponents = 0
	for num_components in range(1, max_components+1):
		for cov in range(0, len(covariances)):
			cov_type = covariances[cov]
			gmm = mixture.GMM(n_components = num_components, n_iter=5000, covariance_type = cov_type, min_covar=0.01)
			gmm.fit(data)
			gmmBIC = gmm.bic(data)
			bicVals[num_components-1][cov] = gmmBIC
			print (num_components, cov_type, gmmBIC)
			if gmmBIC < minBIC:
				minBIC = gmmBIC
				bestNumComponents = num_components
				bestCov = cov_type
	print [bestCov, bestNumComponents]
	return [bestCov, bestNumComponents, bicVals]

def normalGMMpredict(data, bestNumComponents, bestCovType):
	gmm = mixture.GMM(n_components = bestNumComponents, n_iter=5000, covariance_type = bestCovType, min_covar = 0.01)
	gmm.fit(data)
	return gmm.predict(data)

def plotGMM(data, classes, bestCov, bestNumComponents):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], c=classes, marker='o')
	title = "NBA Player into " + str(bestNumComponents) + " Clusters of " + str(bestCov) + " Covariances"
	ax.set_title(title)
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')
	plt.show()

def plot3d (data, title_init, x_axis, y_axis, z_axis):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(data[:,0], data[:,1], data[:,2], marker='o')
	ax.set_title(title_init)
	ax.set_xlabel(x_axis)
	ax.set_ylabel(y_axis)
	ax.set_zlabel(z_axis)
	plt.show()

def pcaData (data):
	pca = PCA(n_components = 6) # I've precomputed this number
	shotData = data.ix[:,1:14].values # data without playerID or names
									  # we can recover this later
	return pca.fit_transform(shotData)

def format(string):
	return "'{}'".format(string)

def genPlayerTeamData():
	playerLookup = {} # map player to team
	with open('playerToTeam.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			player = row[0] + ',' + row[1]
			player = player.strip('"')
			team = row[len(row)-1]
			if player not in playerLookup.keys():
				playerLookup[player] = team
	return playerLookup

class Team: 
	def __init__(self, name_init):
		self.name = name_init
		self.clusterCount = {}

	def push(self, cluster):
		if cluster not in self.clusterCount.keys():
			self.clusterCount[cluster] = 1
		else: 
			self.clusterCount[cluster] += 1

def genClusterCountPerTeam(data, playerToTeam, classes, bestNumComponents):
	teams = {}
	count = 0
	for player in data.ix[:,0]:
		if player == "Nene":
			count = count+1
			continue
		else:
			team = playerToTeam[player]
			if team not in teams.keys():
				teams[team] = Team(team)
			cluster = classes[count]
			teams[team].push(cluster)
			count = count+1
	return teams

def getLogReg(teamClusters, rankingDict, bestNumComponents, data, classes):
	# Construct data matrix
	# each column will represent a particular cluster
	# each row will be a team 
	# at the end of each row will be the W/L % 
	# we will regress to find a fit for getting  50%+ team
	dataMatrix = np.zeros((30, bestNumComponents))
	orderedWinPerc = []
	teamNum = 0
	for team in teamClusters.keys():
		team = team.strip('"')
		if team == "PHI":
			continue
		else: 
			clusterNum = 0
			for cluster in teamClusters[team].clusterCount.keys():
				dataMatrix[teamNum, clusterNum] = teamClusters[team].clusterCount[cluster]
				clusterNum = clusterNum + 1
			if (rankingDict[team] >= 0.5):
				orderedWinPerc.append(1)
			else: 
				orderedWinPerc.append(0)
			teamNum = teamNum + 1
	logReg = LogisticRegression()
	pred = logReg.fit(dataMatrix, orderedWinPerc)
	print "Weights: " + str(pred.coef_)
	return pred

def main(): 
## Load
	data = pd.DataFrame.from_csv('playerstandardized.csv', sep=',')

## PCA	
	pcaedData = pcaData(data)
	plot3d(pcaedData, "PCAed Data", "PC1", "PC2", "PC3")

## GMM Model Selection
	[bestCovType, bestNumComponents, bicVals] = normalGMMmodelSelect(pcaedData, 10)
	classes = normalGMMpredict(pcaedData, bestNumComponents, bestCovType)
	plotGMM(pcaedData, classes, bestCovType, bestNumComponents)

## Associate players with clusters
	playerToTeam = genPlayerTeamData()
	teamClusters = genClusterCountPerTeam(data, playerToTeam, classes, bestNumComponents)
	
## Logistic Regression of team compositions against win %
	rankings = pd.DataFrame.from_csv('rankings.csv', sep=',')
	rankingDict = {}
	for i in range(1, rankings.shape[0]):
		rankingDict[rankings.ix[i,0]] = rankings.ix[i,1]
	pred = getLogReg(teamClusters, rankingDict, bestNumComponents, data, classes)
	
	# Get Recommendation for valuable players
	weights = pred.coef_.tolist()
	m = max(weights)
	bestCluster = [i for i, j in enumerate(weights) if j == m]
	print "Best Cluster: " + str(bestCluster)

	for i in range(0, len(classes)):
		if classes[i] == bestCluster:
			print data.ix[i, 0]

main()