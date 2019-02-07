from numpy import *

# Uses 2 list comprehensions to create the matrix
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

# PCA function takes 2 arguments: the dataset to perform PCA on and topNfeat (top N features to use)
def pca(dataMat, topNfeat=9999999):
    # calculate the mean of the original dataset and remove it
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    # compute the covariance matrix and calculate the eigenvalues
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    # use argsort() to get the order of the eigenvalues
    eigValInd = argsort(eigVals)
    # use the order of the eigenvalues to sort the eigenvectors in reverse order and get the topNfeat largest eigenvectors
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    # reconstruct the original data and return it for debug along with the reduced dimension dataset
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
