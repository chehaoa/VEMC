# -*- coding: utf-8 -*-
import codecs
import json
from sklearn import metrics
import numpy as np
import math


def cluster_acc(Y, Y_pred):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[ind[0][i], ind[1][i]] for i in range(D)]) * 1.0 / Y_pred.size


def cluster_purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def MeanAndVar(dataList):
    mean = sum(dataList)*1.0 / len(dataList)
    varience = math.sqrt(sum((mean - value) ** 2 for value in dataList)*1.0 / len(dataList))
    return (mean, varience)

class ClusterEvaluation():
    def __init__(self, resultFilePath):
        self.tweetsCleaned = []
        self.resultFilePath = resultFilePath

        self.ACCList = []
        self.purityList = []
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []
        self.SCList = []

        self.ACCTopicKList = []
        self.purityTopicKList = []
        self.AMITopicKList = []
        self.NMITopicKList = []
        self.MITopicKList = []
        self.ARITopicKList = []
        self.homogeneityTopicKList = []
        self.completenessTopicKList = []
        self.VTopicKList = []
        self.SCTopicKList = []
        self.docNum = 0
        self.KRealNum = -1

        self.labelsPred = {}
        self.labelsTrue = {}

    # Get labelsPred.
    def getPredLabels(self, inFile):
        self.labelsPred = {}
        with codecs.open(inFile, 'r') as fin:
            for line in fin:
                try:
                    documentID = line.strip().split()[0]
                    clusterNo = line.strip().split()[1]
                    self.labelsPred[int(documentID)] = int(clusterNo)
                except:
                    print(line)

    # Get labelsTrue and docs.
    def getTrueLabels(self, dataset):
        self.labelsTrue = {}
        self.docs = []
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue[int(docObj['tweetId'])] = int(docObj['clusterNo'])
                    self.docs.append([int(docObj['tweetId']), docObj['textCleaned']])
                except:
                    print(docJson)
        self.KRealNum = len(set(self.labelsTrue.values()))

    def getKPredNum(self, inFile):
        labelsPred = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
        KPredNum = np.unique(labelsPred).shape[0]
        return KPredNum

    # Get scaled cluster number and scaled document number. Kthreshold is minimum number of a scaled cluster.
    def getPredNumThreshold(self, inFile, Kthreshold):
        KPredNum = 0
        docRemainNum = 0
        docTotalNum = 0
        with codecs.open(inFile, 'r', 'utf-8') as fin:
            clusterSizeStr = fin.readline().strip().strip(',')
        clusterSizeList = clusterSizeStr.split(',\t')
        for clusterSize in clusterSizeList:
            try:
                clusterSizeCouple = clusterSize.split(':')
                docTotalNum += int(clusterSizeCouple[1])
                if int(clusterSizeCouple[1]) > Kthreshold:
                    KPredNum += 1
                    docRemainNum += int(clusterSizeCouple[1])
            except:
                pass
        return (KPredNum, docRemainNum, docTotalNum)

    # Get evaluation of each sample
    def evaluatePerSample(self, sampleNo):
        labelsTrue = []
        labelsPred = []
        for d in self.docs:
            documentID = d[0]
            if documentID in self.labelsPred:
                labelsTrue.append(self.labelsTrue[documentID])
                labelsPred.append(self.labelsPred[documentID])
        ACC = cluster_acc(labelsTrue, labelsPred)
        purity = cluster_purity(labelsTrue, labelsPred)
        AMI = metrics.adjusted_mutual_info_score(labelsTrue, labelsPred)
        NMI = metrics.normalized_mutual_info_score(labelsTrue, labelsPred)
        MI = metrics.mutual_info_score(labelsTrue, labelsPred)
        ARI = metrics.adjusted_rand_score(labelsTrue, labelsPred)
        homogeneity = metrics.homogeneity_score(labelsTrue, labelsPred)
        completeness = metrics.completeness_score(labelsTrue, labelsPred)
        V = metrics.v_measure_score(labelsTrue, labelsPred)

        self.ACCList.append(ACC)
        self.purityList.append(purity)
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)

    # Get mean and var of all evaluation
    def evaluateAllSamples(self):
        self.ACCTopicKList.append(MeanAndVar(self.ACCList))
        self.purityTopicKList.append(MeanAndVar(self.purityList))
        self.ARITopicKList.append(MeanAndVar(self.ARIList))
        self.MITopicKList.append(MeanAndVar(self.MIList))
        self.AMITopicKList.append(MeanAndVar(self.AMIList))
        self.NMITopicKList.append(MeanAndVar(self.NMIList))
        self.homogeneityTopicKList.append(MeanAndVar(self.homogeneityList))
        self.completenessTopicKList.append(MeanAndVar(self.completenessList))
        self.VTopicKList.append(MeanAndVar(self.VList))

        self.ACCList = []
        self.purityList = []
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []

    def drawEvaluationResult(self, titleStr):
        ACCMeanList = [item[0] for item in self.ACCTopicKList]
        purityMeanList = [item[0] for item in self.purityTopicKList]
        ARIMeanList = [item[0] for item in self.ARITopicKList]
        AMIMeanList = [item[0] for item in self.AMITopicKList]
        NMIMeanList = [item[0] for item in self.NMITopicKList]
        homogeneityMeanList = [item[0] for item in self.homogeneityTopicKList]
        completenessMeanList = [item[0] for item in self.completenessTopicKList]
        VMeanList = [item[0] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('# %s\n\n' % (titleStr))
            fout.write('ACCMean:\t\t\t\t' + repr(ACCMeanList) + '\n')
            fout.write('purityMean:\t\t\t\t' + repr(purityMeanList) + '\n')
            fout.write('ARIMean:\t\t\t\t' + repr(ARIMeanList) + '\n')
            fout.write('AMIMean:\t\t\t\t' + repr(AMIMeanList) + '\n')
            fout.write('NMIMean:\t\t\t\t' + repr(NMIMeanList) + '\n')
            fout.write('homogeneityMean:\t\t' + repr(homogeneityMeanList) + '\n')
            fout.write('completenessMean:\t\t' + repr(completenessMeanList) + '\n')
            fout.write('VMean:\t\t\t\t\t' + repr(VMeanList) + '\n')
            fout.write('\n')

        ACCVarianceList = [item[1] for item in self.ACCTopicKList]
        purityVarianceList = [item[1] for item in self.purityTopicKList]
        ARIVarianceList = [item[1] for item in self.ARITopicKList]
        AMIVarianceList = [item[1] for item in self.AMITopicKList]
        NMIVarianceList = [item[1] for item in self.NMITopicKList]
        homogeneityVarianceList = [item[1] for item in self.homogeneityTopicKList]
        completenessVarianceList = [item[1] for item in self.completenessTopicKList]
        VVarianceList = [item[1] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('ACCVariance:\t\t\t' + repr(ACCVarianceList) + '\n')
            fout.write('purityVariance:\t\t\t' + repr(purityVarianceList) + '\n')
            fout.write('ARIVariance:\t\t\t' + repr(ARIVarianceList) + '\n')
            fout.write('AMIVariance:\t\t\t' + repr(AMIVarianceList) + '\n')
            fout.write('NMIVariance:\t\t\t' + repr(NMIVarianceList) + '\n')
            fout.write('homogeneityVariance:\t' + repr(homogeneityVarianceList) + '\n')
            fout.write('completenessVariance:\t' + repr(completenessVarianceList) + '\n')
            fout.write('VVariance:\t\t\t\t' + repr(VVarianceList) + '\n')
            fout.write('\n')

    def drawPredK(self, KRealNumMeanList, KPredNumMeanList, KRealNumVarianceList, KPredNumVarianceList):
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRealNumMean:\t\t\t' + repr(KRealNumMeanList) + '\n')
            fout.write('KPredNumMean:\t\t\t' + repr(KPredNumMeanList) + '\n')
            fout.write('\n')
            fout.write('KRealNumVariance:\t\t' + repr(KRealNumVarianceList) + '\n')
            fout.write('KPredNumVariance:\t\t' + repr(KPredNumVarianceList) + '\n')
            fout.write('\n' + '-' * 50 + '\n\n')


def CMM_withGMMEvaluation():
    dirName = '%sK%dalpha%sbeta%ssigma%ssigma0%s_lambda%siterNum%dSampleNum%d/' % \
              (dataset, K, alpha, beta, sigma, sigma0, _lambda, iterNum, sampleNum)
    resultFileName = 'CMM_withGMMDataset%sK%dalpha%sbeta%ssigma%ssigma0%s_lambda%siterNum%dSampleNum%dNoiseKThreshold%d.txt' % (
        dataset, K, alpha, beta, sigma, sigma0, _lambda, iterNum, sampleNum, KThreshold)
    resultFilePath = inPath + dirName + resultFileName

    CMM_withGMMEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    KRealNumMeanList = []
    KRealNumVarianceList = []
    noiseNumMeanList = []
    noiseNumVarianceList = []
    KPredNumList = []
    KRealNumList = []
    noiseNumList = []

    for sampleNo in range(1, sampleNum + 1):
        inDir = inPath + dirName
        fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
        inFile = inDir + fileName
        CMM_withGMMEvaluation.getPredLabels(inFile)
        sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
        (KPredNum, docRemainNum, docTotalNum) = CMM_withGMMEvaluation.getPredNumThreshold(sizeFile, KThreshold)
        CMM_withGMMEvaluation.getTrueLabels(datasetPath)

        CMM_withGMMEvaluation.evaluatePerSample(sampleNo)
        KPredNumList.append(KPredNum)
        KRealNumList.append(CMM_withGMMEvaluation.KRealNum)
        noiseNumList.append(docTotalNum - docRemainNum)

    CMM_withGMMEvaluation.evaluateAllSamples()
    KPredNumMeanList.append(np.mean(KPredNumList))      # mean
    KRealNumMeanList.append(np.mean(KRealNumList))
    noiseNumMeanList.append(np.mean(noiseNumList))
    KPredNumVarianceList.append(np.std(KPredNumList))   # standard deviation
    KRealNumVarianceList.append(np.std(KRealNumList))
    noiseNumVarianceList.append(np.std(noiseNumList))

    titleStr = 'CMM_withGMM %s K%d iterNum%d SampleNum%d alpha%s beta%s sigma%s sigma0%s _lambda%s' % \
               (dataset, K, iterNum, sampleNum, alpha, beta, sigma, sigma0, _lambda)
    CMM_withGMMEvaluation.drawEvaluationResult(titleStr)
    CMM_withGMMEvaluation.drawPredK(KRealNumMeanList, KPredNumMeanList, KRealNumVarianceList, KPredNumVarianceList)


if __name__ == '__main__':
    KThreshold = 0
    sampleNum = 5
    iterNum = 10
    K = 300
    alpha = '0.1'
    beta = '0.1'
    sigma = '0.03'
    sigma0 = '0.1'
    _lambda = '0.8'

    # dataset = "News-SIF"
    dataset = "Tweet-SIF"
    # dataset = "News-w2v"
    # dataset = "Tweet-w2v"
    # dataset = "R52-SIF"
    datasetPath = './data/' + dataset
    inPath = './result/'
    lambda_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sigma_list = [0.01,0.02,0.04,0.05,0.06,0.07,0.08]
    for s in sigma_list:
        sigma = str(s)
        CMM_withGMMEvaluation()
