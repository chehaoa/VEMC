from DocumentSet import DocumentSet
from Model import Model


class DMM_plus:

    def __init__(self, K, target_K, alpha, beta, _lambda, iterNum, sampleNum, behavior, dataset, wordsInTopicNum, dataDir, TFN):
        self.K = K
        self.target_K = target_K
        self.alpha = alpha
        self.beta = beta
        self._lambda = _lambda
        self.iterNum = iterNum
        self.sampleNum = sampleNum
        self.behavior = behavior
        self.dataset = dataset
        self.wordsInTopicNum = wordsInTopicNum
        self.dataDir = dataDir

        self.wordList = []
        self.wordToIdMap = {}

        self.TFN = TFN

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
        self.V = self.wordToIdMap.__len__()

    def runDMM_plus(self, sampleNo, outputPath):
       
        ParametersStr = "K" + str(self.K) + "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3)) + "_lambda" + str(round(self._lambda, 3)) + "-|-"
        model = Model(self.K, self.target_K, self.V, self.iterNum, self.alpha, self.beta, self._lambda,
                      self.dataset, ParametersStr, sampleNo, self.wordsInTopicNum, self.behavior, self.TFN, outputPath)
        model.runDMM_plus(self.documentSet, self.wordList)
