import os
import logging
from DMM_plus import DMM_plus
import time
import argparse

KThreshold = 0
wordsInTopicNum = 20

parser = argparse.ArgumentParser()
parser.add_argument("--dataDir", type=str, default="data/DMM_datasets/")
parser.add_argument("--dataset", type=str, default="Tweet")
parser.add_argument("--task", type=str, default="otherData")   # 分文件夹
parser.add_argument("--K", type=int, default=500)
parser.add_argument("--target_K", type=int, default=89)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--freqEntropy", type=int, default=15) # -1 不使用Entropy  15(>0) 使用Entropy
parser.add_argument("--initType", type=int, default=1) # 0 init, 1 init_k
parser.add_argument("--tfIdfMerge", type=int, default=1) # 0 not merge, 1 merge, 2 combined_merge

args = parser.parse_args()

print(f"Using dataset: {args.dataset}, K={args.K}, target_K={args.target_K}, beta={args.beta}")

sampleNum = 10  # default 10
iterNum = 20  # default 20
behavior = {"freqEntropy" : args.freqEntropy,
            "initType" : args.initType,
            "tfIdfMerge" : args.tfIdfMerge}
task = args.task
dataDir = args.dataDir
dataset = args.dataset
alpha = args.alpha
beta = args.beta
K = args.K
target_K = args.target_K
freqEntropy = args.freqEntropy
initType = args.initType
tfIdfMerge = args.tfIdfMerge

# freqEntropy = 15  # -1 不使用Entropy  15(>0) 使用Entropy
# initType = 1  # 0 init, 1 init_k
# tfIdfMerge = 1  # 0 not merge, 1 merge, 2 combined_merge
# # ------------------hyper-parameters--------------------
# alpha = 0.1
# if initType == 0:
#     beta = 0.1
# elif initType == 1:
#     beta = 0.01

# cluster merging停止相似度域值
_lambda = 0.0 # --0.0表示不使用


def runDMM_plus(K, target_K, alpha, beta, _lambda, iterNum, sampleNum, behavior, dataset, wordsInTopicNum, dataDir, TFN):
    mvc = DMM_plus(K, target_K, alpha, beta, _lambda, iterNum, sampleNum, behavior, dataset, wordsInTopicNum, dataDir, TFN)
    mvc.getDocuments()
    for sampleNo in range(1, sampleNum + 1):
        print("SampleNo:" + str(sampleNo))
        mvc.runDMM_plus(sampleNo, outputPath)

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    TFN = f"iterNum({iterNum})+({K}|{alpha}|{beta})+initK({initType})+entropy({freqEntropy})+tfIcf({tfIdfMerge})"

    outputResult = f"result/{task}/{dataset}/{TFN}/"
    outputkMetrics = f"./kMetrics/{task}/{dataset}/{TFN}/"
    outputEntropy = f"./entropy/{task}/{dataset}/{TFN}/"
    outputLogging = f"./log/{task}/{dataset}/"
    outputPath = {"Result": outputResult,
                "Logging": outputLogging,
                "kMetrics": outputkMetrics,
                "Entropy": outputEntropy}
    # -------------------------------------------------------------------------
    outpath = outputPath["Logging"]
    os.makedirs(outpath, exist_ok=True)

    logging.basicConfig(filename=f'{outpath}/merging_log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', encoding='utf-8')
    logging.info(f"runDMM_plus begin | parameters: {TFN}")

    outf = open("time_MVC", "a")
    time1 = time.time()
    runDMM_plus(K, target_K, alpha, beta, _lambda, iterNum, sampleNum, behavior, dataset, wordsInTopicNum, dataDir, TFN)
    time2 = time.time()
    outf.write(str(dataset) + "K" + str(K) + "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) 
               + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) 
               + "freqEntropy" + str(freqEntropy) + "initType" + str(initType) + "tfIdfMerge" + str(tfIdfMerge)
               + "\ttime:" + str(time2 - time1) + "\n")
