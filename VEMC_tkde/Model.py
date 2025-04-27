
import random
import os
import copy
import math
import numpy as np
import sys
import json
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from test_mvc import cluster_accuracy
from scipy.spatial.distance import cdist

class Model:

    def __init__(self, K, target_K, V, iterNum, alpha, beta, _lambda, dataset, ParametersStr, sampleNo, wordsInTopicNum, behavior, TFN, outputPath):
        self.K = K
        self.target_K = target_K
        self.V = V
        self.iterNum = iterNum
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.behavior = behavior

        self.alpha = alpha
        self.beta = beta
        self._lambda = _lambda

        self.alpha0 = float(K) * float(alpha)
        self.beta0 = float(V) * float(beta)

        # Tne importance of words
        self.w_v = V * [beta]
        self.wv0 = float(V) * float(beta)

        self.smallDouble = 1e-150
        self.largeDouble = 1e150
        
        self.TFN = TFN
        self.outputPath = outputPath

    
    def showTempResult(self, stage, processing, cluster_label):
        cluster_pred = pd.Series([self.z[d] for d in range(0, self.D_All)])
        metrics = cluster_accuracy(cluster_label, cluster_pred)
        if isinstance(processing, str):
            # print(f"\t{stage} stage | Metrics:", metrics)
            logging.info(f"\t{stage} stage | Metrics: {metrics}")
            return metrics
            
        elif isinstance(processing, int):
            # print(f"\t{stage} stage before {processing} iternum | Metrics:", metrics)
            logging.info(f"\t{stage} stage before {processing} iternum | Metrics:{metrics}")
            return metrics

    def earlyStopped(self):
        # Early Stopped
        # if sim > -0.3:
        pass

    def compute_tfidf_vectors(self):
        """
        计算每个 cluster 的 TF-IDF 向量。
        """
        matrix = np.array(self.n_zv)
        # 使用 TfidfVectorizer 提取 TF-IDF
        TFtransformer = TfidfTransformer()
        tfidf_matrix = TFtransformer.fit_transform(matrix)
        return tfidf_matrix
    
    def update(self, cluster1, cluster2, clusterInDoc):
        # 合并 cluster1 和 cluster2
        for d in clusterInDoc[cluster2]:
            self.z[d] = cluster1

        clusterInDoc[cluster1].extend(clusterInDoc[cluster2])
        del clusterInDoc[cluster2]

        if cluster2 in self.non_empty_clusters:
            self.non_empty_clusters.remove(cluster2)
        
        self.m_z[cluster1] += self.m_z[cluster2]
        self.m_z[cluster2] = 0
        self.n_z[cluster1] += self.n_z[cluster2]
        self.n_z[cluster2] = 0
        self.n_zv[cluster1] = np.add(self.n_zv[cluster1], self.n_zv[cluster2])
        self.n_zv[cluster2] = [0] * len(self.n_zv[cluster2])
        self.K_current -= 1

    # def compute_combined_similarity(self, tfidf_matrix, cluster1, cluster2, clusterInDoc):
    #     """
    #     计算两个簇的综合相似度，结合余弦相似度和簇的一致性。
    #     """
    #     # TF-IDF 余弦相似度
    #     cosine_sim = cosine_similarity(tfidf_matrix[cluster1], tfidf_matrix[cluster2])[0, 0]
    #     compactness = None
    #     # 综合相似度
    #     _lambda = 0.7
    #     return _lambda * cosine_sim + (1 - _lambda) * compactness

    def tfIdfClusterMerging(self, documentSet):
        """
        使用增量更新优化的 cluster 合并方法。
        """
        clusterInDoc = defaultdict(list)
        for d in range(0, self.D_All):
            cluster = self.z[d]
            clusterInDoc[cluster].append(d)
            
        # 初始化 TF-IDF 矩阵和相似度矩阵
        tfidf_matrix = self.compute_tfidf_vectors()
        self.non_empty_clusters = [k for k in range(self.K) if tfidf_matrix[k].sum() > 0]

        # 计算初始相似度矩阵
        import heapq
        max_heap = []
        for i in self.non_empty_clusters:
            for j in self.non_empty_clusters:
                if i < j:  # 避免重复计算
                    if self.behavior['tfIdfMerge'] == 2:
                        sim = self.compute_combined_similarity(tfidf_matrix, i, j, clusterInDoc)
                    else:
                        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
                    heapq.heappush(max_heap, (-sim, i, j))  # 负值表示最大堆  
        step = 0
        while self.K_current > self.target_K:
            # 找到当前相似度最大的两个 cluster
            while True:  # 循环直到找到有效的簇对
                sim, cluster1, cluster2 = heapq.heappop(max_heap)
                # 检查两个簇是否仍然有效（未被合并）
                if cluster1 in self.non_empty_clusters and cluster2 in self.non_empty_clusters:
                    break
            
            if -sim < self._lambda:
                break
            if step % 1 == 0:
                # 将效果评估加入self.k_metric_all
                self.save_kMetrics(documentSet, "Post", None)
                
            self.update(cluster1, cluster2, clusterInDoc)
            
            logging.info(f"Merged clusters {cluster1} and {cluster2}, similarity: {-sim:.4f}, K_current: {self.K_current}")
            print(f"Merged clusters {cluster1} and {cluster2}, similarity: {-sim:.4f}, K_current: {self.K_current}")

            tfidf_matrix = self.compute_tfidf_vectors()
            # 更新相似度堆（仅更新涉及 cluster1 的部分）
            for other_cluster in self.non_empty_clusters:
                if other_cluster != cluster1:
                # 跳过空簇的相似度更新
                    if tfidf_matrix[other_cluster].sum() == 0:
                        continue
                    # 重新计算合并后 cluster 的相似度
                    if self.behavior['tfIdfMerge'] == 2:
                        new_sim = self.compute_combined_similarity(tfidf_matrix, cluster1, other_cluster, clusterInDoc)
                    else:
                        new_sim = cosine_similarity(tfidf_matrix[cluster1], tfidf_matrix[other_cluster])[0, 0]
                    
                    heapq.heappush(max_heap, (-new_sim, cluster1, other_cluster))
            step += 1
        # 将效果评估加入self.k_metric_all，将其记录在kMetrics文件中
        self.save_kMetrics(documentSet, "Finally", None)

    def runDMM_plus(self, documentSet, wordList):
        # The whole number of documents
        self.D_All = documentSet.D
        # Cluster assignments of each document               (documentID -> clusterID)
        self.z = [-1] * self.D_All
        # The number of documents in cluster z               (clusterID -> number of documents)
        self.m_z = [0] * self.K
        # The number of words in cluster z                   (clusterID -> number of words)
        self.n_z = [0] * self.K
        # The number of occurrences of word v in cluster z   (n_zv[clusterID][wordID] = number)
        self.n_zv = [[0] * self.V for _ in range(self.K)]
        
        self.K_current = copy.deepcopy(self.K)

        if self.behavior["initType"] == 0:
            self.intialize(documentSet)
            print("\tintialize successful! Start to save entropy.")
        else:
            self.intialize_k(documentSet)
            print("\tintialize k_classes successful! Start to save entropy.")

        # 保存所有单词json文件
        # if self.behavior["freqEntropy"] != -1:
        #     self.save_entropy(None, wordList)
        #     print("\tsave entropy successful! Start to Gibbs sampling.")
        # else:
        #     print("\tNo save entropy! Start to Gibbs sampling.")

        self.gibbsSampling(documentSet)
        print("\tGibbs sampling successful! Start to merging clusters.")

        if self.behavior["tfIdfMerge"] != 0:
            self.tfIdfClusterMerging(documentSet)
            print("\tmerges clusters successful! Start to saving results.")
        else:
            print("\tNo merges cluster! Start to saving results.")

        # 最终输出所有结果保存在result文件夹中-路径为outputPath
        os.makedirs(self.outputPath['Result'], exist_ok=True)  # 确保目录存在
        self.output(documentSet, self.outputPath['Result'], wordList)
        print("\tSaving successful!")

    
    def intialize(self, documentSet):
        print("\tInitialization.")

        for d in range(0, self.D_All):
            document = documentSet.documents[d]
            cluster = int(self.K * random.random())
            self.z[d] = cluster
            self.m_z[cluster] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

        for k_t in range(0, self.K):
            self.checkEmpty(k_t)
    
    def intialize_k(self, documentSet):
        print("\tInitialization.")

        random_indices = np.random.choice(self.D_All, self.K, replace=False)
        for k_t in range(0, self.K): # 这是随机初始化k个中心
            document = documentSet.documents[random_indices[k_t]]
            self.z[random_indices[k_t]] = k_t
            self.m_z[k_t] += 1
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[k_t][wordNo] += wordFre
                self.n_z[k_t] += wordFre
        for d in tqdm(range(0, self.D_All)):
            if d not in random_indices:
                document = documentSet.documents[d]
                cluster = self.simplesampleCluster(-1, document, "iter")
                self.z[d] = cluster
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre
        
        for k_t in range(0, self.K):
            self.checkEmpty(k_t)

    def entropy(self):  # 现在熵值范围(0, 1]，可以再研究一下--》(0, 0.2]??
        # n_zv = np.array(self.n_zv) + self.beta
        n_zv = np.array(self.n_zv)
        # 计算 n_zv 的按列归一化概率
        column_sums = n_zv.sum(axis=0) + 1e-10  # 避免分母为0
        v_probs = (n_zv / column_sums).clip(1e-10, 1.0)  # 合并归一化和clip操作

        # 使用矢量化计算每一列的熵
        entropy = -np.sum(v_probs * np.log2(v_probs), axis=0)

        max_entropy = np.max(entropy)
        scaled_entropy = entropy / max_entropy

        return scaled_entropy
    
    def save_entropy(self, all_w_v, wordList):
        os.makedirs(self.outputPath['Entropy'], exist_ok=True)  # 确保目录存在
        if wordList != None:
            if not os.path.exists(os.path.join(self.outputPath['Entropy'], "wordlist.json")):
                with open(os.path.join(self.outputPath['Entropy'], "wordlist.json"), mode="w", encoding="utf-8") as file:
                    json.dump(wordList, file, ensure_ascii=False, indent=4)
        else:
            # 将 all_w_v 保存为 CSV 文件
            df = pd.DataFrame(all_w_v).transpose()  # 转置使每列对应一次迭代
            df.to_csv(f"{self.outputPath['Entropy']}w_v_iterations_{self.sampleNo}.csv", index=False, header=[f"Iter_{i+1}" for i in range(self.iterNum)])

    def save_kMetrics(self, documentSet, type, i):
        if type == "last":
            if self.behavior["tfIdfMerge"] == 0:
                record_temp = self.showTempResult("gibbsSampling", "complete", documentSet.labels)
                record_temp["Kcurrent"] = self.K_current
                self.k_metric_all.append(record_temp)
                outpath = self.outputPath["kMetrics"]
                os.makedirs(outpath, exist_ok=True)  # 确保目录存在
                df_k_metric_all = pd.DataFrame(self.k_metric_all)
                df_k_metric_all.to_csv(f"{outpath}kMetrics_{self.sampleNo}.csv", index=False, encoding='utf-8')
            else:
                pass
        elif type == "iter":
            record_temp = self.showTempResult("gibbsSampling", i + 1, documentSet.labels)
            record_temp["Kcurrent"] = self.K_current
            self.k_metric_all.append(record_temp)
        elif type == "Post":
            record_temp = self.showTempResult("tfIdfClusterMerging", "Post", documentSet.labels)
            record_temp["Kcurrent"] = self.K_current
            self.k_metric_all.append(record_temp)
        elif type == "Finally":
            record_temp = self.showTempResult("tfIdfClusterMerging", "Finally", documentSet.labels)
            record_temp["Kcurrent"] = self.K_current
            self.k_metric_all.append(record_temp)
            outpath = self.outputPath["kMetrics"]
            os.makedirs(outpath, exist_ok=True)  # 确保目录存在
            df_k_metric_all = pd.DataFrame(self.k_metric_all)
            df_k_metric_all.to_csv(f"{outpath}kMetrics_{self.sampleNo}.csv", index=False, encoding='utf-8')

    def gibbsSampling(self, documentSet):
        intervalEntropy = int(self.D_All / self.behavior["freqEntropy"])
        all_w_v = []
        self.k_metric_all = []
        for i in range(self.iterNum):
            # 将效果评估加入self.k_metric_all
            self.save_kMetrics(documentSet, "iter", i)
            
            print("\titer is ", i + 1, end="\t")
            print("Kcurrent is" + " %f." % self.K_current, end='\n')
            print(f"\tbeta is (mean:{np.mean(self.w_v)}, std:{np.std(self.w_v)}, max:{np.max(self.w_v)}, min:{np.min(self.w_v)})")
            for d in tqdm(range(0, self.D_All)):
                document = documentSet.documents[d]
                cluster = self.z[d]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)

                if d % intervalEntropy == 0 and self.behavior["freqEntropy"] > 0:
                    self.w_v = self.entropy()
                    self.wv0 = np.sum(self.w_v)

                if i != self.iterNum - 1:
                    cluster = self.simplesampleCluster(i, document, "iter")
                else:
                    cluster = self.simplesampleCluster(i, document, "last")

                self.z[d] = cluster
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

            # 保存每次迭代后的 self.w_v
            all_w_v.append(self.w_v.copy())
                    
        # 将效果评估加入self.k_metric_all，如果无cluster merging则将其记录在kMetrics文件中，否则跳过
        self.save_kMetrics(documentSet, "last", self.iterNum)
        print("\tKcurrent is" + " %f." % self.K_current, end='\n')
        # 保存每次迭代所有单词的Entropy
        if self.behavior["freqEntropy"] != -1:
            self.save_entropy(all_w_v, None)

    def sumNormalization(self, x):
        """Normalize the prob."""
        x = np.array(x)
        norm_x = x / np.sum(x)
        return norm_x

    '''
    MODE
    "iter"  Iteration after initialization.
    "last"  Last iteration.
    '''
    def simplesampleCluster(self, _iter, document, MODE):
        prob = [float(0.0)] * (self.K)
        overflowCount = [float(0.0)] * (self.K)

        for k in range(self.K):
            if self.m_z[k] == 0:
                prob[k] = 0
                continue
            valueOfRule1 = (self.m_z[k] + self.alpha) # / (self.D_All - 1 + self.alpha0)
            valueOfRule2 = 1.0
            i = 0
            for _, w in enumerate(range(document.wordNum)):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if valueOfRule2 < self.smallDouble:
                        overflowCount[k] -= 1
                        valueOfRule2 *= self.largeDouble
                  
                    valueOfRule2 *= (self.n_zv[k][wordNo] + self.w_v[wordNo] + j) / (self.n_z[k] + self.wv0 + i)
                    i += 1
            prob[k] = valueOfRule1 * valueOfRule2

        max_overflow = -sys.maxsize
        for k in range(self.K):
            if overflowCount[k] > max_overflow and prob[k] > 0.0:
                max_overflow = overflowCount[k]
        for k in range(self.K):
            if prob[k] > 0.0:
                prob[k] = prob[k] * math.pow(self.largeDouble, overflowCount[k] - max_overflow)
        prob = self.sumNormalization(prob)  # dmm

        if MODE == "iter":
            kChoosed = 0
            for k in range(1, self.K):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K - 1]
            while kChoosed < self.K:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            return kChoosed

        elif MODE == "last":
            kChoosed = 0
            bigPro = prob[0]
            for k in range(1, self.K):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            return kChoosed
    

    # update K_current
    def checkEmpty(self, cluster):
        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output(self, documentSet, outputPath, wordList):
        outputDir = outputPath
        try:
            # create result/
            isExists = os.path.exists(outputPath)
            if not isExists:
                os.mkdir(outputPath)
                print("\tCreate directory:", outputPath)
            # create after result
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):    # φ
        self.phi_zv = [[0] * self.V for _ in range(self.K)]
        for cluster in range(self.K):
            for v in range(self.V):
                self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.w_v[v]) / float(
                    self.n_z[cluster] + self.wv0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in range(len(array)):
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if self.m_z[k] == 0:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key=lambda tc: tc[1], reverse=True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(0, self.D_All):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[d]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()
