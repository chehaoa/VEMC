import pandas as pd
from utils import metric
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, homogeneity_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics.cluster import contingency_matrix as cm
from scipy.optimize import linear_sum_assignment

def externalValidation(truthClusters, predictedClusters):
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix_mine = cm(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix_mine, axis=0)) / np.sum(contingency_matrix_mine) 
    scores = {}
    scores['_adjusted_rand_index'] = adjusted_rand_score(truthClusters, predictedClusters)
    scores['_homogeneity_score'] = homogeneity_score(truthClusters, predictedClusters)
    scores['_purity_score'] = purity_score(truthClusters, predictedClusters)
    scores['_adjusted_mutual_info_score'] = adjusted_mutual_info_score(truthClusters, predictedClusters)
    scores['_fowlkes_mallows_score'] = fowlkes_mallows_score(truthClusters, predictedClusters)  
    return scores

def cluster_accuracy_2(y_true, y_pred):
    """
    Calculates and returns the accuracy for two lists of labels.
    :param y_true: y_true
    :param y_pred: y_pred
    :return accuracy: accuracy
    """
    
    # compute confusion matrix
    contingency_matrix = cm(y_true, y_pred)
    # Find best mapping between cluster labels and gold labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    #return result
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)

def cluster_accuracy_3(y_true, y_predicted, cluster_number):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy with Munkres (Hungarian) algorithm.
    Args:
        y_true: Array of ground truth labels.
        y_pred: Array of predicted labels.
    Returns:
        Accuracy as a float.
    """
    # 获取唯一类别
    true_labels = list(set(y_true))
    pred_labels = list(set(y_pred))

    num_true_classes = len(true_labels)
    num_pred_classes = len(pred_labels)

    # 创建并填充成本矩阵
    cost_matrix = np.zeros((num_true_classes, num_pred_classes), dtype=int)
    for i, true_label in enumerate(true_labels):
        true_indices = [index for index, label in enumerate(y_true) if label == true_label]
        for j, pred_label in enumerate(pred_labels):
            pred_indices = [index for index in true_indices if y_pred[index] == pred_label]
            cost_matrix[i][j] = -len(pred_indices)  # 使用负数构造“成本”

    # 使用匈牙利算法
    m = Munkres()
    indexes = m.compute(cost_matrix.tolist())
    total_correct = sum(-cost_matrix[i][j] for i, j in indexes)

    # 计算准确率
    accuracy = total_correct / len(y_true)

    # 计算NMI值
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    # 计算 ARI
    ari_score = adjusted_rand_score(y_true, y_pred)

    # 使用最佳匹配的索引重新标记预测类别
    new_predict = np.zeros(len(y_pred))
    for i, (true_index, pred_index) in enumerate(indexes):
        new_predict[np.where(y_pred == pred_labels[pred_index])] = true_labels[true_index]
    
    # 计算 F1-score
    f1 = f1_score(y_true, new_predict, average='macro')

    return {"ACC": accuracy, "NMI": nmi_score, "ARI": ari_score, "F1-score": f1}

def cluster_purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    purity_per_cluster = []  # 用于记录每个聚类的纯度
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        if len(idx) == 0:  # 如果该聚类没有样本
            continue
        labels_tmp = labels_true[idx, :].reshape(-1)
        purity = np.bincount(labels_tmp).max() / len(labels_tmp)
        purity_per_cluster.append((c, purity))  # 记录聚类标签和对应的纯度
        count.append(np.bincount(labels_tmp).max())
    
    total_purity = np.sum(count) / labels_true.shape[0]
    
    # 可视化过程
    # 1. 真实标签与预测标签的分布
    plt.figure(figsize=(12, 6))

    # 绘制真实标签与预测标签的分布
    plt.subplot(1, 2, 1)
    sns.countplot(x=labels_true.reshape(-1), palette="Set2")
    plt.title("True Labels Distribution")
    plt.xlabel("True Labels")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.countplot(x=labels_pred.reshape(-1), palette="Set2")
    plt.title("Predicted Labels Distribution")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("result_img/LD.png")

    # 2. 绘制每个聚类的纯度
    cluster_labels, purity_values = zip(*purity_per_cluster)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=cluster_labels, y=purity_values, palette="Set1")
    plt.title("Purity per Cluster")
    plt.xlabel("Cluster Labels")
    plt.ylabel("Purity")
    plt.savefig("result_img/CL.png")

    return total_purity



# 定义文件路径的通配符模式
# file_pattern = "result/DMM_finally/News-TK500alpha0.1beta1_lambda0.0-|-dmm_iterNum(20)+initK(1|1)+entropy(15)+tfIdf(1|0.0)/News-TSampleNo*ClusteringResult.txt"
file_pattern = "result/origin_GSDMM/Tweet/iterNum(20)+(500|0.1|0.1)+initK(0)+entropy(-1)+tfIcf(0)/TweetSampleNo*ClusteringResult.txt"

label_path = "data/Tweet"
if __name__ == '__main__':

    # 获取所有符合条件的文件路径
    file_paths = glob.glob(file_pattern)

    # 创建一个字典，用于存储每个文件的内容
    file_data_dict = {}

    # 遍历所有文件路径，逐个读取文件的内容
    for file_path in file_paths:
        # 提取文件编号，例如将 "SampleNo1" 转为数字 1
        sample_no = int(file_path.split("SampleNo")[1].split("ClusteringResult")[0])
        
        file_data_dict[sample_no - 1] = file_path

    acc_list = []
    nmi_list = []
    ari_list = []
    f1_list = []
    # file_data_dict 现在包含每个文件的内容，键为 SampleNo 编号
    for No in range(len(file_data_dict)):
        # Load the data from the file
        pred_file = pd.read_csv(file_data_dict[No], sep=" ", header=None, names=["DocumentID", "PredictedLabel"])

        # print(len(set(pred_file['PredictedLabel'])))

        # 读取文件内容
        label_data = []

        # 逐行读取文件，解析 JSON 内容
        with open(label_path, 'r') as file:
            for line in file:
                # 解析 JSON 行
                entry = json.loads(line.strip())
                label_data.append({
                    'DocumentID': len(label_data) + 1,  # 行号作为 DocumentID
                    'Label': int(entry['cluster'])  # cluster 作为 Label
                })

        # 创建 DataFrame
        label_data = pd.DataFrame(label_data)

        metrics = cluster_accuracy(label_data["Label"], pred_file["PredictedLabel"])
        print("Metrics:", metrics)

        # metrics_2 = externalValidation(label_data["Label"], pred_file["PredictedLabel"])
        # print("Metrics_2:", metrics_2)

        # cluster_purity(label_data, pred_file)
        # acc_2 = cluster_accuracy_2(label_data["Label"], pred_file["PredictedLabel"])
        # acc_3 = cluster_accuracy_3(label_data["Label"], pred_file["PredictedLabel"], 152)
        # print(acc_2, acc_3)

        # Calculate the accuracy
        # acc, nmi, ari, f1 = metric.eva(label_data["Label"], pred_file["PredictedLabel"])

        # print(acc, nmi, ari, f1)

        acc_list.append(metrics["ACC"])
        nmi_list.append(metrics["NMI"])
        ari_list.append(metrics["ARI"])
        f1_list.append(metrics["F1-score"])

    acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
    print("ACC: {:.4f} ± {:.4f}".format(acc_list.mean(), acc_list.std()))
    print("NMI: {:.4f} ± {:.4f}".format(nmi_list.mean(), nmi_list.std()))
    print("ARI: {:.4f} ± {:.4f}".format(ari_list.mean(), ari_list.std()))
    print("F1 : {:.4f} ± {:.4f}".format(f1_list.mean(), f1_list.std()))

