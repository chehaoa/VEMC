import json
import os
from Document import Document
import numpy as np
import pandas as pd
import random

def convert_label_to_ids(labels):
    # unique_labels = list(dict.fromkeys(labels))  # 保持顺序
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]
    return np.asarray(label_ids), n_clusters

class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0
        self.documents = []
        self.labels = []
        if dataDir[-4:] == "json":
            with open(dataDir, "r", encoding="utf-8") as json_file:
                loaded_data = json.load(json_file)
            for i, obj in enumerate(loaded_data):
                self.D += 1
                text = obj['text']
                document = Document(text, wordToIdMap, wordList, int(i+1))
                self.documents.append(document)
                self.labels.append({
                        'DocumentID': i + 1,
                        'Label': int(obj['label']-1)
                    })
        elif "DMM_datasets" in dataDir:
            with open(dataDir) as input:
                line = input.readline()
                while line:
                    self.D += 1
                    obj = json.loads(line)
                    text = obj['text']
                    document = Document(text, wordToIdMap, wordList, self.D)
                    self.documents.append(document)
                    self.labels.append({
                        'DocumentID': self.D,
                        'Label': int(obj['cluster'])-1
                    })
                    line = input.readline()
        elif "sccl_datasets" in dataDir:
            data_entries = []
            with open(dataDir, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    parts = line.strip().split('\t', 1)
                    if len(parts) != 2:
                        continue
                    cluster_id, text = parts
                    data_entries.append((int(cluster_id), text))
            random.shuffle(data_entries)
            for cluster_id, text in data_entries:
                self.D += 1
                document = Document(text, wordToIdMap, wordList, self.D)
                self.documents.append(document)
                self.labels.append({
                    'DocumentID': self.D,
                    'Label': int(cluster_id)-1
                })
        elif "LLM_datasets" in dataDir:
            data_entries = []
            labels = []
            with open(dataDir, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    obj = json.loads(line)
                    text = obj['input']
                    label = obj['label']
                    data_entries.append((text, label))
                    labels.append(label)
            
            # random.shuffle(data_entries)
            label_ids, _ = convert_label_to_ids(labels)
            
            for i, (text, label) in enumerate(data_entries):
                self.D += 1
                document = Document(text, wordToIdMap, wordList, self.D)
                self.documents.append(document)
                self.labels.append({
                    'DocumentID': self.D,
                    'Label': label_ids[i]
                })
        
        self.labels = pd.DataFrame(self.labels)["Label"]
        print("number of documents is ", self.D)
