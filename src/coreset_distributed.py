from __future__ import annotations

import pandas as pd
import argparse
import os

import math


from pathlib import Path
from transformers import AutoTokenizer, AutoModel, TextClassificationPipeline
import wandb
import numpy as np
import pandas as pd

import torch

from sentence_transformers import SentenceTransformer, models

from datasets import load_dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizerFast

import numpy as np
import pdb

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import sys

inp_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/"
out_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/acitve_out/"
default_model_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/active_learn_init/checkpoint-2352/"
default_tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/src/tmp/nyutron-big"

impression_idx = None

class Points:
    def __init__(self, X, X_set, texts):
        self.X = X
        self.X_set = X_set
        self.texts = texts
        self.min_dist, self.dist_idxs = self.init_distance()

    def pprint(self):
        print('points')
        print(self.X)
        print('current set')
        print(self.X_set)
        print('texts')
        print(self.texts)

    def pop(self):
        idx = np.argmax(self.min_dist)
        self.X  = np.delete(self.X, idx, axis = 0)
        self.min_dist  = np.delete(self.min_dist, idx)
        self.texts = np.delete(self.texts, idx)

    def get_max(self):
        idx = np.argmax(self.min_dist)
        return np.max(self.min_dist), idx, self.texts[idx]

    def init_distance(self):
        m = np.shape(self.X)[0]
        min_dists = []
        min_idxs = []
        for i in range(m):
            dists = pairwise_distances(self.X_set, [self.X[i]])
            mindist = np.amin(dists, axis=0)
            minidx = np.argmin(dists)
            min_dists.append(mindist)
            min_idxs.append(minidx)
        return np.array(min_dists), np.array(min_idxs)

    def update_distance(self, new_add):
        dists = []
        self.X_set.append(new_add)
        for cord, dist in zip(self.X, self.min_dist):
          dists.append(min(dist, pairwise_distances([cord], [new_add])))
        self.min_dist = np.array(dists)
        return self.min_dist

def reduce_max(dists, texts):
    idx = np.argmax(dists)
    return idx, texts[idx]



parser = argparse.ArgumentParser()

parser.add_argument('--budget', help='budget to be chosen', type=int, default=1000)
parser.add_argument('--ifname', help='output file name', type=str, default='budget.csv')
parser.add_argument('--ofname', help='output file name', type=str, default='train.csv')
parser.add_argument('--outdir', help='output directory', type=str, default=out_path)
parser.add_argument('--encoder', help='Type of encoder to generate embeddings', type=str, default='sentence')
parser.add_argument('--text_pth', help='budget to be chosen', type=str, default='/gpfs/home/lt2504/pathology-extractor-bert/data/raw/historical_report_parts_filter/1_historical_reports.csv')



opts = parser.parse_args()
print(opts)

BUDGET = opts.budget
folder_path = opts.outdir

input_file = opts.ifname


if not os.path.exists(folder_path):
    # If it doesn't exist, create it
    os.makedirs(folder_path)
    print(f"Folder {folder_path} created.")
else:
    print(f"Folder {folder_path} already exists.")
    
X = np.load(input_file)

msk_10k_path = "/gpfs/home/lt2504/pathology-extractor-bert/embeddings_10k/"

x1 = np.load(msk_10k_path + "out_1.npy")
x2 = np.load(msk_10k_path + "out_2.npy")
x3 = np.load(msk_10k_path + "out_3.npy")

X_set = np.vstack((x1,x2,x3))

texts = pd.read_csv(opts.text_pth)["text"].to_list()

print(texts[0])

p = Points(X, X_set, texts)

for i in range(BUDGET):
    distance, idx, text = p.get_max()
    print(distance)
    print(idx)
    print(text)
    p.pop()






