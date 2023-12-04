from __future__ import annotations

import pandas as pd
import argparse
import os

import math


from pathlib import Path
from transformers import AutoTokenizer, AutoModel, TextClassificationPipeline
import wandb
import pandas as pd

import torch


import numpy as np
import pdb


import pickle
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import sys

inp_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/"
out_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/raw/selected/"
default_model_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/active_learn_init/checkpoint-2352/"
default_tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/src/tmp/nyutron-big"

impression_idx = None

class Points:
    def __init__(self, X, X_set, texts, idxs):
        self.X = X
        self.X_set = X_set
        self.texts = texts
        self.idxs = idxs
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
        self.X_set = np.vstack([self.X_set, new_add])
        for cord, dist in zip(self.X, self.min_dist):
            p = min(dist, int(pairwise_distances([cord], [new_add])[0].item()))

            dists.append(int(p))

        self.min_dist = np.array(dists)
        return self.min_dist

def reduce_max(dists, texts):
    idx = np.argmax(dists)
    return idx, texts[idx]

def create_dataframe(strings_list, column_name='Strings'):
    # Create a dictionary with the specified column name and the list of strings
    data = {column_name: strings_list}
    
    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    return df



parser = argparse.ArgumentParser()

parser.add_argument('--budget', help='budget to be chosen', type=int, default=1000)
parser.add_argument('--ifname', help='output file name', type=str, default='budget.csv')
parser.add_argument('--ofname', help='output file name', type=str, default='train.csv')
parser.add_argument('--outdir', help='output directory', type=str, default=out_path)
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

data_choose = pd.read_csv(opts.text_pth)
texts = data_choose["text"].to_list()
idxs = data_choose.index.to_list()

print(texts[0])

p = Points(X, X_set, texts, idxs)

texts = []

for i in range(BUDGET):
    distance, idx, text = p.get_max()
    p.update_distance(X[idx])
    texts.append(text)
    p.pop()


df = create_dataframe(texts, column_name='text')
df.to_csv(opts.outdir+opts.ofname , index = False)
