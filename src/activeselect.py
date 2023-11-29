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

def true_indices(arr):
    return [index for index, value in enumerate(arr) if value]

def get_texts(dataset: DatasetDict, train_or_val: str, idx: int | None = None) -> str | list[str]:
    """Extracts all the texts from a dataset.

    The text can be either the impression or the report, depending on how the dataset was created.

    If an index is given, only one impression/report is returned.

    """
    if idx is None:
        return dataset[train_or_val]["text"]
    else:
        return dataset[train_or_val]["text"][idx]

def preprocess_data(batch: LazyBatch, tokenizer: PreTrainedTokenizerFast, labels: list[str]):
    """Tokenizes and encodes the labels. This function will be passed to map to process multiple batches at once."""
    encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

    # Add labels
    labels_batch = {k: batch[k] for k in batch.keys() if k in labels}

    # Create numpy array of shape (batch_size, num_labels) and fill it
    labels_matrix = np.zeros((len(batch["text"]), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding

def get_dataset(data_csv , tokenizer):
    ds = load_dataset("csv", data_files= data_csv)

      

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Temporary to speed up training while debugging
    #train_dataset["train"] = train_dataset["train"].select(range(10))
    #test_dataset["train"] = test_dataset["train"].select(range(10))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Ensure we take all labels, even though the features in train and test should be the same
    labels_str = ds["train"].features.keys() 
    labels_str = sorted({x for x in labels_str if x not in ["id", "text"]})
    id2label = {idx: label for idx, label in enumerate(labels_str)}
    label2id = {label: idx for idx, label in enumerate(labels_str)}


    # If tokenizer is not provided, only load the datasets, not the encoded datasets
    if tokenizer is None:
        return ds, None, id2label, label2id

    encoded_dataset = ds.map(preprocess_data, batched=True, remove_columns=ds['train'].column_names,
                                  fn_kwargs={"tokenizer": tokenizer, "labels": labels_str})
    encoded_dataset.set_format("torch")

    return ds, encoded_dataset, id2label, label2id

def predict(text: str | list[str], tokenizer, model, threshold: float):
    pipe = TextClassificationPipeline(tokenizer=tokenizer, model=model, top_k=None)
    preds_with_probs = pipe(text, padding="max_length", truncation=True, max_length=512)

    pred_labels = []
    pred_scores = []
    
    label_names = sorted([pred["label"] for pred in preds_with_probs[0]])

    
    for preds in preds_with_probs:
        pred_labels.append([pred["label"] for pred in preds if pred["score"] > threshold])

        # Sort by label so that every prediction vector has the same order
        sorted_preds = sorted(preds, key=lambda k: k['label'])
        pred_scores.append(np.array([pred["score"] for pred in sorted_preds]))
        
    # If no labels are predicted, it means it is a normal case
    # for preds in pred_labels:
    #     if not preds:
    #         preds.append("normal")

    return pred_labels, np.array(pred_scores), label_names

def encode_texts(dataset, tokenizer, model, encoder_type):
    #

    texts = get_texts(dataset, "train", impression_idx)

    # Define batch size
    batch_size = 1

    # Calculate the total number of batches
    num_batches = math.ceil(len(texts) / batch_size)

    # Initialize a list to store the intermediate outputs
    intermediate_outputs = []

    for i in range(num_batches):
        # Get the current batch of texts
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]

        # Tokenize and pad/truncate the current batch
        tokenized_batch = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)

        # Move the input data to the GPU
        tokenized_batch.to(device)

        # Forward pass through the model for the current batch
        if encoder_type == 'sentence':
            outputs = model.encode(batch_texts)  
            outputs_np = outputs
        elif encoder_type == 'pooler':
            print(tokenized_batch)
            with torch.no_grad():
                outputs = model(**tokenized_batch)
            outputs_np = outputs.pooler_output.cpu().numpy()
            
        intermediate_outputs.extend(outputs_np)
        
    return intermediate_outputs

def init_distance(X, X_set):
    m = np.shape(X)[0]
    min_dists = []
    for i in range(m):
        dists = pairwise_distances(X, [X[i]])
        mindist = np.amin(dists, axis=1)
        min_dists.append(mindist)
    return np.array(min_dists)
        

def furthest_first(X, X_set, n):
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, [X[idx]])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs
    
def k_means_active(X, n):
    X = np.array(X)
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(X)

    cluster_idxs = cluster_learner.predict(X)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (X - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(X.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
    return q_idxs
    

def k_center(X, n, k_mean=100):

    lb_flag = np.array([False]*np.shape(X)[0])
    embedding = np.array(X)

    ret = k_means_active(X,k_mean)
    n = n - int(n/20)
    print('K means initialization')
    print(ret)
    for i in ret:
        lb_flag[i] = True

    from datetime import datetime

    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(X), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    
    print(datetime.now() - t_start)

    mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(n):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(np.shape(X)[0])[~lb_flag][q_idx_]
        lb_flag[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

    return true_indices(lb_flag)

import copy

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import numpy as np

class GraphDensitySampler():
    def __init__(self, X, seed=420):
        self.name = 'graph_density'
        self.X = X 
        self.flat_X = X
        # Set gamma for gaussian kernel to be equal to 1/n_features
        self.gamma = 1. / self.X.shape[1]
        self.compute_graph_density()

    def compute_graph_density(self, n_neighbor=15):
        # kneighbors graph is constructed using k=15
        connect = kneighbors_graph(self.flat_X, n_neighbor,p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0],neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances(self.flat_X[[i]],self.flat_X[[j]],metric='manhattan')
            distance = distance[0,0]
            weight = np.exp(-distance * self.gamma)
            connect[i,j] = weight
            connect[j,i] = weight
        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        self.graph_density = np.zeros(self.X.shape[0])
        for i in np.arange(self.X.shape[0]):
            self.graph_density[i] = connect[i,:].sum() / (connect[i,:]>0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def select_batch_(self, N, already_selected, **kwargs):
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        batch = set()
        self.graph_density[already_selected] = min(self.graph_density) - 1
        while len(batch) < N:
            selected = np.argmax(self.graph_density)
            neighbors = (self.connect[selected,:] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            batch.add(selected)
            self.graph_density[already_selected] = min(self.graph_density) - 1
            self.graph_density[list(batch)] = min(self.graph_density) - 1
        return list(batch)


inp_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/"
out_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/acitve_out/"
default_model_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/active_learn_init/checkpoint-2352/"
default_tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/src/tmp/nyutron-big"

impression_idx = None


parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--budget', help='budget to be chosen', type=int, default=200)
parser.add_argument('--ifname', help='output file name', type=str, default='budget.csv')
parser.add_argument('--ofname', help='output file name', type=str, default='train.csv')
parser.add_argument('--outdir', help='output directory', type=str, default=out_path)
parser.add_argument('--encoder', help='Type of encoder to generate embeddings', type=str, default='sentence')

parser.add_argument('--tokenizer_path', help='tokenizer path', type=str, default=default_tokenizer_path)
parser.add_argument('--model_path', help='model path', type=str, default=default_model_path)




opts = parser.parse_args()
print(opts)

BUDGET = opts.budget
folder_path = opts.outdir
tokenizer_path = opts.outdir
encoder_type = 'tfidf'#opts.encoder

model_name_or_path = opts.model_path
tokenizer_path = opts.tokenizer_path

if not os.path.exists(folder_path):
    # If it doesn't exist, create it
    os.makedirs(folder_path)
    print(f"Folder {folder_path} created.")
else:
    print(f"Folder {folder_path} already exists.")
    

out_path_save = opts.outdir + opts.ofname
input_file_path = inp_path+opts.ifname

train_file_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/splits/active/train.csv"

df = pd.read_csv(input_file_path)
df_train = pd.read_csv(train_file_path)

#

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
dataset, encoded_dataset, id2label, label2id = get_dataset(input_file_path, tokenizer)
dataset_train, encoded_dataset_train, id2label_train, label2id_train = get_dataset(train_file_path, tokenizer)
model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)

model = SentenceTransformer(model_name_or_path)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if encoder_type != 'tfidf':
    train_embedding = encode_texts(dataset_train, tokenizer, model, encoder_type)
    active_embedding = encode_texts(dataset, tokenizer, model, encoder_type)
else:
    train_text = get_texts(dataset_train, "train", impression_idx)
    active_text = get_texts(dataset, "train", impression_idx)
    documents = train_text + active_text
    tfidf_vectorizer = TfidfVectorizer()
    tfidffit= tfidf_vectorizer.fit(documents)
    train_embedding = tfidffit.transform(train_text).toarray()
    active_embedding = tfidffit.transform(active_text).toarray()

df_sel = pd.DataFrame()

        
if opts.alg == 'rand':
    df_sel = df.sample(n=BUDGET)
    

if opts.alg == 'coreset-simple':
    idx = furthest_first(active_embedding, train_embedding, BUDGET)
    print(idx)
    df_sel = df.iloc[idx]
    
if opts.alg == 'k-means':
    idx = k_means_active(active_embedding, BUDGET)
    print(idx)
    df_sel = df.iloc[idx]
    
if opts.alg == 'k-center':
    idx = k_center(active_embedding, BUDGET)
    print(idx)
    df_sel = df.iloc[idx]
    
if opts.alg == 'graph':
    embeddings = np.vstack((active_embedding, train_embedding))
    graphSampler = GraphDensitySampler(embeddings)
    as_ = [np.shape(active_embedding)[0]+i for i in range(np.shape(train_embedding)[0])]
    idx = graphSampler.select_batch_(N = BUDGET, already_selected = as_)
    print(idx)
    df_sel = df.iloc[idx]
       

#if opts.alg == 'core-simple':

df_stats = df_sel.drop(columns = ['text', 'id'])
df_stats = df_stats.apply(pd.to_numeric, errors='coerce')
row_sums = df_stats.sum().tolist()

plt.bar(range(len(row_sums)), row_sums)

# Set labels and title
plt.xlabel('Pathology')
plt.ylabel('Frequency in selected reports')
plt.title('Bar Plot of pathologies')

# Save the plot as an image (e.g., as a PNG file)
plt.savefig(f'{opts.alg}_bar_plot.png')

print(row_sums)

probs = np.array(row_sums)/BUDGET
print(probs)
entropy = -sum(p * math.log2(p) for p in probs.tolist() if p > 0)
print(entropy)


df_final = pd.concat([df_train, df_sel]).sample(frac=1).reset_index(drop=True)   
print('output file path')
print(out_path_save)
df_final.to_csv(out_path_save, index = False)
    