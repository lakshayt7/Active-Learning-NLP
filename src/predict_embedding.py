from __future__ import annotations

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, AutoModel
import wandb
import numpy as np
import pandas as pd

import math

import os

import torch

import sys

from data_processing.dataset_loader import get_dataset
from data_processing.dataset_utils import get_texts, get_true_labels_binarized
from metrics_new import calculate_metrics_simple

import pandas as pd
from datasets import load_dataset, DatasetDict
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer, models

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


def predict(text: str | list[str], tokenizer, model, threshold: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Move the tokenizer to the specified device (if applicable)
    if hasattr(tokenizer, 'to'):
        tokenizer.to(device)


    pipe = TextClassificationPipeline(tokenizer=tokenizer, model=model, top_k=None, device='cuda:0')
    
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

def get_dataset_inference(data_dir: Path  , tokenizer: None | PreTrainedTokenizerFast = None):
    dataset = load_dataset("csv", data_files=str(data_dir / "msk_filtered.csv"))

    dataset = DatasetDict()
    dataset["train"] = train_dataset["train"]

    if tokenizer is None:
        return dataset, None
    encoded_dataset = dataset.map(preprocess_dataset_pretraining, batched=True, remove_columns=dataset["train"].column_names,
                                  fn_kwargs={"tokenizer": tokenizer})
    encoded_dataset.set_format("torch")
    return dataset, encoded_dataset

def encode_texts(texts, tokenizer, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define batch size
    batch_size = 128

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

        outputs = model.encode(batch_texts)  
        outputs_np = np.array(outputs)

        intermediate_outputs.extend(outputs_np)
        
    return intermediate_outputs


def main():
    wandb.init(mode="disabled")
    
    data_dir = Path("/gpfs/home/lt2504/pathology-extractor-bert/data/splits").resolve()
    #model_name_or_path = sys.argv[-1]x

    model_name_or_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/bignyutron39/checkpoint-7931"
    tokenizer_path = "/gpfs/home/lt2504/pathology-extractor-bert/models/pretrained/historical/bignyutron39/checkpoint-7931"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
    
    
    model = SentenceTransformer(model_name_or_path)
  

    directory_path = "/gpfs/data/chopralab/ad6489/pathology-extractor-bert/data/splits/msk_10K/"
    directory_path = "/gpfs/home/lt2504/pathology-extractor-bert/data/raw/historical_report_parts_filter/"
    files = os.listdir(directory_path) 
    
    outs = []

    i = 0
    #files = ['train.csv', 'test.csv', 'val.csv']
    
    for file_name in files:
        i+=1
        print(file_name)
        file_path = os.path.join(directory_path, file_name)
        df =   pd.read_csv(file_path, encoding = 'cp850')
        texts = df["text"].to_list()
        encodings = encode_texts(texts, tokenizer, model)
        print(i)
        np.save(f'/gpfs/home/lt2504/pathology-extractor-bert/embeddings_filter/out_{i}.npy', np.array(encodings))
        
       
    
if __name__ == "__main__":
    main()
