import os
import random
import time
import datetime
import json
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import copy
from data import SequenceClassificationDataset
from utils import batch_to_tensor
from utils import load_checkpoint, save_npy
import argparse
from utils import load_config

def create_datastore(model, dataloader, device):
    outputs = []  # Initialize an empty list to store the iteration outputs
    labels = []
    model.to(device)
    model.eval()
    for batch in dataloader:
      with torch.no_grad():
          batch = batch_to_tensor(batch)
          for key, tensor in batch.items():
              batch[key] = tensor.to(device)
          label = batch['label_ids']

          # Forward pass
          _,output = model(batch, label, get_embeddings = True)
          outputs.append(output)
          labels.append(label)
        
    datastore_keys = torch.cat(outputs, dim=1) # torch.Size([1, M, 768])
    datastore_values = torch.cat(labels, dim=1) # torch.Size([1, M])
  
    return datastore_keys, datastore_values


def prepare_test_data(model, dataloader, device):
    embeddings = []  # Initialize an empty list to store the iteration outputs
    labels = []
    predicted_labels = []
    logits = []
    model.to(device)
    model.eval()
    for batch in dataloader:
      with torch.no_grad():
          batch = batch_to_tensor(batch)
          for key, tensor in batch.items():
              batch[key] = tensor.to(device)
          label = batch['label_ids']

          # Forward pass
          outputs, embedding = model(batch, labels=label, get_embeddings = True)
          
          # Apply argmax to get the predicted labels
          tlogit = outputs['logits'].unsqueeze(0)
          predicted_label = outputs['predicted_label'].unsqueeze(0)

          logits.append(tlogit)
          embeddings.append(embedding)
          labels.append(label)
          predicted_labels.append(predicted_label)

      
    embeddings = torch.cat(embeddings, dim=1)
    labels = torch.cat(labels, dim=1)
    logits = torch.cat(logits, dim=1)
    predicted_labels = torch.cat(predicted_labels, dim=1)

    return embeddings, labels, logits, predicted_labels






def main():
    # LOAD CONFIG
    parser = argparse.ArgumentParser(description="Run DL experiments.")
    parser.add_argument("config", type=str, help="Name of the config.json file to be loaded.")
    args = parser.parse_args()
    config_file = args.config
    mconfig = load_config(config_file)

    # GET DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # LOAD DATA
    root = dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = root + f"/datasets/{mconfig['dataset']}/"
    print('dataset_path:',dataset_path)
    train_dataset = SequenceClassificationDataset(Path(dataset_path,'train.json'))
    dev_dataset = SequenceClassificationDataset(Path(dataset_path, 'dev.json'))
    test_dataset = SequenceClassificationDataset(Path(dataset_path, 'test.json'))
    
    # LOAD DATALOADER
    batch_size=1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
     
    # LOAD SAVED MODEL
    model_path = root + f"/models/{mconfig['dataset']}/model.pth"
    model = load_checkpoint(model_path, mconfig, device)
    
    # CREATE DATASTORE
    print('CREATE DATASTORE KEYS AND VALUES')
    output_files_path = root + f"/output_files/{mconfig['dataset']}/"
    datastore_keys, datastore_values = create_datastore(model, train_dataloader, device)    
    save_npy(datastore_keys,Path(output_files_path,'datastore_keys.npy'))
    save_npy(datastore_values,Path(output_files_path,'datastore_values.npy'))
    print(f"Shape of datastore_keys: {datastore_keys.shape}")
    print(f"Shape of datastore_values: {datastore_values.shape}")
    
    # CREATE TEST FILES
    print('CREATE test file')
    embeddings, labels, logits, predicted_labels = prepare_test_data(model, test_dataloader, device)    
    save_npy(embeddings,Path(output_files_path,'test_embeddings.npy'))
    save_npy(labels,Path(output_files_path,'test_labels.npy'))
    save_npy(logits,Path(output_files_path,'test_logits.npy'))
    save_npy(predicted_labels,Path(output_files_path,'test_predicted_labels.npy'))
    print(f"Shape of test embeddings: {embeddings.shape}")
    print(f"Shape of test labels: {labels.shape}")
    print(f"Shape of test logits: {logits.shape}")
    print(f"Shape of test predicted_labels: {predicted_labels.shape}")
    
    
    # CREATE DEV FILES
    print('CREATE validation files')
    embeddings, labels, logits, predicted_labels = prepare_test_data(model, valid_dataloader, device)
    save_npy(embeddings,Path(output_files_path,'val_embeddings.npy'))
    save_npy(labels,Path(output_files_path,'val_labels.npy'))
    save_npy(logits,Path(output_files_path,'val_logits.npy'))
    save_npy(predicted_labels,Path(output_files_path,'val_predicted_labels.npy'))
    print(f"Shape of validation embeddings: {embeddings.shape}")
    print(f"Shape of validation labels: {labels.shape}")
    print(f"Shape of validation logits: {logits.shape}")
    print(f"Shape of validation predicted_labels: {predicted_labels.shape}")
if __name__ == "__main__":
  main()
    