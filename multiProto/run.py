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
from models import BertHSLN
import functools
import operator
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from train import training_step, validation_step, testing_step
from utils import save_model_state_dict, load_config, save_config
import argparse
import json
import os

seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


BERT_MODEL = "bert-base-uncased"


parser = argparse.ArgumentParser(description="Run DL experiments.")
parser.add_argument("config", type=str, help="Name of the config.json file to be loaded.")
args = parser.parse_args()
config_file = args.config

# Load the configuration from the JSON file
mconfig = load_config(config_file)
print(f'loaded config file {mconfig}')


# Replace slashes with underscores or another allowed character
today = datetime.datetime.now().strftime('%Y_%m_%d')
root = dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(root, f"outputs_{today}", mconfig['dataset'])

# Create the directory if it doesn't exist
if not os.path.exists(os.path.join(root, f"outputs_{today}")):
    os.makedirs(os.path.join(root, f"outputs_{today}"))    

# Create the directory if it doesn't exist
if not os.path.exists(os.path.join(root, f"outputs_{today}", mconfig['dataset'])):
    os.makedirs(os.path.join(root, f"outputs_{today}", mconfig['dataset']))    

#save the experiment config
save_config(mconfig, os.path.join(save_path, 'exp_config.json'))


# LOAD DATA
root = dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = root + f"/datasets/{mconfig['dataset']}/"
print('dataset_path:',dataset_path)
train_dataset = SequenceClassificationDataset(Path(dataset_path,'train.json'), mconfig['dataset'])
dev_dataset = SequenceClassificationDataset(Path(dataset_path, 'dev.json'), mconfig['dataset'])
test_dataset = SequenceClassificationDataset(Path(dataset_path, 'test.json'), mconfig['dataset'])

print(f" size of train:{len(train_dataset)}, size of dev:{len(dev_dataset)}, size of test:{len(test_dataset)}")

# INIT DATA LOADERS
batch_size=1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# INIT MODEL
print('Cude is available:',torch.cuda.is_available())
model = BertHSLN(mconfig, num_labels = mconfig['nlabels'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device used: {device}")
model.to(device)

# INIT OPTIMIZER
optimizer = Adam(model.parameters(), lr=mconfig['lr'])
epoch_scheduler = StepLR(optimizer, step_size=20, gamma=mconfig["lr_epoch_decay"])

# START TRAINING
train_epoch_losses = {'loss': [], 'cls':[], 'sc':[], 'd':[]}

dev_metrics = {'loss':[], 'f1':[]}

best_dev_f1 = 0
best_model = None

epochs = mconfig['max_epochs']

# Training loop
for epoch in range(epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))

    # # Training
    train_loss = training_step(model, optimizer, epoch_scheduler, train_dataloader, device)
    train_epoch_losses['loss'].append(train_loss['loss'])
    train_epoch_losses['cls'].append(train_loss['cls'])
    train_epoch_losses['sc'].append(train_loss['sc'])
    train_epoch_losses['d'].append(train_loss['d'])
    print(f"loss {train_loss['loss']} cls_loss {train_loss['cls']}, sc {train_loss['sc']}, diversity {train_loss['d']}")

    # Validation
    dev_f1, dev_loss = validation_step(model, valid_dataloader, device)
    dev_metrics['loss'].append(dev_loss)
    dev_metrics['f1'].append(dev_f1)
    print(f"Epoch {epoch+1}/{epochs} - dev loss {dev_loss} - F1 {dev_f1}")

    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_model = copy.deepcopy(model)
        print(f'saving the checkpoint with dev f1 - {best_dev_f1}')

# TESTING
test_loss, test_accuracy, predictions, true_labels = testing_step(best_model, test_dataloader, device)

# SAVE BEST MODEL
model_save_path = os.path.join(save_path, 'model.pth')
print('model_save_path:',model_save_path)
save_model_state_dict(best_model, model_save_path)




prototypes = best_model.proto_sim_model.prototypes.weight.data.cpu().numpy()
np.save(os.path.join(save_path, 'prototypes.npy'), prototypes)


#save metrics
with open(os.path.join(save_path, 'train_metrics.json'), 'w') as f:
    json.dump(train_epoch_losses, f)
print('saved train metric in ',os.path.join(save_path, 'train_metrics.json'),)

with open(os.path.join(save_path, 'dev_metrics.json'), 'w') as f:
    json.dump(dev_metrics, f)
print('saved dev metric in ',os.path.join(save_path, 'dev_metrics.json'))




import matplotlib.pyplot as plt

# Plot the epoch loss
plt.plot(train_epoch_losses['loss'], label='Training Loss')
plt.plot(train_epoch_losses['cls'], label='cls Loss')
plt.plot(train_epoch_losses['sc'], label='sc Loss')
plt.plot(train_epoch_losses['d'], label='diversity Loss')
plt.plot(dev_metrics['loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Epoch Loss')
plt.savefig(os.path.join(save_path, 'losses_plt.png'))
plt.show()
plt.clf()
# Plot the epoch accuracies
plt.plot(dev_metrics['f1'], label='Validation f1')
plt.xlabel('Epoch')
plt.ylabel('f1')
plt.legend()
plt.title('Epoch f1')
plt.savefig(os.path.join(save_path, 'dev_f1_plt.png'))
plt.show()