import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import json


class SequenceClassificationDataset(Dataset):
    def __init__(self, file_path, dataset_name):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.data = self.load_documents()
        self.input_ids = []
        self.doc_name = []
        self.label_ids = []
        self.sentence_masks = []
        self.attention_masks = []
        self.process_data()
        self.label_mapper = self.get_label_mapper(dataset_name)

    def load_documents(self):
        with open(self.file_path) as f:
            data_dict = json.load(f)
            return data_dict

    
    def process_data(self):
                
        for key in self.data:
            doc_data = self.data[key]
            input_ids = doc_data["input_ids"]
            labels = doc_data["label_ids"]
            attention_mask = doc_data["attention_mask"]
            sentence_mask = doc_data["sentence_mask"]
            doc_name = doc_data["doc_name"]
            
            self.input_ids.append(input_ids)
            self.label_ids.append(labels)
            self.attention_masks.append((attention_mask))
            self.sentence_masks.append((sentence_mask))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        label_ids = self.label_ids[index]
        labels = []
        if self.dataset_name == 'paheli':
            for label_id in label_ids:
                labels.append(self.label_mapper[label_id])
            labels = torch.tensor(labels)
        else:
            labels = torch.tensor([self.label_mapper[i] for i in label_ids])
        attention_masks = self.attention_masks[index]
        sentence_masks = self.sentence_masks[index]
        
        return {'input_ids':input_ids, 'label_ids':labels, 'attention_mask':attention_masks, 'sentence_mask':sentence_masks} 

    def collate_fn(self, batch):
        # Sort the batch by document length in descending order
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        documents, labels = zip(*batch)
        
        # Convert documents to tensors
        documents = [torch.tensor(doc) for doc in documents]
        
        # Calculate the actual lengths of the sequences
        lengths = [len(doc) for doc in documents]  
        
        # Pad sequences within the batch
        padded_documents = pad_sequence(documents, batch_first=True)

        # Convert labels to a tensor
        label_tensors = [torch.tensor(label) for label in labels]
        label_tensor = torch.cat(label_tensors)
        
        return padded_documents, label_tensor, lengths
    def get_label_mapper(self, dataset_name):
        label_mappers = {
            'malik_cl': {
                "Fact": 0,
                "Argument": 1,
                "Precedent": 2,
                "Ratio": 3,
                "RulingL": 4,
                "RulingP": 5,
                "Statute": 6,
            },
            'malik_it': {
                "Fact": 0,
                "Argument": 1,
                "Precedent": 2,
                "Ratio": 3,
                "RulingL": 4,
                "RulingP": 5,
                "Statute": 6,
            },
            'paheli': {
                'Argument': 0,
                'Statute': 1,
                'Facts': 2,
                'Ratio of the decision': 3,
                'Ruling by Lower Court': 4,
                'Ruling by Present Court': 5,
                'Precedent': 6,
            },
            'build': {
                "PREAMBLE": 0,
                "FAC": 1,
                "RLC": 2,
                "ISSUE": 3,
                "ARG_PETITIONER": 4,
                "ARG_RESPONDENT": 5,
                "ANALYSIS": 6,
                "STA": 7,
                "PRE_RELIED": 8,
                "PRE_NOT_RELIED": 9,
                "RATIO": 10,
                "RPC": 11,
                "NONE": 12,
            }
        }

        return label_mappers.get(dataset_name, 'not implemented')