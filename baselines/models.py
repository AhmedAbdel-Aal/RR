from allennlp.common.util import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import masked_mean, masked_softmax
import copy

from transformers import BertModel

from allennlp.modules import ConditionalRandomField
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class CRFOutputLayer(torch.nn.Module):
    ''' CRF output layer consisting of a linear layer and a CRF. '''
    def __init__(self, in_dim, num_labels):
        super(CRFOutputLayer, self).__init__()
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(in_dim, self.num_labels)
        self.crf = ConditionalRandomField(self.num_labels)

    def forward(self, x, mask, labels=None):
        ''' x: shape: batch, max_sequence, in_dim
            mask: shape: batch, max_sequence
            labels: shape: batch, max_sequence
        '''

        batch_size, max_sequence, in_dim = x.shape

        logits = self.classifier(x)
        outputs = {}
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask)
            loss = -log_likelihood
            outputs["loss"] = loss
        else:
            best_paths = self.crf.viterbi_tags(logits, mask)
            predicted_label = [x for x, y in best_paths]
            predicted_label = [pad_sequence_to_length(x, desired_length=max_sequence) for x in predicted_label]
            predicted_label = torch.tensor(predicted_label)
            outputs["predicted_label"] = predicted_label
            outputs["logits"] = logits
        return outputs




class AttentionPooling(torch.nn.Module):
    def __init__(self, in_features, dimension_context_vector_u=200, number_context_vectors=5):
        super(AttentionPooling, self).__init__()
        self.dimension_context_vector_u = dimension_context_vector_u
        self.number_context_vectors = number_context_vectors
        self.linear1 = torch.nn.Linear(in_features=in_features, out_features=self.dimension_context_vector_u, bias=True)
        self.linear2 = torch.nn.Linear(in_features=self.dimension_context_vector_u,
                                       out_features=self.number_context_vectors, bias=False)

        self.output_dim = self.number_context_vectors * in_features

    def forward(self, tokens, mask):
        #shape tokens: (batch_size, tokens, in_features)

        # compute the weights
        # shape tokens: (batch_size, tokens, dimension_context_vector_u)
        a = self.linear1(tokens)
        a = torch.tanh(a)
        # shape (batch_size, tokens, number_context_vectors)
        a = self.linear2(a)
        # shape (batch_size, number_context_vectors, tokens)
        a = a.transpose(1, 2)
        a = masked_softmax(a, mask)

        # calculate weighted sum
        s = torch.bmm(a, tokens)
        s = s.view(tokens.shape[0], -1)
        return s



class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]
        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]


        if not self.bert_trainable:
            # cache the embeddings of BERT if it is not fine-tuned
            # to save GPU memory put the values on CPU
            batch["bert_embeddings"] = bert_embeddings.to("cpu")

        return bert_embeddings

class BertHSLN(torch.nn.Module):
    '''
    Model for Baseline, Sequential Transfer Learning and Multitask-Learning with all layers shared (except output layer).
    '''
    def __init__(self, config, num_labels):
        super(BertHSLN, self).__init__()
        self.use_crf = config['use_crf']
        self.num_labels = num_labels
        self.bert = BertTokenEmbedder(config)

        # Jin et al. uses DROPOUT WITH EXPECTATION-LINEAR REGULARIZATION (see Ma et al. 2016),
        # we use instead default dropout
        self.dropout = torch.nn.Dropout(config["dropout"])

        self.lstm_hidden_size = config["word_lstm_hs"]

        self.classifier = torch.nn.Linear(self.lstm_hidden_size * 2, self.num_labels)

        self.word_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=self.bert.bert_hidden_size,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))

        self.attention_pooling = AttentionPooling(2 * self.lstm_hidden_size,
                                                  dimension_context_vector_u=config["att_pooling_dim_ctx"],
                                                  number_context_vectors=config["att_pooling_num_ctx"])

        self.init_sentence_enriching(config)
        self.reinit_output_layer(config)

    def init_sentence_enriching(self, config):
        input_dim = self.attention_pooling.output_dim
        print(f"Attention pooling dim: {input_dim}")
        self.sentence_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_size=input_dim,
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True))


    def reinit_output_layer(self, config):
        input_dim = self.lstm_hidden_size * 2
        self.crf = CRFOutputLayer(in_dim=input_dim, num_labels=self.num_labels)
        
        
    def forward(self, batch, labels=None, get_embeddings = False):

        documents, sentences, tokens = batch["input_ids"].shape

        # shape (documents*sentences, tokens, 768)
        bert_embeddings = self.bert(batch)

        # in Jin et al. only here dropout
        bert_embeddings = self.dropout(bert_embeddings)

        tokens_mask = batch["attention_mask"].view(-1, tokens)
        # shape (documents*sentences, tokens, 2*lstm_hidden_size)
        bert_embeddings_encoded = self.word_lstm(bert_embeddings, tokens_mask)


        # shape (documents*sentences, pooling_out)
        # sentence_embeddings = torch.mean(bert_embeddings_encoded, dim=1)
        sentence_embeddings = self.attention_pooling(bert_embeddings_encoded, tokens_mask)
        # shape: (documents, sentences, pooling_out)
        sentence_embeddings = sentence_embeddings.view(documents, sentences, -1)
        # in Jin et al. only here dropout
        sentence_embeddings = self.dropout(sentence_embeddings)

        
        sentence_mask = batch["sentence_mask"]

        # shape: (documents, sentence, 2*lstm_hidden_size)
        sentence_embeddings_encoded = self.sentence_lstm(sentence_embeddings, sentence_mask)
        # in Jin et al. only here dropout
        sentence_embeddings_encoded = self.dropout(sentence_embeddings_encoded)
        

        logits = self.classifier(sentence_embeddings_encoded)
        if self.use_crf:
          output = self.crf(sentence_embeddings_encoded, sentence_mask, labels)
        else:
          output = {}
          if labels is not None:
            logits = logits.squeeze()
            labels = labels.squeeze()
            predicted_labels = torch.argmax(logits, dim=1)
            output['predicted_label'] = predicted_labels
            
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss
            output['logits']=logits
          else:
            logits = logits.squeeze()
            predicted_labels = torch.argmax(logits, dim=1)
            output['predicted_label'] = predicted_labels
            output['logits']=logits


        if get_embeddings:
          return output, sentence_embeddings_encoded
        
        return output
