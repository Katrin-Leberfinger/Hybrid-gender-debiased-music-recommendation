import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertConfig, BertPreTrainedModel, BertModel, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
from transformers import BertModel
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
import gc


class EasyBERTModel(nn.Module):
    """
    BERT model to check if everything is fine. No further layer.
    Input: User ID + Text Embedding
    """
    def __init__(self, item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items,
                 bert_model_name):
        super().__init__()

        self.hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
        self.item_embeddings_size = item_embeddings_size
        self.user_embeddings_size = user_embeddings_size
        self.num_items = num_items
        self.num_users = num_users

        self.bert = BertModel.from_pretrained(bert_model_name, return_dict=True)
        self.embedding_user = nn.Embedding(self.num_users, 100)
        self.classifier = nn.Linear(self.bert.config.hidden_size + 100, 1)

    def forward(self, user_id, input_ids, input_text_ids, attention_mask):
        output_item = self.bert(input_text_ids, attention_mask=attention_mask)#.pooler_output
        output_item = output_item[0][:, 0, :]
        output_user = self.embedding_user(user_id)
        output = torch.cat((output_item, output_user), 1)
        output = self.classifier(output)
        return torch.sigmoid(output)


class MatrixFactorizationModel(nn.Module):
    """Matrix Factorization Model with multiple linear layers.
    Input: User ID + Track ID
    """

    def __init__(self, item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items,
                 bert_model_name):
        super().__init__()

        self.hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
        self.item_embeddings_size = item_embeddings_size
        self.user_embeddings_size = user_embeddings_size
        self.num_items = num_items
        self.num_users = num_users

        self.model1_layer1 = nn.Embedding(self.num_items, self.item_embeddings_size, max_norm=1.)
        nn.init.uniform_(self.model1_layer1.weight, a=-0.05, b=0.05)

        self.model2_layer1 = nn.Embedding(self.num_users, self.user_embeddings_size, max_norm=1.)
        nn.init.uniform_(self.model2_layer1.weight, a=-0.05, b=0.05)

        self.linear1 = nn.Linear(self.hidden_dense_layer_size, 128)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(128, 1)

        self.drop = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, input_ids):
        y1 = self.model1_layer1(input_ids)

        y2 = self.model2_layer1(user_id)
        y = torch.cat((y1, y2), 1)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return self.sigmoid(y)


class AMARBertEmbeddings(nn.Module):
    """BERT model using item ID embeddings and multiple linear layer.
    Input: User ID + Track ID + Text Embedding
    """
    def __init__(self, item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items,
                 bert_model_name):
        super().__init__()

        self.item_embeddings_size = item_embeddings_size
        self.user_embeddings_size = user_embeddings_size
        self.num_items = num_items
        self.num_users = num_users

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.model1_layer3 = nn.Dropout(p=0.3)

        self.hidden_dense_layer_size = item_embeddings_size + user_embeddings_size + self.bert.config.hidden_size

        self.model2_layer1 = nn.Embedding(self.num_users, self.user_embeddings_size)
        nn.init.uniform_(self.model2_layer1.weight, a=0.0, b=0.05)

        self.model3_layer1 = nn.Embedding(self.num_items, self.item_embeddings_size)
        nn.init.uniform_(self.model3_layer1.weight, a=0.0, b=0.05)

        self.linear1 = nn.Linear(self.hidden_dense_layer_size, self.hidden_dense_layer_size)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dense_layer_size, 32)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)

        self.linear3 = nn.Linear(32, 1)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(0.01)

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, input_ids, input_text_ids, attention_mask):
        output = self.bert(input_text_ids, attention_mask=attention_mask)
        y1 = output[0]
        # y1 = y1.mean(axis=2) # alterantive to CLS token, combine with y1[:,0,:] possible
        y1 = y1[:, 0, :]
        # "Since BERT is transformer based contextual model, the idea is [CLS] token would have captured the entire context and would be sufficient for simple downstream tasks such as classification."
        y1 = self.model1_layer3(y1)

        y2 = self.model2_layer1(user_id)

        y3 = self.model3_layer1(input_ids)

        y = torch.cat([y1, y2, y3], 1)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.linear3(y)
        return self.sigmoid(y)


class AMARBert(nn.Module):
    """BERT model with multiple linear layers.
    Input: User ID + Text Embedding
    """
    def __init__(self, item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items,
                 bert_model_name):
        super().__init__()

        self.item_embeddings_size = item_embeddings_size
        self.user_embeddings_size = user_embeddings_size
        self.num_items = num_items
        self.num_users = num_users

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.model1_layer3 = nn.Dropout(p=0.3)

        self.hidden_dense_layer_size = self.bert.config.hidden_size + user_embeddings_size

        self.model2_layer1 = nn.Embedding(self.num_users, self.user_embeddings_size)
        nn.init.uniform_(self.model2_layer1.weight, a=0.0, b=0.05)

        self.linear = nn.Linear(self.hidden_dense_layer_size, 1)
        #self.relu = nn.ReLU()
        #self.linear2 = nn.Linear(self.hidden_dense_layer_size, 128)
        #self.linear3 = nn.Linear(128, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, input_ids, input_text_ids, attention_mask):
        output = self.bert(input_text_ids, attention_mask=attention_mask)
        y1 = output[0]
        #y1 = y1.mean(axis=1) # alterantive to CLS token, combine with y1[:,0,:] possible
        y1 = y1[:, 0, :]
        # pooled_output (=y1) is the output of the CLS token
        # "Since BERT is transformer based contextual model, the idea is [CLS] token would have captured the entire context and would be sufficient for simple downstream tasks such as classification."
        # https://stackoverflow.com/questions/63673511/how-to-use-the-outputs-of-bert-model?rq=1
        # https://towardsdatascience.com/bert-to-the-rescue-17671379687f
        y1 = self.model1_layer3(y1)

        y2 = self.model2_layer1(user_id)

        y = torch.cat([y1, y2], 1)
        y = self.linear(y)
       # y = self.relu(y)
    #    y = self.linear2(y)
    #    y = self.linear3(y)
        return self.sigmoid(y)


class AMARBertTwoInput(nn.Module):
    def __init__(self, item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items,
                 bert_model_name):
        super().__init__()

        self.hidden_dense_layer_size = item_embeddings_size + user_embeddings_size
        self.item_embeddings_size = item_embeddings_size
        self.user_embeddings_size = user_embeddings_size
        self.num_items = num_items
        self.num_users = num_users

        self.bert1 = BertModel.from_pretrained(bert_model_name)
        self.model1_layer3 = nn.Dropout(p=0.2)

        self.bert2 = BertModel.from_pretrained(bert_model_name)
        self.model4_layer3 = nn.Dropout(p=0.2)

        self.model2_layer1 = nn.Embedding(self.num_users, self.user_embeddings_size, max_norm=1.)
        nn.init.uniform_(self.model2_layer1.weight, a=0.0, b=0.05)

        self.model3_layer1 = nn.Embedding(self.num_items, self.item_embeddings_size, max_norm=1.)
        nn.init.uniform_(self.model3_layer1.weight, a=0.0, b=0.05)

        self.linear1 = nn.Linear(self.hidden_dense_layer_size, self.hidden_dense_layer_size)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(0.01)

        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dense_layer_size, hidden_layer)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.fill_(0.01)

        self.linear3 = nn.Linear(hidden_layer, 1)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(0.01)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.bert1(x[1], attention_mask=x[3])
        y1 = output[0]
        # y1 = y1.mean(axis=2) # alterantive to CLS token, combine with y1[:,0,:] possible
        y1 = y1[:, 0, :]
        # "Since BERT is transformer based contextual model, the idea is [CLS] token would have captured the entire context and would be sufficient for simple downstream tasks such as classification."
        y1 = self.model1_layer3(y1)

        output = self.bert2(x[4], attention_mask=x[5])
        y4 = output[0]
        # y1 = y1.mean(axis=2) # alterantive to CLS token, combine with y1[:,0,:] possible
        y4 = y4[:, 0, :]
        # "Since BERT is transformer based contextual model, the idea is [CLS] token would have captured the entire context and would be sufficient for simple downstream tasks such as classification."
        y4 = self.model4_layer3(y4)

        y2 = self.model2_layer1(x[2])

        y3 = self.model3_layer1(x[0])

        y = torch.cat([y1, y2, y3, y4], 1)

        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        y = self.linear3(y)

        return self.sigmoid(y)