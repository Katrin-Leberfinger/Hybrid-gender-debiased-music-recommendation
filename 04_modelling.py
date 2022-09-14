# -*- coding: utf-8 -*-
"""04_modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17AAuhBxaitKEO2fhVkKloTfZ4nvTQ11r
"""

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
import matplotlib.pyplot as plt
import gc
from torch.utils.data import DataLoader
import os

from dataloaders import LFM2bDataset, LFM2bDatasetMF
from model import EasyBERTModel, AMARBertEmbeddings, AMARBert, MatrixFactorizationModel
from evaluation_metrics import get_coverage, get_ndcg

os.environ["TOKENIZERS_PARALLELISM"] = "false"

gc.collect()
torch.cuda.empty_cache()

"""## Read Data: Lyrics """

dir = "preprocessed_data_binary/"

data_interaction = pd.read_csv(dir + "data_interaction.csv").drop(['Unnamed: 0'],axis=1)
data_tracks_tags_lyrics = pd.read_csv(dir + "data_tracks_tags_lyrics.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_test = pd.read_csv(dir + "data_interaction_test.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_train = pd.read_csv(dir + "data_interaction_train.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_val = pd.read_csv(dir + "data_interaction_val.csv").drop(['Unnamed: 0'],axis=1)

# url_test = "https://drive.google.com/uc?export=download&id=1KeLTmgp83d9WleEigGAUHtMQw2OP9q8t&confirm=t&uuid=080e048b-c305-4cd2-b614-30a9539ec6f5"
# url_train = "https://drive.google.com/uc?export=download&id=1-2PgdIUSZIpw8_Z8ed2aJbg0Xb88nPq7&confirm=t&uuid=20543e6a-0aa9-441a-9455-dabcce18dd09"
# url_val = "https://drive.google.com/uc?export=download&id=1-7lEgrhd4I8LgYPALy9Om_KYjADzjHQV&confirm=t&uuid=fe3e0bc2-3f45-4c02-94bb-e6e43ef9d78d"
# url_interaction = 'https://drive.google.com/file/d/1-9Ng6q7ifGPlkn1g-C2yJ83m1z0zUwdD/view?usp=sharing'
# url_tracks = "https://drive.google.com/file/d/1-Bnc2sBl5xsTOTdzo2Wha-QcpBYwJ3MD/view?usp=sharing"

# data_interaction = pd.read_csv(path + url_interaction.split('/')[-2]).drop(['Unnamed: 0'],axis=1)
# data_tracks_tags_lyrics = pd.read_csv(path + url_tracks.split('/')[-2]).drop(['Unnamed: 0'],axis=1)
# data_interaction_test = pd.read_csv(url_test).drop(['Unnamed: 0'],axis=1)
# data_interaction_train = pd.read_csv(url_train).drop(['Unnamed: 0'],axis=1)
# data_interaction_val = pd.read_csv(url_val).drop(['Unnamed: 0'],axis=1)

print("Data loaded successfully!")

data_interaction_train["abstract"] = data_interaction_train["abstract"].str.lower()
data_interaction_test["abstract"] = data_interaction_test["abstract"].str.lower()
data_interaction_val["abstract"] = data_interaction_val["abstract"].str.lower()

pos2item = {i:v for i, v in enumerate(data_tracks_tags_lyrics.track_id.sort_values().unique())}
item2pos = {v:i for i, v in enumerate(data_tracks_tags_lyrics.track_id.sort_values().unique())}

id2user = {i:v for i, v in enumerate(data_interaction.user_id.sort_values().unique())}
user2id = {v:i for i, v in enumerate(data_interaction.user_id.sort_values().unique())}


"""# Model Training"""

BERT_MODEL_NAME = 'bert-base-uncased'
#BERT_MODEL_NAME = 'prajjwal1/bert-tiny'
BERT_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
item_text_embeddings_size = 128
user_embeddings_size = 100
item_embeddings_size = 10#128
text_col = 'tags'

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME) 

token_dict = dict()
attention_dict = dict()

for row in data_tracks_tags_lyrics.iterrows():
  text = row[1][text_col]
  track_id = row[1]['track_id']
  inputs = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            padding='max_length',
            max_length = item_text_embeddings_size,
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True)
  token = inputs["input_ids"]
  mask = inputs["attention_mask"]
  token_dict.update({track_id: token})
  attention_dict.update({track_id : mask})

dataset_train= LFM2bDataset(data_interaction_train,item_text_embeddings_size, text_col,  item2pos, user2id, token_dict, attention_dict)
dataset_val= LFM2bDataset(data_interaction_val,item_text_embeddings_size, text_col,  item2pos, user2id,token_dict, attention_dict)
dataset_test= LFM2bDataset(data_interaction_test,item_text_embeddings_size, text_col,  item2pos, user2id, token_dict, attention_dict)

# dataset_train= LFM2bDatasetMF(data_interaction_train,tokenizer,item_text_embeddings_size, text_col,  item2pos, user2id)
# dataset_val= LFM2bDatasetMF(data_interaction_val,tokenizer,item_text_embeddings_size, text_col,  item2pos, user2id)
# dataset_test= LFM2bDatasetMF(data_interaction_test,tokenizer,item_text_embeddings_size, text_col,  item2pos, user2id)

# dataset_train= LFM2bDatasetMulitpleText(data_interaction_train,tokenizer,item_text_embeddings_size,  item2pos, user2id)
# dataset_val= LFM2bDatasetMulitpleText(data_interaction_val,tokenizer,item_text_embeddings_size,  item2pos, user2id)

num_users = len(data_interaction['user_id'].drop_duplicates())
num_items = len(pos2item)

modelBert = EasyBERTModel(item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items, BERT_MODEL_NAME)

model_name = type(modelBert).__name__
data = "_binary"
freeze = ""#"_freeze"
optim = "_adamw"
optim = "_rmsprop"
text = text_col

model_dir = f"./results/{text}{model_name}{data}{freeze}{optim}_{BERT_MODEL_NAME}/"
import os
os.makedirs(model_dir, exist_ok=True)
print(f"Save results in: {model_dir}")

from transformers import get_linear_schedule_with_warmup
#gc.collect()
#torch.cuda.empty_cache()

modelBert = modelBert.to(device)

print(f"Start training on device {device}")

# freeze bert model parameter
if freeze == "_freeze":
  for param in modelBert.bert.parameters():
    param.requires_grad = False  
  # for param in modelBert.bert1.parameters():
  #   param.requires_grad = False   
  # for param in modelBert.bert2.parameters():
  #   param.requires_grad = False   

num_epochs=15
batch_size=8

dataloader_train=DataLoader(dataset=dataset_train,batch_size=batch_size, num_workers=4, shuffle=True)
dataloader_val=DataLoader(dataset=dataset_val,batch_size=batch_size, num_workers=4, shuffle=True)
dataloader_test=DataLoader(dataset=dataset_test,batch_size=batch_size, num_workers=4, shuffle=True)

criterion = nn.BCELoss()
lr = 2e-5
#lr = 1e-3
num_total_steps = len(dataset_train) * num_epochs
num_warmup_steps = 0
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  
#optimizer =  torch.optim.RMSprop(modelBert.parameters(), lr= lr, alpha=0.9)
optimizer = torch.optim.AdamW(modelBert.parameters(), lr=lr, weight_decay=0.1)

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=num_warmup_steps,
  num_training_steps=num_total_steps
)

loss_train = []
loss_val = []
ndcg_val = []

best_ndcg = 10
cnt = 0

for e in range(num_epochs):

  losses_train = []

  modelBert.train()

  for data_train in dataloader_train:
      #gc.collect()
      #torch.cuda.empty_cache()
      modelBert.zero_grad()
      #items positions
      input_id = data_train['track_id'].to(device)
      # items descriptions
      curr_items_batch = data_train['input_ids_lyrics'].to(device)
      curr_attentions_batch = data_train['attention_mask_lyrics'].to(device)

      # additional items descriptions
      # curr_items_batch1 = data_train['input_ids_tags'].to(device)
      # curr_attentions_batch1 = data_train['attention_mask_tags'].to(device)
      
      # users ids
      user_id =data_train['user_id'].to(device)

      # model targets
      targets = data_train['target'].reshape(-1,1).to(device)

      # backward propagation
      outputs = modelBert(user_id, input_id, curr_items_batch, curr_attentions_batch)
      loss = criterion(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(modelBert.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      
      # evaluate current loss function value
      losses_train.append(loss.item())

  # compute loss on validation set
  modelBert.eval()

  losses_val = []

  res = pd.DataFrame(columns = ['user_id', 'track_id', 'count'])
  for data_val in dataloader_val:
    gc.collect()
    torch.cuda.empty_cache()
    #items positions
    input_id = data_val['track_id'].to(device)
    # items descriptions
    curr_items_batch = data_val['input_ids_lyrics'].to(device)
    curr_attentions_batch = data_val['attention_mask_lyrics'].to(device)

    # additional items descriptions
    # curr_items_batch1 = data_val['input_ids_tags'].to(device)
    # curr_attentions_batch1 = data_val['attention_mask_tags'].to(device)
    
    # users ids
    user_id =data_val['user_id'].to(device)

    # model targets
    targets_val = data_val['target'].reshape(-1,1).to(device)

    
    with torch.no_grad():        
      outputs_val = modelBert(user_id, input_id, curr_items_batch, curr_attentions_batch)

    loss = criterion(outputs_val, targets_val)
    losses_val.append(loss.item())

    for i in range(outputs_val.shape[0]):
      res = pd.concat([res, pd.DataFrame({'user_id': id2user[user_id[i].item()], 
                                          'track_id': pos2item[input_id[i].item()], 
                                          'count':  outputs_val[i].item()} , index=[0])])

  # compute ndcg for each user
  ndcg = []
  for user, df in res.groupby('user_id'):
    df = df.drop_duplicates(subset='track_id').sort_values('count', ascending=False).head(10)
    y_true_sorted = data_interaction_val.loc[data_interaction_val['user_id'] == user].sort_values('count', ascending=False).drop_duplicates()
    rel_true = pd.merge(df[['track_id']], y_true_sorted[['track_id', 'count']].drop_duplicates(), 'left').fillna(0)['count'].values
    rel_true[rel_true > 0] = 1
    ndcg.append(get_ndcg(rel_true, 10))

  ndcg_e = np.mean(losses_val)#np.mean(ndcg)

  # save best model based on ndcg
  if ndcg_e < best_ndcg: 
    best_ndcg = ndcg_e
    print(f"\nSaving best model for epoch: {e}\n")
    torch.save(modelBert.state_dict(), model_dir + 'best_model.pth')
    cnt = 0
  else:
    cnt = cnt+1
  if cnt > 4:
    break

  average_los_val = np.mean(losses_val)
  loss_val.append(average_los_val.item())    
  ndcg_val.append(ndcg_e)

  # evaluate average cost per epoch
  average_loss_train = np.mean(losses_train)
  loss_train.append(average_loss_train.item())
  print(f"Epoch {e} | Average loss per epoch: Train: {average_loss_train.item()} , Val: {average_los_val.item()}, NDCG: {ndcg_e}")

"""## **BERT**: Get predictions"""

res = pd.DataFrame(columns = ['user_id', 'track_id', 'count'])

model = EasyBERTModel(item_embeddings_size, item_text_embeddings_size, user_embeddings_size, num_users, num_items, BERT_MODEL_NAME)

model.load_state_dict(torch.load(model_dir + 'best_model.pth'))
model.to(device)
model.eval()

print("Start predicting")

for data in dataloader_test:
      gc.collect()
      torch.cuda.empty_cache()
      #items positions
      input_id = data['track_id'].to(device)
      # items descriptions
      curr_items_batch = data['input_ids_lyrics'].to(device)
      curr_attentions_batch = data['attention_mask_lyrics'].to(device)

      # additional items descriptions
      # curr_items_batch1 = data['input_ids_tags'].to(device)
      # curr_attentions_batch1 = data['attention_mask_tags'].to(device)
      
      # users ids
      user_id =data['user_id'].to(device)

      with torch.no_grad():
        outputs = model(user_id, input_id, curr_items_batch, curr_attentions_batch)

      # save prediction for each user
      for i in range(outputs.shape[0]):

        res = pd.concat([res, pd.DataFrame({'user_id': id2user[user_id[i].item()], 
                                          'track_id': pos2item[input_id[i].item()], 
                                          'count':  outputs[i].item()} , index=[0])])

topn=100
results = []
results_df = pd.DataFrame(columns = ['user_id', 'track_id', 'count'])

for user, user_prediction in res.groupby('user_id'):
    results_df = pd.concat([results_df, user_prediction.sort_values('count', ascending=False).head(topn)])

results_df.to_csv(model_dir + 'predictions.csv', encoding = 'utf-8-sig')

topn=100

f1_scores = []
recall_scores = []
ndcg_scores = []

for user, df in results_df.groupby('user_id'):
  df = df.drop_duplicates(subset='track_id').head(topn)

  y_true_sorted = data_interaction_test.loc[data_interaction_test['user_id'] == user].sort_values('count', ascending=False).drop_duplicates()
  y_ndcg = pd.merge(df[['track_id']], y_true_sorted[['track_id', 'count']].drop_duplicates(), 'left').fillna(0)['count'].values
  y_true_df = pd.merge(data_tracks_tags_lyrics[['track_id']].drop_duplicates(), y_true_sorted[['track_id', 'count']].drop_duplicates(), 'left').fillna(0)
  y_ndcg[y_ndcg > 0] = 1
  y_true_df.loc[y_true_df['count'] > 0, 'count'] = 1
  y_true = y_true_df['count'].values

  y_pred_df = pd.merge(data_tracks_tags_lyrics[['track_id']].drop_duplicates(), df[['track_id', 'count']], 'left').fillna(0)
  y_pred_df.loc[y_pred_df['count'] > 0, 'count'] = 1
  y_pred = y_pred_df['count'].values

  if y_true.sum() >= 1:
    f1_scores.append(f1_score(y_true, y_pred))
    recall_scores.append(recall_score(y_true, y_pred))
    ndcg_scores.append(get_ndcg(y_ndcg, topn))

coverage = get_coverage(results_df, data_tracks_tags_lyrics)

print("F1 Score: ", np.mean(f1_scores))
print("NDCG Score: ", np.mean(ndcg_scores))
print("Coverage: , ", coverage)