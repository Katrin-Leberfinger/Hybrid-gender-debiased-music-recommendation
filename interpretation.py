import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertConfig, BertPreTrainedModel, BertModel, BertForSequenceClassification
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import gc          

"""## Read Data: Lyrics """

dir = "preprocessed_data_binary/"
#model_dir = "results/tagsMatrixFactorizationModel_binary_adamw_prajjwal1/bert-tiny/"

model_dir = "results/itemknn/bert-tiny/"

data_interaction = pd.read_csv(dir + "data_interaction.csv").drop(['Unnamed: 0'],axis=1)
data_tracks_tags_lyrics = pd.read_csv(dir + "data_tracks_tags_lyrics.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_test = pd.read_csv(dir + "data_interaction_test.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_train = pd.read_csv(dir + "data_interaction_train.csv").drop(['Unnamed: 0'],axis=1)
data_interaction_val = pd.read_csv(dir + "data_interaction_val.csv").drop(['Unnamed: 0'],axis=1)


data_predictions = pd.read_csv(model_dir + "predictions.csv").drop(['Unnamed: 0'],axis=1)
topn=100
results = []
results_df = pd.DataFrame(columns = ['user_id', 'track_id', 'count'])
for user, user_prediction in data_predictions.groupby('user_id'):
    results_df = pd.concat([results_df, user_prediction.sort_values('count', ascending=False).head(topn)])
data_interaction_new = results_df.merge(data_tracks_tags_lyrics, on = 'track_id')

url_info = "https://drive.google.com/u/0/uc?id=1LewaXgy5hu8wsSHGnqL6jd_D3MFPSh3U&export=download&confirm=t&uuid=2bef7bee-82dc-4806-ba64-3e372d1f5bd3"
url_user = "https://drive.google.com/u/0/uc?id=1cdhJ5-gHZnFp0y5G1Qiv57eNWCQZ-m2c&export=download"

data_all_info = pd.read_csv(url_info, sep="\t").drop(['Unnamed: 0'],axis=1)
data_all_info = data_all_info[['track_artist', 'gender_artist', 'track_id']]

data_user = pd.read_csv(url_user, sep="\t").drop(['Unnamed: 0'],axis=1)
data_user.columns = ['user_id', 'gender_user']

data_classification = data_all_info
replace_dict1 = {'m' : 0, 'f' : 1}
replace_dict2 = {'male' : 0, 'female' : 1}
data_classification['gender_artist'] = data_classification['gender_artist'].replace(replace_dict2)
data_classification = data_classification.merge(data_tracks_tags_lyrics[['track_id', 'tags', 'abstract']], on = 'track_id')
data_classification

df_tmp = pd.merge(data_interaction, data_user, 'inner')
df_all = pd.merge(df_tmp, data_all_info, on = 'track_id').drop_duplicates()

replace_dict1 = {'m' : 0, 'f' : 1}
replace_dict2 = {'male' : 0, 'female' : 1}
df_all['gender_user'] = df_all['gender_user'].replace(replace_dict1)
df_all['gender_artist'] = df_all['gender_artist'].replace(replace_dict2)

df_interaction = df_all[df_all['count'] == 1]

df_tmp = pd.merge(data_interaction_train, data_user, 'inner')
df_all = pd.merge(df_tmp, data_all_info, on = 'track_id').drop_duplicates()

replace_dict1 = {'m' : 0, 'f' : 1}
replace_dict2 = {'male' : 0, 'female' : 1}
df_all['gender_user'] = df_all['gender_user'].replace(replace_dict1)
df_all['gender_artist'] = df_all['gender_artist'].replace(replace_dict2)

df_train = df_all[df_all['count'] == 1]

df_tmp = pd.merge(data_interaction_test, data_user, 'inner')
df_all = pd.merge(df_tmp, data_all_info, on = 'track_id').drop_duplicates()

df_all['gender_user'] = df_all['gender_user'].replace(replace_dict1)
df_all['gender_artist'] = df_all['gender_artist'].replace(replace_dict2)

df_test = df_all[df_all['count'] == 1]

df_tmp = pd.merge(results_df, data_user, 'inner')
df_all = pd.merge(df_tmp, data_all_info, on = 'track_id').drop_duplicates()

df_all['gender_user'] = df_all['gender_user'].replace(replace_dict1)
df_all['gender_artist'] = df_all['gender_artist'].replace(replace_dict2)

df_rec = df_all

def get_coverage(df_rec, data_tracks_tags_lyrics):
  return len(df_rec['track_id'].drop_duplicates()) / len(data_tracks_tags_lyrics['track_id'].drop_duplicates())

def get_ndcg(rel_true, k):

  rel_opt = np.zeros(len(rel_true))
  rel_opt[:k] = 1
  #rel_opt = rel_opt[:k]
  #rel_true = rel_true[:k]

  def _dcg(rel):
    i = np.arange(1, len(rel)+ 1)
    denom = np.log2(i + 1)
    dcg = np.sum(rel / denom)
    return dcg

  return _dcg(rel_true) / _dcg(rel_opt)

topn=100

f1_scores = []
recall_scores = []
ndcg_scores = []

for user, df in data_predictions.groupby('user_id'):
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
coverage_test = get_coverage(data_interaction_test[data_interaction_test['count'] == 1], data_tracks_tags_lyrics)

print("F1 Score: ", np.mean(f1_scores))
print("NDCG Score: ", np.mean(ndcg_scores))
print("Coverage: , ", coverage)
print("Coverage Test: , ", coverage_test)
print("===========================")

"""Interpretation"""

print(f"In the history data, {round(df_interaction.gender_artist.sum() / len(df_interaction) * 100, 2)} % female items were consumed.")
print(f"In the train data, {round(df_train.gender_artist.sum() / len(df_train) * 100, 2)} % female items were consumed.")
print(f"In the test data, {round(df_test.gender_artist.sum() / len(df_test) * 100, 2)} % female items were consumed.")
print(f"In the recommendation data, {round(df_rec.gender_artist.sum() / len(df_rec) * 100, 2)} % female items were recommended.")
print("-------------------------------")

prop_female_rec = df_rec.gender_artist.sum() / len(df_rec)
prop_female_history = df_train.gender_artist.sum() / len(df_train)
delta = (prop_female_rec - prop_female_history) / prop_female_history

if delta > 0:
  print(f'The value of delta in recommendations is {round(delta, 4)} and therefore more female tracks are recommended to user.')
else:
  print(f'The value of delta in recommendations is {round(delta, 4)} and therefore more male tracks are recommended to user.')

prop_female_rec = df_test.gender_artist.sum() / len(df_test)
prop_female_history = df_interaction.gender_artist.sum() / len(df_interaction)
delta = (prop_female_rec - prop_female_history) / prop_female_history

if delta > 0:
  print(f'The value of delta in test data is {round(delta, 4)} and therefore more female tracks are recommended to user.')
else:
  print(f'The value of delta in test data is {round(delta, 4)} and therefore more male tracks are recommended to user.')
