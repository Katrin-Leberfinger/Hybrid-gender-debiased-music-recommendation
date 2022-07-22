import numpy as np
import pandas as pd


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