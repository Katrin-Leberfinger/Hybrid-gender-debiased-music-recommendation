import torch
from torch.utils.data import Dataset

class LFM2bDatasetMF(Dataset):
    def __init__(self, data_all,tokenizer,max_length, text_col, item2pos, user2id):
        super().__init__()
        self.data_all = data_all
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.item2pos = item2pos
        self.user2id = user2id
        
    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, index):
        
        text1 = self.data_all.iloc[index][self.text_col]
        user_id = self.user2id[self.data_all.iloc[index]['user_id']]
        track_id = self.item2pos[self.data_all.iloc[index]['track_id']]
        target = self.data_all.iloc[index]['count']

        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'track_id': torch.tensor(track_id, dtype=torch.long),
            'target': torch.FloatTensor([target])
            }


class LFM2bDataset(Dataset):
    def __init__(self, data_all,max_length, text_col, item2pos, user2id, token_dict, attention_dict):
        super().__init__()
        self.data_all = data_all
        self.max_length = max_length
        self.text_col = text_col
        self.item2pos = item2pos
        self.user2id = user2id
        self.token_dict  = token_dict
        self.attention_dict = attention_dict
        
    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, index):
        
        text1 = self.data_all.iloc[index][self.text_col]
        user_id = self.user2id[ self.data_all.iloc[index]['user_id']]
        track_id = self.item2pos[ self.data_all.iloc[index]['track_id']]
        target = self.data_all.iloc[index]['count']

        ids = self.token_dict[self.data_all.iloc[index]['track_id']]
        mask = self.attention_dict[self.data_all.iloc[index]['track_id']]

        return {
            'input_ids_lyrics': ids.flatten(),
            'attention_mask_lyrics': mask.flatten(),
            'user_id': torch.tensor(user_id, dtype=torch.long),
             'track_id': torch.tensor(track_id, dtype=torch.long),
            'target': torch.FloatTensor([target])
            }

class LFM2bDatasetMulitpleText(Dataset):
    def __init__(self, data_all,max_length, item2pos, user2id, token_dict, attention_dict):
        super().__init__()
        self.data_all = data_all
        self.max_length = max_length
        self.item2pos = item2pos
        self.user2id = user2id
        self.token_dict  = token_dict
        self.attention_dict = attention_dict
        
    def __len__(self):
        return len(self.data_all)
    
    def __getitem__(self, index):
        
        text1 = self.data_all.iloc[index]['lyrics_cleaned']
        text2 = self.data_all.iloc[index]['tags']
        text3 = self.data_all.iloc[index]['abstract']
        user_id = self.user2id[ self.data_all.iloc[index]['user_id']]
        track_id = self.item2pos[ self.data_all.iloc[index]['track_id']]
        
        inputs1 = tokenizer.encode_plus(
            text1, 
            add_special_tokens=True,
            padding='max_length',
            max_length = self.max_length,
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True
            
        )
        ids1 = inputs1["input_ids"]
        mask1 = inputs1["attention_mask"]

        inputs2 = tokenizer.encode_plus(
            text2, 
            add_special_tokens=True,
            padding='max_length',
            max_length = self.max_length,
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True
            
        )
        ids2 = inputs2["input_ids"]
        mask2 = inputs2["attention_mask"]

        inputs3 = tokenizer.encode_plus(
            text3, 
            add_special_tokens=True,
            padding='max_length',
            max_length = self.max_length,
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True
            
        )
        ids3 = inputs1["input_ids"]
        mask3 = inputs1["attention_mask"]

        return {
            'input_ids_lyrics': torch.tensor(ids1, dtype=torch.long),
            'attention_mask_lyrics': torch.tensor(mask1, dtype=torch.long),
            'input_ids_tags': torch.tensor(ids2, dtype=torch.long),
            'attention_mask_tags': torch.tensor(mask2, dtype=torch.long),
            'input_ids_abstract': torch.tensor(ids3, dtype=torch.long),
            'attention_mask_abstract': torch.tensor(mask3, dtype=torch.long),
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'track_id': torch.tensor(track_id, dtype=torch.long),
            'target': torch.tensor(self.data_all.iloc[index]['count'], dtype=torch.float)
            }
