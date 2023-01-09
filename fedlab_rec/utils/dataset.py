from abc import abstractmethod, ABC
import pandas as pd
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def encode_column(df, col_name):
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return encoder

class DataFrameDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __getitem__(self, index):
        return {col: self.df[col].iloc[index] for col in self.df.columns}

    def __len__(self):
        return len(self.df)


    


class RecDataset(FedDataset):
    def __init__(self, df, client_ids_list=None, 
                 batch_size=64, split_ratio=(0.8, 0.1), 
                 user_col='user_id'):
        self.num_clients = num_clients
        self.user_col = user_col
        uid = df[user_col].unique()
        self.client_ids_list = client_ids_list
        self.batch_size = batch_size
        if client_ids_list is None:
            num_client_ids = len(uid)//(num_clients)
            random.shuffle(uid)
            self.client_ids_list = []
            for i in range(self.num_clients):
                self.client_ids_list.append(uid[i*num_client_ids:(i+1)*num_client_ids])
        self.split_ratio = split_ratio
        self.generate_dataloader(df)
    
    def generate_dataloader(self, df):
        self.client_dataloder_list = []
        for i in range(self.num_clients):
            client_df = df[df[self.user_col].isin(self.client_ids_list[i])].reset_index(drop=True)
            client_dict = df_to_dict(client_df)
            client_dataset = TorchDataset(client_dict)
            train_length = int(len(client_dataset) * self.split_ratio[0])
            val_length = int(len(client_dataset) * self.split_ratio[1])
            test_length = len(client_dataset) - train_length - val_length
            train_dataset, val_dataset, test_dataset = random_split(client_dataset,
                                                                    (train_length, val_length, test_length))
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            self.client_dataloder_list.append((train_dataloader, val_dataloader, test_dataloader))

    def get_dataloader(self, idx, mode='train'):
            if mode=="train":
                return self.client_dataloder_list[idx][0]
            elif mode=="valid":
                return self.client_dataloder_list[idx][1]
            else:
                return self.client_dataloder_list[idx][2]

if __name__ == "__main__":
    # rating_data = pd.read_csv("data/ml-1m/ratings.dat", sep='::', header=None)
    # rating_data = rating_data.sample(frac=0.1).reset_index(drop=True)
    # rating_data.columns = ['user_id', 'item_id', 'rating', 'time']
    # user_pos = defaultdict(set)
    # item_set = set(rating_data['item_id'].unique())
    # for index, row in rating_data.iterrows():
    #     user_id, item_id = row['user_id'], row['item_id']
    #     user_pos[user_id].add(item_id)
    # new_df = pd.DataFrame(columns=['user_id', 'item_id', 'label'])
    # i = 0
    # # neg_sample
    # for user, pos in tqdm(user_pos.items()):
    #     neg = item_set - pos
    #     pos, neg = list(pos), list(neg)
    #     neg = np.random.choice(neg, size=len(pos), replace=False)
    #     for iid in pos:
    #         new_df.loc[i] = [user, iid, 1]
    #         i += 1
    #     for iid in neg:
    #         new_df.loc[i] = [user, iid, 0]
    #         i += 1
    # new_df.reset_index(drop=True).to_feather("data/ml-1m/ratings.feather")
    rating_data = pd.read_feather("data/ml-1m/ratings.feather")
    dataset = RecDataset(rating_data, num_clients=20)

            
            


        