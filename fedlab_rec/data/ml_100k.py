import pandas as pd
import os
from .utils import create_negative_samples, filter_df_by_count, encode_column, df_to_dataloader
from .dataset import FedDataset

class ML100kDataSet(FedDataset):
    def __init__(self, root_dir, pos_threshold=5, neg_train_rate=4, neg_test_rate=100, batch_size=256):
        if os.path.exists(os.path.join(root_dir, 'train.data')) and\
            os.path.exists(os.path.join(root_dir, 'test.data')):
                self.train_df = pd.read_csv(os.path.join(root_dir, 'train.data'))
                self.test_df = pd.read_csv(os.path.join(root_dir, 'test.data'))
        else:
            self.train_df, self.test_df = self.generate_train_test(root_dir, 
                                                                   pos_threshold=pos_threshold, 
                                                                   neg_train_rate=neg_train_rate, 
                                                                   neg_test_rate=neg_test_rate)
        self.build_dataloader(batch_size)
            
    def generate_train_test(self, root_dir, pos_threshold=5, neg_train_rate=4, neg_test_rate=100):
        # 读入dataframe
        ratings_path = os.path.join(root_dir, 'u.data')
        ratings_df = pd.read_csv(ratings_path, sep='\t', header=None, 
                                 names=['user_id', 'item_id', 'rating', 'timestamp'])
        # 过滤交互次数少于pos_threshold的用户
        ratings_df = filter_df_by_count(ratings_df, count_col='user_id', threshold=pos_threshold)
        user_id_encoder = encode_column(ratings_df, 'user_id')
        item_id_encoder = encode_column(ratings_df, 'item_id')
        # 按照时间戳排序
        sorted_df = ratings_df.sort_values(by='timestamp', ascending=True)
        sorted_df.reset_index(drop=True)
        # 负采样生成训练集和测试集
        train_df, test_df = create_negative_samples(sorted_df, 
                                             col_user='user_id', 
                                             col_item='item_id',
                                             col_rating='rating',
                                             negative_train_rate=neg_train_rate,
                                             negative_test_rate=neg_test_rate)
        # 保存
        for col in train_df.columns:
            train_df[col] = train_df[col].astype(int)
        for col in test_df.columns:
            test_df[col] = test_df[col].astype(int)
        train_df.to_csv(os.path.join(root_dir, 'train.data'), index=False)
        test_df.to_csv(os.path.join(root_dir, 'test.data'), index=False)
        return train_df, test_df
    
    def build_dataloader(self, batch_size=256):
        self.id_to_train_dl, self.id_to_test_dl = {}, {}
        train_df_grouped, test_df_grouped =\
            self.train_df.groupby('user_id'), self.test_df.groupby('user_id')
        for user_id, user_df in train_df_grouped:
            self.id_to_train_dl[user_id] = df_to_dataloader(user_df, batch_size)
        for user_id, user_df in test_df_grouped:
            self.id_to_test_dl[user_id] = df_to_dataloader(user_df, batch_size)
    
    def get_dataloader(self, idx, mode='train'):
        if mode=='train':
            return self.id_to_train_dl[idx]
        else:
            return self.id_to_test_dl[idx]
    
    @property
    def users_num(self):
        return self.test_df['user_id'].max()+1
    
    @property
    def items_num(self):
        return self.test_df['item_id'].max()+1
    
        
    
        
