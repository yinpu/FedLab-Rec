import pandas as pd
import os
from .utils import create_negative_samples, filter_df_by_count, encode_column, df_to_dataloader
from .dataset import FedDataset
from tqdm import tqdm
import gzip
import json
import html
import re

class AmazonDataSet(FedDataset):
    def __init__(self, root_dir, dataset_name, pos_threshold=5, neg_train_rate=4, neg_test_rate=100, batch_size=256):
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
    
    def load_ratings(self, file_path, sample_ratio=1):
        users, items, inters = set(), set(), set()
        with open(file_path, 'r') as fp:
            for line in tqdm(fp, desc='Load ratings'):
                try:
                    item, user, rating, time = line.strip().split(',')
                    users.add(user)
                    items.add(item)
                    inters.add((user, item, float(rating), int(time)))
                except ValueError:
                    print(line)
        return users, items, inters
    
    def load_meta(file_path):
        item_info = {}
        with gzip.open(file_path, 'r') as fp:
            for line in tqdm(fp, desc='Load metas'):
                data = json.loads(line)
                item = data['asin']
                text, image_url = '', ''
                for meta_key in ['title', 'category', 'brand']:
                    if meta_key in data:
                        meta_value = clean_text(data[meta_key])
                        text += meta_value + ' '
                if len(data['imageURLHighRes'])>0:
                    image_url = data['imageURLHighRes'][0].strip()
                if len(text.strip())!=0 and len(image_url)!=0:
                    item_info[item] = [text, image_url]
            print(len(item_info))
        return item_info
    
    def clean_text(self, raw_text):
        if isinstance(raw_text, list):
            cleaned_text = ' '.join(raw_text)
        elif isinstance(raw_text, dict):
            cleaned_text = str(raw_text)
        else:
            cleaned_text = raw_text
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
        index = -1
        while -index < len(cleaned_text) and cleaned_text[index] == '.':
            index -= 1
        index += 1
        if index == 0:
            cleaned_text = cleaned_text + '.'
        else:
            cleaned_text = cleaned_text[:index] + '.'
        if len(cleaned_text) >= 2000:
            cleaned_text = ''
        return cleaned_text
    
    
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
    
        
    
        
