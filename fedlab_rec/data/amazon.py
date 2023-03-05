import pandas as pd
import os
from .utils import create_negative_samples, filter_df_by_count, encode_column, df_to_dataloader
from .dataset import FedDataset
from tqdm import tqdm
import gzip
import json
import html
import re
import numpy as np
from .extract_embedding import ExtractImageEmbedding, ExtractTextEmbedding

class AmazonDataSet(FedDataset):
    def __init__(self, root_dir, dataset_name, pos_threshold=5, neg_train_rate=4, neg_test_rate=100, batch_size=256):
        self.dataset_name = dataset_name
        if os.path.exists(os.path.join(root_dir, dataset_name, 'train.data')) and\
            os.path.exists(os.path.join(root_dir, dataset_name, 'test.data')) and\
            os.path.exists(os.path.join(root_dir, dataset_name, 'item_image_embedding')) and\
            os.path.exists(os.path.join(root_dir, dataset_name, 'item_text_embedding')):
                self.train_df = pd.read_csv(os.path.join(root_dir, dataset_name, 'train.data'))
                self.test_df = pd.read_csv(os.path.join(root_dir, dataset_name, 'test.data'))
                self.item_image_embedding = np.load(os.path.join(root_dir, dataset_name, 'item_image_embedding.npy'))
                self.item_text_embedding = np.load(os.path.join(root_dir, dataset_name, 'item_text_embedding.npy'))
        else:
            self.train_df, self.test_df, self.item_embedding = self.generate_train_test(
                                                                os.path.join(root_dir, dataset_name), 
                                                                pos_threshold=pos_threshold, 
                                                                neg_train_rate=neg_train_rate, 
                                                                neg_test_rate=neg_test_rate)
        self.build_dataloader(batch_size)
            
    def generate_train_test(self, root_dir, pos_threshold=5, neg_train_rate=4, neg_test_rate=100):
        meta_path = os.path.join(root_dir, f'meta_{self.dataset_name}.json.gz')
        ratings_path = os.path.join(root_dir, f'{self.dataset_name}.csv')
        # 读取rating文件和meta文件
        ratings_df = self.load_ratings(ratings_path)
        item_info = self.load_meta(meta_path) # 字典, key为item_id, value为对应的[文本，图片url]
        # 去除不在item_info的item id
        ratings_df = ratings_df[ratings_df['item_id'].isin(item_info.keys())]
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
        # 对item_info进行编码
        item_id_encoder_dict = dict(zip(item_id_encoder.classes_, item_id_encoder.transform(item_id_encoder.classes_)))
        item_info = {item_id_encoder_dict.get(k): v for k, v in item_info.items() if k in item_id_encoder_dict}
        # 使用Bert获取item的文本特征，使用resnet获取图片特征，并保存在npy中
        item_text = {k:v[0] for k, v in item_info.items()}
        item_image = {k:v[1] for k, v in item_info.items()}
        text_embedding = ExtractTextEmbedding(item_text).get_embedding()
        image_embedding = ExtractImageEmbedding(download_path=root_dir, image_dict=item_image)
        return train_df, test_df
    
    # 读取rating数据，返回一个dataframe
    # 列名为"user_id"、"item_id"、"rating"、"timestamp"
    def load_ratings(self, file_path):
        ratings = []
        with open(file_path, 'r') as fp:
            for line in tqdm(fp, desc='Load ratings'):
                try:
                    item, user, rating, time = line.strip().split(',')
                    ratings.append((str(user), str(item), float(rating), int(time)))
                except ValueError:
                    print(line)
        ratings_df = pd.DataFrame(ratings, columns=["user_id", "item_id", "rating", "timestamp"])
        return ratings_df
    
    def load_meta(self, file_path):
        item_info = {}
        with gzip.open(file_path, 'r') as fp:
            for line in tqdm(fp, desc='Load metas'):
                data = json.loads(line)
                item = data['asin']
                text, image_url = '', ''
                for meta_key in ['title', 'category', 'brand']:
                    if meta_key in data:
                        meta_value = self.clean_text(data[meta_key])
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
    
        
    
        
