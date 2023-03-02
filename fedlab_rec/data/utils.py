from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import random
import pandas as pd
from tqdm import tqdm

class DataFrameDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __getitem__(self, index):
        return {col: self.df[col].iloc[index] for col in self.df.columns}

    def __len__(self):
        return len(self.df)

def encode_column(df, col_name):
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return encoder

def df_to_dataloader(df, batch_size):
    df = df.reset_index(drop=True)
    dataset = DataFrameDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def filter_df_by_count(df, count_col, threshold):
    counts = df[count_col].value_counts()
    filtered_df = df[df[count_col].isin(counts[counts >= threshold].index)]
    return filtered_df

def create_negative_samples(df, col_user, col_item, col_rating, 
                            negative_train_rate=4, negative_test_rate=100):
    print("create_negative_samples")
    # Create a list of negative samples for each user
    train_pos_samples = {}
    train_neg_samples = {}
    test_pos_samples = {}
    test_neg_samples = {}
    for user_id, group in tqdm(df.groupby(col_user)):
        # Get the item IDs that the user has interacted with
        interacted_item_ids = group[col_item].tolist()
        # Select negative_rate times as many negative samples as positive samples
        train_neg_count = (len(interacted_item_ids)-1) * negative_train_rate
        test_neg_count = negative_test_rate
        # Select negative samples from the list of all item IDs
        all_item_ids = df[col_item].unique()
        non_interacted_item_ids = list(set(all_item_ids) - set(interacted_item_ids))
        negative_samples = random.choices(non_interacted_item_ids,
                                         k=train_neg_count+test_neg_count)
        train_pos_samples[user_id] = interacted_item_ids[:-1]
        train_neg_samples[user_id] = negative_samples[:train_neg_count]
        test_pos_samples[user_id] = interacted_item_ids[-1:]
        test_neg_samples[user_id] = negative_samples[train_neg_count:]

    # Convert the negative samples into a DataFrame
    train_df = pd.DataFrame({col_user: [], col_item: [], col_rating: []})
    test_df = pd.DataFrame({col_user: [], col_item: [], col_rating: []})
    for user_id, item_ids in train_pos_samples.items():
        train_df = train_df.append(pd.DataFrame({col_user: [user_id] * len(item_ids), 
                                                 col_item: item_ids, 
                                                 col_rating: [1] * len(item_ids)}))
    for user_id, item_ids in train_neg_samples.items():
        train_df = train_df.append(pd.DataFrame({col_user: [user_id] * len(item_ids), 
                                                 col_item: item_ids, 
                                                 col_rating: [0] * len(item_ids)}))
    for user_id, item_ids in test_pos_samples.items():
        test_df = test_df.append(pd.DataFrame({col_user: [user_id] * len(item_ids), 
                                               col_item: item_ids, 
                                               col_rating: [1] * len(item_ids)}))
    for user_id, item_ids in test_neg_samples.items():
        test_df = test_df.append(pd.DataFrame({col_user: [user_id] * len(item_ids), 
                                               col_item: item_ids, 
                                               col_rating: [0] * len(item_ids)}))

    return train_df, test_df


