import os
import concurrent.futures
import pickle
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np


# 定义一个dataset类来读取图像
class ImageDataset(torch.utils.data.Dataset):
  def __init__(self, image_dir, image_names, transform=None):
    self.image_dir = image_dir # 图片所在的文件夹
    self.image_names = image_names # 图片的名字列表
    self.transform = transform # 可选的转换函数

  def __len__(self):
    return len(self.image_names) # 返回数据集的长度

  def __getitem__(self, index):
    image_path = self.image_dir + "/" + self.image_names[index] # 根据索引拼接图片路径
    image = Image.open(image_path).convert("RGB") # 打开图片
    if self.transform: # 如果有转换函数，则对图片进行转换
      image = self.transform(image)
    return image # 返回图片张量

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_dict):
        self.text_dict = text_dict

    def __len__(self):
        return len(self.text_dict)

    def __getitem__(self, idx):
        text = self.text_dict[idx]
        return text

class ExtractImageEmbedding:
    def __init__(self, download_path, image_dict):
        self.download_folder =f"{download_path}/images"
        if not os.path.exists(self.download_folder):
            os.mkdir(self.download_folder)
        self.image_dict = image_dict
    
    def download_image(self, key, url):
        os.system(f"curl -s -o {self.download_folder}/{key}.jpg {url}")
        
    def load_pretrain_model(self):
        self.device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = models.resnet50(pretrained=True) # 加载resnet50预训练模型
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)    
        self.model.eval() # 设置模型为评估模式
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])
    
    def get_embedding(self):
        self.load_pretrain_model()
        print("download imgs")
        # 并行下载图片到imgs文件夹
        max_threads = 16
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.download_image, key, url) for key, url in self.image_dict.items()]
            for future in concurrent.futures.as_completed(futures):
                if future.exception() is not None:
                    print(f"Failed to download image: {future.exception()}")
        print("extract image feature")
        image_names = [str(i) + ".jpg"for i in range(len(self.image_dict))]
        dataset = ImageDataset(self.download_folder, image_names, self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 shuffle=False,
                                                 batch_size=64, 
                                                 num_workers=16)
        all_features = []
        for batch in dataloader:
          batch = batch.to(self.device)
          with torch.no_grad():
              features = self.model(batch).cpu().numpy()
          all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        assert all_features.shape[0]==len(self.image_dict)
        print(f"image embedding shape: {all_features.shape}")
        return all_features
    
class ExtractTextEmbedding:
    def __init__(self, text_dict):
        self.text_dict = text_dict
    
    def load_pretrain_model(self):
        # 生成文本模态embedding    
        self.device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 加载预训练的BERT tokenizer和model
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model.eval() # 设置模型为评估模式
    
    def get_embedding(self):
        self.load_pretrain_model()
        dataset = TextDataset(self.text_dict)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        all_features = []
        for batch in dataloader:
            batch = self.tokenizer(batch, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
                features = outputs[0][:, 0, :].cpu().numpy()  # 取CLS位置对应的特征向量
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        assert all_features.shape[0]==len(self.text_dict)
        print(f"text embedding shape: {all_features.shape}")
        return all_features