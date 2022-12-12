# FedLab-Rec

## 简介

基于FedLab的联邦学习与推荐算法结合的框架

框架实现以下功能

1. 推荐系统常用的数据分割方法
2. 多类型的推荐模型以及本地训练方式
3. 模型联邦化方法，常用部分模型参数上传
4. 推荐算法常用的评估函数
5. 可扩展对客户端采样、模型压缩、个性化联邦学习的研究
6. 支持配置文件传参数

不足点：

1. 仅支持单线程模拟实验，无法部署于真实系统
2. 仅支持CS架构下的同步和半异步模拟

## 用法

```bash
cd example/ncf
python pipeline --conf run.conf
```

## 主要模块介绍

### 设计的逻辑

- Dataset：数据集类，获取各个客户端的dataloader
- Trainer：模型本地训练、测试、验证的逻辑，根据不同的model设计不同训练方式，可实现召回、排序、多任务类型的trainer，还可以各种训练设定，例如earlystop、蒸馏学习等
- Client：接受Server参数、利用Dataset和Trainer进行本地更新，上传数据给Server
- Server：下发参数、客户端选择、参数聚合等

### Dataset

在推荐算法中，数据集形式一般为表格类型数据。

目前内置了RecDataset类，支持将pandas读入的Dataframe表格转换为pytorch神经网络的数据

```python
# fedlab_rec/utils/dataset.py
class RecDataset(FedDataset):
    def __init__(self, df, num_clients, client_ids_list=None, batch_size=64,
                split_ratio=(0.8, 0.1), user_col='user_id'):
      pass
    def get_dataloader(self, idx, mode='train'):
      pass
```

## Trainer

负责每个客户端本地更新的方法。

实现本地模型和训练方法的解耦，不同的Trainer支持不同的本地更新方式。

方便实现召回、排序、多任务等类型推荐模型和灵活的实验设定、验证方式

### Client

定义客户端的全流程，包括：接受参数、更新本地模型、上传参数

### Server

定义客户端的全流程，包括：下发参数、客户端选择、接受参数、更新全局模型等