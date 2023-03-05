import torch

class MyModel(torch.nn.Module):
    def __init__(self, user_num, item_num, 
                 item_text_embedding, item_image_embedding,
                 embedding_size=32, 
                 user_id_col="user_id", item_id_col="item_id", label_col="rating"):
        super().__init__()
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.label_col = label_col
        item_text_embedding = torch.FloatTensor(item_text_embedding)
        item_image_embedding = torch.FloatTensor(item_image_embedding)
        self.item_text_embedding = torch.nn.Embedding.from_pretrained(item_text_embedding, freeze=True)
        self.item_image_embedding = torch.nn.Embedding.from_pretrained(item_image_embedding, freeze=True)
        self.user_embedding = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=embedding_size)
        self.item_mlp = torch.nn.Sequential(
            torch.nn.Linear(768+2048, 256), 
            torch.nn.ReLU(), 
            torch.nn.Linear(256, embedding_size), 
            torch.nn.ReLU()
            )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2*embedding_size, 128), 
            torch.nn.ReLU(), 
            torch.nn.Linear(128, 32), 
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1), 
            )
        self.loss_fct = torch.nn.BCELoss()

    def forward(self, data_dict):
        user_id, item_id = data_dict[self.user_id_col], data_dict[self.item_id_col]
        user_emb = self.user_embedding(user_id)
        item_emb = torch.cat([self.item_image_embedding(item_id), self.item_text_embedding(item_id)], dim=-1)
        item_emb = self.item_mlp(item_emb)
        pred = torch.sigmoid(self.mlp(torch.cat([user_emb, item_emb], dim=1))).view(-1)
        if self.label_col in data_dict:
            y = data_dict[self.label_col].float()
        else:
            y = None
        return user_id, pred, y

    def cal_loss(self, data_dict):
        _, pred, y = self.forward(data_dict)
        return self.loss_fct(pred, y)

if __name__ == "__main__":
    ncf = NCF(10, 10)
    ncf.to(torch.device("cuda:0"))
    weights = ncf.state_dict()
    print("state...")
    import copy
    for k, v in weights.items():
        print(k)
        print(v.requires_grad)
        print(v.device)
        break