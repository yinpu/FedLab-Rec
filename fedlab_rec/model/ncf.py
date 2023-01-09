import torch

class NCF(torch.nn.Module):
    def __init__(self, user_num, item_num, predictive_factor=32, 
                 user_id_col="user_id", item_id_col="item_id", label_col="rating"):
        super().__init__()
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.label_col = label_col
        self.mlp_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.mlp_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.gmf_user_embeddings = torch.nn.Embedding(num_embeddings=user_num, embedding_dim=2*predictive_factor)
        self.gmf_item_embeddings = torch.nn.Embedding(num_embeddings=item_num, embedding_dim=2*predictive_factor)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(4*predictive_factor, 2*predictive_factor), 
            torch.nn.ReLU(), 
            torch.nn.Linear(2*predictive_factor, predictive_factor), 
            torch.nn.ReLU(),
            torch.nn.Linear(predictive_factor, predictive_factor//2), 
            torch.nn.ReLU()
            )
        self.output_logits = torch.nn.Linear(2*predictive_factor+predictive_factor//2, 1)
        self.loss_fct = torch.nn.BCELoss()

    def forward(self, data_dict):
        user_id, item_id = data_dict[self.user_id_col], data_dict[self.item_id_col]
        gmf_product = self.gmf_forward(user_id, item_id)
        mlp_output = self.mlp_forward(user_id, item_id)
        pred = torch.sigmoid(self.output_logits(torch.cat([gmf_product, mlp_output], dim=1)).view(-1))
        if self.label_col in data_dict:
            y = data_dict[self.label_col].float()
        else:
            y = None
        return pred, y


    def gmf_forward(self, user_id, item_id):
        user_emb = self.gmf_user_embeddings(user_id)
        item_emb = self.gmf_item_embeddings(item_id)
        return torch.mul(user_emb, item_emb)

    def mlp_forward(self, user_id, item_id):
        user_emb = self.mlp_user_embeddings(user_id)
        item_emb = self.mlp_item_embeddings(item_id)
        return self.mlp(torch.cat([user_emb, item_emb], dim=1))

    def cal_loss(self, data_dict):
        pred, y = self.forward(data_dict)
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